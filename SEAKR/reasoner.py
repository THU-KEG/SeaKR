from typing import Union, List, Tuple
from .utils import LLMOutputWithUncertainty, Step, UncertaintyScore, StepStatus
from .retriever import BM25
from .dataset import Dataset
import logging
import re
import string
import spacy
nlp = spacy.load("en_core_web_sm")

from vllm import AsyncLLMEngine, SamplingParams
from vllm.sequence import Logprob
from vllm.utils import merge_async_iterators
import numpy as np

import os
logger = logging.getLogger(__name__)

class MultiHopReasoner:
    def __init__(self, qid: str, question:str, dataset:Dataset, llm_engine: AsyncLLMEngine, retriever: BM25, logger_dir: str=None, eigen_threshold: float=-6.0, prob_threshold: float=0.1) -> None:
        self.qid = qid
        self.question = question.replace('.', ' ')
        self.llm_engine = llm_engine
        self.retriever = retriever
        self.dataset = dataset
        self.set_logger(logger_dir)
        self.eigen_threshold = eigen_threshold
        self.prob_threshold = prob_threshold # for building query
        self.llm_call_times = 0
        self.running_steps: List[Step] = []
        self.searched_queries = set()
        self.doc_id_list = []
        self.docs = []
        self.final_step_answer = None
        self.final_read_answer = None
        self.sentence_solver = "max"

    def output_current_state(self):
        return {
            'qid': self.qid,
            'question': self.question,
            'eigen_threshold': self.eigen_threshold,
            'prob_threshold': self.prob_threshold,
            'llm_call_times': self.llm_call_times,
            'running_steps': self.running_steps,
            'searched_queries': self.searched_queries,
            'doc_id_list': self.doc_id_list,
            'final_step_answer': self.final_step_answer,
            'final_read_answer': self.final_read_answer
        }

    def set_logger(self, logger_dir):
        if logger_dir is None:
            logger = logging.getLogger(__name__)
        else:
            logger = logging.getLogger(f"logger_{self.qid}")
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(os.path.join(logger_dir, f"{self.qid}.log"))
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)

    async def call_llm(self, prompt:str, stop_tokens: List[str]) -> LLMOutputWithUncertainty:
        greedy_params = SamplingParams(**{
            "n": 1,
            "temperature":0.0,
            "top_p": 1.0,
            "max_tokens": 100,
            "logprobs": 0,
            "seed": 42,
            "stop": stop_tokens
        })
        sample_params = SamplingParams(**{
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
            "max_tokens": 100,
            "n": 20,
            "logprobs": 0,
            "seed": 42,
            "stop": stop_tokens,
        })
        
        generators = []
        request_ids = [f"{self.qid}_{self.llm_call_times}_greedy", f"{self.qid}_{self.llm_call_times}_sample"]
        generators.append(self.llm_engine.generate(inputs=prompt, sampling_params=greedy_params, request_id=request_ids[0]))
        generators.append(self.llm_engine.generate(inputs=prompt, sampling_params=sample_params, request_id=request_ids[1]))
        result_generator = merge_async_iterators(*generators)
        final_res_batch = [None] * 2
        async for i, res in result_generator:
            final_res_batch[i] = res
        for r_id in request_ids:
            await self.llm_engine.abort(r_id)

        self.llm_call_times += 1
        greedy_outputs, sampl_outputs = final_res_batch[0], final_res_batch[1]
        perplexity = greedy_outputs.uncertainty.get("perplexity", 1e3)
        energy_score = greedy_outputs.uncertainty.get("energy_score", 0)
        eigen_score = sampl_outputs.uncertainty.get("eigen_score", 0)
        ln_entropy = sampl_outputs.uncertainty.get("ln_entropy", 1e3)

        return LLMOutputWithUncertainty(
            greedy_response=greedy_outputs.outputs[0].text,
            sample_responses=[cpl.text for cpl in sampl_outputs.outputs],
            uncertainty=UncertaintyScore(
                perplexity=perplexity, logprobs=greedy_outputs.outputs[0].logprobs,
                energy_score=energy_score, eigen_score=eigen_score, ln_entropy=ln_entropy,
            )
        )


    def _check_is_token_punct(self, token):
        token_doc = nlp(token.strip())
        return all(t_w.is_punct for t_w in token_doc)
    
    def _restruct_tokens(self, step: Step):
        sentences = [sent.text.strip() for sent in nlp(step.content).sents]
        logprob_list: List[Logprob] = [list(one_token_dict.values())[0] for one_token_dict in step.logprobs]

        def remove_special_characters(input_string):
            pattern = re.compile(r'[^a-zA-Z0-9]')
            cleaned_string = re.sub(pattern, '', input_string)
            return cleaned_string

        cleaned_log_problist = []
        for lp in logprob_list:
            cleaned_tokens = remove_special_characters(lp.decoded_token)
            if cleaned_tokens:
                lp.decoded_token = cleaned_tokens
                cleaned_log_problist.append(lp)

        reconstructed_tokens = []
        token_idx = 0
        
        for sentence in sentences:
            doc = nlp(sentence)
            sentence_tokens = []
            i_word = 0
            while i_word < len(doc):
                word = doc[i_word]
                if len(remove_special_characters(word.text)) == 0:
                    i_word += 1
                    continue
                full_word_text = word.text
                clean_word_text = remove_special_characters(word.text)
                word_tokens = []
                while clean_word_text and token_idx < len(cleaned_log_problist):
                    token_logprob = cleaned_log_problist[token_idx]
                    token_str = token_logprob.decoded_token
                    if clean_word_text.startswith(token_str.strip()):
                        word_tokens.append(token_logprob)
                        clean_word_text = clean_word_text[len(token_str.strip()):]
                    elif token_str.startswith(clean_word_text) and i_word + 1 < len(doc):
                        # Language model token is longer than spacy word: Combine the current word with the next word
                        i_word += 1
                        next_word = doc[i_word]
                        full_word_text += next_word.text
                        clean_word_text += remove_special_characters(next_word.text)
                        continue
                    token_idx += 1
                sentence_tokens.append({'word': full_word_text, 'tokens': word_tokens})
                i_word += 1
            reconstructed_tokens.append(sentence_tokens)
        
        if len(reconstructed_tokens[-1]) > 0 and len(reconstructed_tokens[-1][-1]['tokens']) == 0:
            # we calculate the embedding in vllm, so last token might don't have prob
            reconstructed_tokens[-1] = reconstructed_tokens[-1][:-1]
        return reconstructed_tokens



    def _build_query(self, direct_failed_step: Step) -> str:
        reconstructed_tokens = self._restruct_tokens(direct_failed_step)

        for sentence in reconstructed_tokens:
            # get the max
            sure_words = []
            for word_with_logprobs in sentence:
                token_probs: List[Logprob] = word_with_logprobs['tokens']
                if len(token_probs) == 0:
                    continue
                prob_list = [1-np.exp(tp.logprob) for tp in token_probs]
                max_prob = np.max(prob_list)
                if word_with_logprobs['word'] in self.question or (
                    self.running_steps and word_with_logprobs['word'] in self.running_steps[-1].content) or (
                    max_prob <= self.prob_threshold):
                    sure_words.append(word_with_logprobs['word'])
        if sure_words:
            return " ".join(sure_words)

        return ""
    
    def _filter_words(self, doc, suspicious_word_indices={}):
        if isinstance(doc, str):
            doc = nlp(doc.strip())
        desired_pos_tags = {'NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM', 'ADV'}
        filtered_words = [word.text for i, word in enumerate(doc) if i not in suspicious_word_indices and word.pos_ in desired_pos_tags]
        filtered_words = [fw for fw in filtered_words if fw not in [ 'thus', 'answer', 'is', 'so']]
        return " ".join(filtered_words)
    
    def _retrieve(self, step):
        query = ""
        if step.content.strip():
            # try:
            query = self._build_query(step)  # 尝试生成查询
        query = self._filter_words(query)
        if not query.strip():
            query = self._filter_words(self.question)
        if query in self.searched_queries:
            self.logger.debug(f"query already searched, use full question")
            raise ValueError(f"Retrieval Duplicated")
        self.searched_queries.add(query)
        self.logger.debug(f"Search for: {query}")
        try:
            docs_ids, docs = self.retriever.retrieve(queries=[query], topk=3, max_query_length=64)
        except Exception as e:
            raise ValueError(f"Retrieval failed: {e}")
        return query, docs_ids, docs
               
    async def rag(self) -> Step:
        last_step = self.running_steps.pop()
        assert last_step.status == StepStatus.DIRECT_FAILED
        try:
            query, docs_ids, docs = self._retrieve(last_step)
        except ValueError as e:
            self.logger.error(f"An error occurred: {e}")
            return None
        read_outputs, read_ids, read_docs = [], [], []
        for d_id, doc in zip(docs_ids[0], docs[0]):
            if d_id in self.doc_id_list:
                continue
            self.logger.debug(f"Answer based on doc {d_id}")
            curr_doc_list = self.docs + [doc]
            self.logger.debug("="*100)
            self.logger.debug(doc)
            prompt, stop_tokens = self.prepare_llm_input(
                question=self.question, cot_step=self.running_steps, docs=curr_doc_list
            )
            curr_step_output = await self.call_llm(prompt=prompt, stop_tokens=stop_tokens)
            self.logger.debug("="*100)
            self.logger.debug(curr_step_output.greedy_response)
            self.logger.debug(curr_step_output.uncertainty.eigen_score)
            # print(doc)
            # print(curr_step_output.uncertainty.eigen_score)
            read_ids.append(d_id)
            read_docs.append(doc)
            read_outputs.append(curr_step_output)
        
        def sort_key(item):
            step_output = item[2]
            if step_output.greedy_response.strip():
                return step_output.uncertainty.eigen_score
            else:
                return float('inf')

        if len(read_ids)==0:
            return None
            
        combined = list(zip(read_ids, read_docs, read_outputs))
        combined.sort(key=sort_key)
        read_ids, read_docs, read_outputs = zip(*combined)

        first_doc_id = read_ids[0]
        first_doc = read_docs[0]
        self.logger.debug(f"doc {first_doc_id} has the best answer")
        best_possible_step = read_outputs[0]
        self.doc_id_list.append(first_doc_id)
        self.docs.append(first_doc)

        self.logger.debug(f"current doc list: {self.doc_id_list}")

        return Step(
            status=StepStatus.RAG_FINISHED,
            search_query=query,
            best_docid=first_doc_id,
            content=best_possible_step.greedy_response, 
            score=best_possible_step.uncertainty.eigen_score
        )
    
    async def read_all_docs(self):
        self.logger.debug(f"read all docs and generate all steps one-time. Doc_ids: {self.doc_id_list}")
        prompt, stop_tokens = self.prepare_llm_input(
            question=self.question, docs=self.docs, is_final=True
        )
        output = await self.call_llm(prompt=prompt, stop_tokens=stop_tokens)
        
        self.final_read_answer_full = output.greedy_response

        self.logger.debug(f"Read Result: {self.final_read_answer_full}")

        checked_final_read_answer = self._check_final_answer(self.final_read_answer_full)

        if checked_final_read_answer is None:
            # force generate answer
            prompt += " So the answer is "
            stop_tokens.append("\n")
            output = await self.call_llm(prompt=prompt, stop_tokens=stop_tokens)
            checked_final_read_answer = output.greedy_response
            self.final_read_answer_full += f" So the answer is {checked_final_read_answer}"
        
        self.final_read_answer = checked_final_read_answer
        self.final_read_answer_score = output.uncertainty.eigen_score

    def compare_answer(self):
        # check if unknown
        def check_none_or_unknown(short_answer):
            if short_answer is None:
                return True
            if "unknown" in short_answer.lower():
                return True
            return False
        
        if check_none_or_unknown(self.final_step_answer):
            self.logger.info(f"No valid final Step answer, choose final read")
            self.final_answer = self.final_read_answer
            return
        if check_none_or_unknown(self.final_read_answer):
            self.logger.info(f"Final read gives an unknown answer, choose final step")
            self.final_answer = self.final_step_answer
            return
        # compare eigen score
        if self.final_step_score < self.final_read_answer_score:
            self.logger.info(f"Final step answer Better {self.final_step_score} < {self.final_read_answer_score}")
            self.final_answer = self.final_step_answer
        else:
            self.logger.info(f"Final read answer Better {self.final_step_score} > {self.final_read_answer_score}")
            self.final_answer = self.final_read_answer

    def _check_final_answer(self, output_text: str):
        if not output_text:
            return None
        pattern = r'the answer is(?:\s*:\s*)?(.*?)[,.]'
        match = re.search(pattern, output_text.lower(), re.DOTALL)
        if match:
            return match.group(1).strip()
        pattern2 = r'[.?!]\s*([^?!]*?)\s+is the answer\b'
        match = re.search(pattern2, output_text.lower(), re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

    
    def prepare_llm_input(self, question: str, cot_step: List[Step]=None, docs: List[str]=None, is_final: bool=False) -> Tuple[str, List[str]]:
        stop_tokens = ["\n\n", "\nQuestion:", "\nContext"]
        if cot_step:
            cot_step_content = [c.content for c in cot_step]
            stop_tokens += ["\n"]
        else:
            cot_step_content = None
        if not is_final:
        #     stop_tokens += ["\n\n", "\nQuestion:", "\nContext"]
        # else:
            stop_tokens += [". "]
        prompt = self.dataset(question=question, cot_steps=cot_step_content, docs=docs)
        return prompt, stop_tokens
    
    async def answer_direct(self) -> Step:
        prompt, stop_tokens = self.prepare_llm_input(
            question=self.question,
            cot_step=self.running_steps,
            docs=self.docs
        )
        direct_output = await self.call_llm(prompt=prompt, stop_tokens=stop_tokens)
        candidate_next_step = Step(
            status=StepStatus.DIRECT_GENERATED,
            content=direct_output.greedy_response, score=direct_output.uncertainty.eigen_score,
            logprobs=direct_output.uncertainty.logprobs
        )
        self.logger.info(f"Direct Output: {candidate_next_step.content}")
        # check direct
        if candidate_next_step.content.strip() == "":
            self.logger.debug("Direct Answer Failed, Empty response")
            candidate_next_step.status = StepStatus.DIRECT_FAILED
        elif candidate_next_step.score > self.eigen_threshold:
            candidate_next_step.status = StepStatus.DIRECT_FAILED
            self.logger.debug(f"Direct Answer Failed, Low eigen score: {candidate_next_step.score:.2f}")
        else:
            candidate_next_step.status = StepStatus.DIRECT_SUCCESS
            self.logger.debug(f"Valid output: {candidate_next_step.content}, Eigen: {candidate_next_step.score:.2f}")
        return candidate_next_step
    

    def check_final_step(self):
        final_step_answer = self._check_final_answer(self.running_steps[-1].content)
        if final_step_answer:
            self.final_step_answer = final_step_answer
            self.logger.debug(f"Last Step Generated")
            self.last_step_score = self.running_steps[-1].score
            return True
        
    def filter_last_step(self):
        last_step_content = self.running_steps[-1].content
        doc = nlp(last_step_content.strip())
        filtered_sentences = []
        for sentence in doc.sents:
            if "the answer is" not in sentence.text.lower() and "is the answer" not in sentence.text.lower():
                filtered_sentences.append(sentence.text)
        last_step_content_filterd = ' '.join(filtered_sentences)

        if len(last_step_content_filterd.strip()) == 0:
            self.running_steps = self.running_steps[:-1]
        else:
            self.running_steps[-1].content = last_step_content_filterd
        

    async def read_all_steps(self):
        self.filter_last_step()
        step_contents = [s.content for s in self.running_steps]
        prompt, stop_tokens = self.prepare_llm_input(
            question=self.question,
            docs=step_contents
        )
        stop_tokens.append("\n")
        stop_tokens.remove('. ')

        read_steps_output = await self.call_llm(prompt=prompt, stop_tokens=stop_tokens)

        final_step_answer = self._check_final_answer(read_steps_output.greedy_response)

        if not final_step_answer:
            prompt += " So the answer is "
            force_read_steps_output = await self.call_llm(prompt=prompt, stop_tokens=stop_tokens)
            final_step_answer = force_read_steps_output.greedy_response

        self.final_step_answer = final_step_answer
        self.final_step_score = read_steps_output.uncertainty.eigen_score

    
    async def solve(self, max_reasoning_steps = 10, max_docs=3):
        self.logger.info(self.question)
        while True:
            self.logger.info(f"\n{'='*50}\nStep {len(self.running_steps) + 1}")
            candidate_next_step = await self.answer_direct()
            self.running_steps.append(candidate_next_step)
            if self.check_final_step():
                break
            if self.running_steps[-1].status == StepStatus.DIRECT_FAILED:
                rag_candidate_next_step = await self.rag()
                if rag_candidate_next_step is None:
                    self.logger.info(f"No More Useful docs")
                    break
                else:
                    self.logger.info(f"RAG Output: {rag_candidate_next_step.content}")
                self.running_steps.append(rag_candidate_next_step)
            if self.check_final_step():
                break
            if len(self.running_steps) >= max_reasoning_steps or \
                len(self.docs) >= max_docs:
                break

        # regenerate_last
        await self.read_all_steps()

        self.logger.info(f"\n{'='*50}\nQuestion: {self.question}")
        self.logger.debug(f"All Steps: {[step.content for i, step in enumerate(self.running_steps)]}")
        await self.read_all_docs()
        self.compare_answer()
        
        self.logger.info(f"Final Read Answer Full: {self.final_read_answer_full}")

        output_data = {
            "qid": self.qid,
            "question": self.question,
            "Retrieval Times": len(self.doc_id_list),
            "Call LLM Times": self.llm_call_times,
            "Final Step Answer": self.final_step_answer,
            "Final Read Answer": self.final_read_answer,
            "Final Answer": self.final_answer,
        }

        for key, value in output_data.items():
            self.logger.info(f"{key}: {value}")

        return output_data