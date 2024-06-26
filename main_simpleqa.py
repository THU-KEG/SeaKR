from vllm import LLM, SamplingParams
import argparse
import re
from SEAKR.dataset import SingleQA
import pandas as pd
import json
import os
from tqdm import tqdm

def get_answer(output_text):
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

def _gen(prompts, model):
    prompt_list = []
    sample_params_list = []

    greedy_params = SamplingParams(**{
        "n": 1,
        "temperature":0.0,
        "top_p": 1.0,
        "max_tokens": 100,
        "logprobs": 0,
        "seed": 42,
        "stop": ["\n", "\n\n", "\nQuestion:", "\nContext"]
    })

    sample_params = SamplingParams(**{
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
        "max_tokens": 100,
        "n": 20,
        "logprobs": 0,
        "seed": 42,
        "stop": ["\n", "\n\n", "\nQuestion:", "\nContext"]
    })

    for p in prompts:
        prompt_list.append(p)
        prompt_list.append(p)
        sample_params_list.append(greedy_params)
        sample_params_list.append(sample_params)

    outputs = model.generate(prompts=prompt_list, sampling_params=sample_params_list)

    results = []
    for i in range(0, len(outputs), 2):
        greedy_output = outputs[i]
        samp_output = outputs[i+1]
        perplexity = greedy_output.uncertainty.get('perplexity', 1e3)
        energy_score = greedy_output.uncertainty.get('energy_score', 0)
        ln_entropy = samp_output.uncertainty.get('ln_entropy', 1e3)
        eigen_score = samp_output.uncertainty.get('eigen_score', 0)
        answer = greedy_output.outputs[0].text
        results.append({
            "answer": answer,
            "perplexity": perplexity,
            "energy_score": energy_score,
            "ln_entropy": ln_entropy,
            "eigen_score": eigen_score
        })
    return results

def gen_loop(prompts, output_file_name):
    results = _gen(prompts, model)

    final_greedy_params = SamplingParams(**{
        "n": 1,
        "temperature":0.0,
        "top_p": 1.0,
        "max_tokens": 20,
        "logprobs": 0,
        "seed": 42,
        "stop": ["\n", "\n\n", "\nQuestion:", "\nContext"]
    })

    for i, res in tqdm(enumerate(results), total=len(results), desc="Post Processing"):
        filtered_ans = get_answer(res['answer'])
        if filtered_ans is None or len(filtered_ans.strip()) == 0:
            prompt = prompts[i] + res['answer'] + " So the answer is "
            output = model.generate(prompts=[prompt], sampling_params=final_greedy_params, use_tqdm=False)
            filtered_ans = output[0].outputs[0].text
        res['answer'] = filtered_ans

    with open(output_file_name, 'w') as output_file:
        for result in results:
            output_file.write(json.dumps(result) + '\n')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=['nq', 'sq', 'tq'])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--selected_intermediate_layer", type=int, default=15)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = SingleQA(10)
    os.mkdir(args.output_dir)

    top10_data = pd.read_json(f"./data/singlehop_data/{args.dataset_name}_top10.json")
    direct_prompts = []
    for i, entry in top10_data.iterrows():
        direct_prompts.append(dataset(question=entry['question']))
    rag_prompts = []
    for i, entry in top10_data.iterrows():
        for doc_i, doc in enumerate(entry['ctxs']):
            prompt = dataset(question=entry['question'], docs=[doc['doc']])
            rag_prompts.append(prompt)

    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        selected_intermediate_layer=args.selected_intermediate_layer, #default 15
        eigen_alpha=1e-3, # default 1e-3,
        enable_prefix_caching=True,
        enforce_eager=True
    )

    gen_loop(direct_prompts, os.path.join(args.output_dir, 'direct.jsonl'))
    gen_loop(rag_prompts, os.path.join(args.output_dir, 'rag.jsonl'))