from vllm import AsyncLLMEngine, AsyncEngineArgs
from SEAKR.dataset import get_dataset
from SEAKR.reasoner import MultiHopReasoner

from transformers import AutoTokenizer

from SEAKR.retriever import BM25
import warnings
from elasticsearch.exceptions import ElasticsearchDeprecationWarning
warnings.simplefilter('ignore', ElasticsearchDeprecationWarning)

import asyncio
import aiofiles
from tqdm.asyncio import tqdm
import json
import os
import pickle
from dataclasses import dataclass

@dataclass
class HyperParams:
    eigen_threshold: float 
    prob_threshold: float
    max_reasoning_steps: int
    max_docs: int

error_count = 0
async def run_one_question(semaphore, entry, dataset_obj, llm_engine, retriever, logger_dir, finished_file, failed_file, lock, progress_bar, hyperparams: HyperParams):
    global error_count
    async with semaphore: 
        reasoner = MultiHopReasoner(
            qid = entry['qid'],
            question=entry['question'],
            dataset=dataset_obj,
            llm_engine=llm_engine,
            retriever=retriever,
            logger_dir=logger_dir,
            eigen_threshold=hyperparams.eigen_threshold,
            prob_threshold=hyperparams.prob_threshold
        )
        try:
            output_data = await asyncio.wait_for(
                reasoner.solve(
                    max_reasoning_steps=hyperparams.max_reasoning_steps,
                    max_docs=hyperparams.max_docs
                ),
                timeout=20*60  # 超时时间，单位为秒
            )
            output_data['ground_truth'] = entry['answer']
            reasoner.logger.info(f"\nGround Truth: {entry['answer']}")
            async with lock:
                await finished_file.write(json.dumps(output_data) + '\n')
            progress_bar.update(1)
        except Exception as e:
            reasoner.logger.error(e)
            if len(reasoner.running_steps) > 0:
                current_state = reasoner.output_current_state()
                parent_dir = os.path.dirname(logger_dir)
                storage_dir = os.path.join(parent_dir, "reasoning_ckpt")
                os.makedirs(storage_dir, exist_ok=True)
                pickle_file_name = os.path.join(storage_dir, f"{entry['qid']}.pkl")
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(current_state, f)
                reasoner.logger.info(f"States Saved to {pickle_file_name}")
            progress_bar.update(1)
            async with lock:
                await failed_file.write(json.dumps(
                    {
                        "qid": entry['qid'],
                        "error": str(e)
                    }
                )+"\n")
            async with lock:
                error_count += 1
                if error_count >= 10:
                    for task in asyncio.all_tasks():
                        task.cancel()
                    raise Exception("Error limit reached, stopping all tasks")


async def run_full(dataset_list, dataset_obj, llm_engine, retriever, save_dir, hyperparams: HyperParams, max_workers=50):
    logger_dir = os.path.join(save_dir, 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    finished_filename = os.path.join(save_dir, "results.jsonl")
    failed_filename = os.path.join(save_dir, "failed.jsonl")
    semaphore = asyncio.Semaphore(max_workers)  # 控制最大并发数

    lock = asyncio.Lock()
    async with aiofiles.open(finished_filename, mode='a') as finished_file, \
               aiofiles.open(failed_filename, mode='a') as failed_file:
        progress_bar = tqdm(total=len(dataset_list), desc="Processing dataset")
        tasks = [run_one_question(semaphore, entry, dataset_obj, llm_engine, retriever, logger_dir, finished_file, failed_file, lock, progress_bar, hyperparams) for entry in dataset_list]
        await asyncio.gather(*tasks)
        progress_bar.close()

async def main(args):
    dataset_obj = get_dataset(args.dataset_name, args.n_shot)
    dataset_list = dataset_obj.load_data()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    retriever = BM25(
        tokenizer=tokenizer, 
        index_name="wiki", 
        engine="elasticsearch",
        port=args.retriever_port,
    )

    engine_args = AsyncEngineArgs(
        model=args.model_name_or_path,
        served_model_name=args.served_model_name,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        selected_intermediate_layer=args.selected_intermediate_layer, #default 15
        eigen_alpha=args.eigen_alpha, # default 1e-3,
        worker_use_ray=True,
        disable_log_requests=True,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enforce_eager=True
    )
    
    hyperparams = HyperParams(eigen_threshold=args.eigen_threshold, prob_threshold=args.prob_threshold,
                              max_reasoning_steps=args.max_reasoning_steps, max_docs=args.max_docs)

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    await run_full(
        dataset_list=dataset_list,
        dataset_obj=dataset_obj,
        llm_engine=llm_engine,
        retriever=retriever,
        save_dir=args.save_dir,
        hyperparams=hyperparams
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the model with provided arguments.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")
    parser.add_argument("--retriever_port", required=True, help="Port of Elastic Search Service.")
    parser.add_argument("--n_shot", type=int, default=10, help="Number of examples per task.")
    parser.add_argument("--model_name_or_path", required=True, help="Pre-trained model name or path.")
    parser.add_argument("--served_model_name", required=True, help="Model name for serving.")
    parser.add_argument("--selected_intermediate_layer", type=int, default=15, help="Selected layer for processing.")
    parser.add_argument("--eigen_alpha", type=int, default=1e-3, help="eigen alpha to compute eigen score")
    parser.add_argument("--eigen_threshold", type=float, default=-6.0, help="Threshold for eigen score.")
    parser.add_argument("--prob_threshold", type=float, default=0.1, help="Log probability threshold to form query.")
    parser.add_argument("--max_reasoning_steps", type=int, default=10, help="Maximum reasoning steps.")
    parser.add_argument("--max_docs", type=int, default=5, help="Maximum documents to retrieve.")
    parser.add_argument("--save_dir", required=True, help="Directory to save the results.")
    args = parser.parse_args()

    if os.path.exists(args.save_dir):
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        args.save_dir = f"{args.save_dir}_{timestamp}"
        
    os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
    asyncio.run(main(args))
