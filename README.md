# SeaKR

## Getting Started

### Installing Environment

```bash
conda create -n seakr python=3.10
conda activate seakr
pip install beir==1.0.1 spacy==3.7.2 aiofiles tenacity
python -m spacy download en_core_web_sm
```

We modify the vllm to get the uncertainty measures.

```bash
cd vllm_uncertainty
pip install -e .
```

### Download dataset

For multihop QA datasets, we use the same files as [dragin](https://github.com/oneal2000/DRAGIN). You can download and unzip it into the `data/multihop_data` folder. We provide a packed multihop data files here: [multihop_data.zip](https://drive.google.com/file/d/1xDqaPa8Kpnb95l7nHpwKWsBQUP9Ck7cn/view?usp=sharing)

For simple QA datasets, we use the same files as [DPR](https://github.com/facebookresearch/DPR). packed files are [singlehop_data.zip](https://drive.google.com/file/d/1T4ZRHZb4C6akZdMHIP1MUgNzSamCBn7N/view?usp=sharing). You can download and unzip it into the `data/singlehop_data` folder. 

### Prepare Retriever

Followed by [dragin](https://github.com/oneal2000/DRAGIN). Use the Wikipedia dump and elastic search to build the retriever

#### download wikipedia dump

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

#### run Elasticsearch service

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
```

#### build the index

```bash
python build_wiki_index.py --data_path $YOUR_WIKIPEDIA_TSV_PATH --index_name wiki --port $YOUR_ELASTIC_SERVICE_PORT
```

## Run SeaKR

### 2WikiHop
```bash
python main.py \
    --n_shot 10 \
    --retriever_port $YOUR_ELASTIC_SERVICE_PORT \
    --dataset_name twowikihop \
    --eigen_threshold -6.0 \
    --save_dir "outputs/twowikihop" \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --served_model_name llama2-7b-chat \
    --max_reasoning_steps 7 \
    --max_docs 5
```

### HotpotQA
```bash
python main_multihop.py \
    --n_shot 10 \
    --retriever_port $YOUR_ELASTIC_SERVICE_PORT \
    --dataset_name hotpotqa \
    --eigen_threshold -6.0 \
    --save_dir "outputs/hotpotqa" \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --served_model_name llama2-7b-chat \
    --max_reasoning_steps 7 \
    --max_docs 5
```

### IIRC
```bash
python main_multihop.py \
    --n_shot 10 \
    --retriever_port $YOUR_ELASTIC_SERVICE_PORT \
    --dataset_name iirc \
    --eigen_threshold -6.0 \
    --save_dir "outputs/iirc" \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --served_model_name llama2-7b-chat \
    --max_reasoning_steps 7 \
    --max_docs 5
```

## Evaluate

We provide a jupyter notebook `` to do evaluation. You just need to replace the output jsonline file name with your own output.