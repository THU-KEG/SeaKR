# SeaKR

## Getting Started

### Install Environment

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

### Prepare Retriever

Followed by [dragin](https://github.com/oneal2000/DRAGIN). Use the Wikipedia dump and elastic search to build the retriever

#### Download Wikipedia dump

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

#### Run Elasticsearch service

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

## Run SeaKR on Multihop QA

For multihop QA datasets, we use the same files as [dragin](https://github.com/oneal2000/DRAGIN). You can download and unzip it into the `data/multihop_data` folder. We provide a packed multihop data files here: [multihop_data.zip](https://drive.google.com/file/d/1xDqaPa8Kpnb95l7nHpwKWsBQUP9Ck7cn/view?usp=sharing).
We use an asynchronous reasoning engine to accelerate multi hop reasoning.

### 2WikiHop
```bash
python main_multihop.py \
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

### Evaluate

We provide a jupyter notebook `eval_multihop.ipynb` to do evaluation. You just need to replace the output jsonline file name with your own output.


## Run SeaKR on Single QA

The original files are from [DPR](https://github.com/facebookresearch/DPR). We provide a packed version containing top 10 retrieved documents [singlehop_data.zip](https://drive.google.com/file/d/1hn4Om_KkIGJpgG2wJjUu1mpPv9oq8M6G/view?usp=sharing). You can download and unzip it into the `data` folder. 

```bash
python main_simpleqa.py \
    --dataset_name tq \
    --model_name_or_path $YOUR_MODEL_CHECKPOINT_PATH \
    --selected_intermediate_layer 15 \
    --output_dir $OUTPUT_DIR
```

You can evaluate the output in the `eval_singlehop.ipynb` notebook