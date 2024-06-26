# SeaKR

## Getting Started

### Installing Environment

```bash
conda create -n seakr python=3.10
conda activate seakr
```

We modify the vllm to get the uncertainty measures.

```bash
cd vllm_uncertainty
pip install -e .
```