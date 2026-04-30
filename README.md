# MetaSieve: SQL-Based Metapath Selection for Relational Deep Learning

## Environment Setup

**Requirements:** Linux x86_64, NVIDIA GPU, and [Conda](https://docs.conda.io/en/latest/miniconda.html). 

Verify your CUDA driver:
```bash
nvidia-smi
```
The `CUDA Version` shown must be ≥ 12.1.

### Install

From the repo root:
```bash
conda env create -f environment.yml
conda activate metasieve
```
First run takes 10–20 minutes (downloads ~3 GB of PyTorch + CUDA libraries).

### Verify

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch_scatter, torch_sparse, torch_cluster, pyg_lib; print('PyG OK')"
python -c "from relbench.datasets import get_dataset; print('RelBench OK')"
```

All three should succeed and the first should print `True`.


## MetaSieve Code 

## MPS-GNN Code (configured for [Relbench](https://relbench.stanford.edu/) datasets)

## RelGT Code ([main-code](https://github.com/snap-stanford/relgt/tree/main))