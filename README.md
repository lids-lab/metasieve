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


## MetaSieve Code 

## MPS-GNN Code ([main-code](https://github.com/francescoferrini/MPS-GNN))

## RelGT Code ([main-code](https://github.com/snap-stanford/relgt/tree/main))