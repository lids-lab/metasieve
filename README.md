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

## MPS-GNN Code
The original MPS-GNN [paper](https://openreview.net/pdf?id=8Q4qxe9a9Z), [code](https://github.com/francescoferrini/MPS-GNN) (Ferrini et al., TMLR 2025) was designed to search for informative metapaths in small non-temporal relational databases (EICU, MONDIAL, ErgastF1).

Relbench datasets are large **heterogeneous** and **temporal** graphs. In order to bridge this gap, the official MPS-GNN code was adapted to work with the Relbench datasets and tasks. The changes are detailed in the [MPS-GNN](./MPS-GNN) folder.

## RelGT Code ([main-code](https://github.com/snap-stanford/relgt/tree/main))