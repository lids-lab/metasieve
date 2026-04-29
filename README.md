# MetaSieve: SQL-Based Metapath Selection for Relational Deep Learning

## Environment Setup

**Requirements:** Linux x86_64, NVIDIA GPU with CUDA driver ≥ 12.1, [Conda](https://docs.conda.io/en/latest/miniconda.html). macOS and Windows are not supported (several dependencies ship Linux-only wheels).

Verify your CUDA driver:
```bash
nvidia-smi
```
The `CUDA Version` shown must be ≥ 12.1.

### Install

From the repo root:
```bash
conda env create -f environment.yml
conda activate metasieve```

First run takes 10–20 minutes (downloads ~3 GB of PyTorch + CUDA libraries).

### Verify

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch_scatter, torch_sparse, torch_cluster, pyg_lib; print('PyG OK')"
python -c "from relbench.datasets import get_dataset; print('RelBench OK')"
```

All three should succeed and the first should print `True`.

### Updating

If `environment.yml` changes, recreate from scratch:
```bash
conda deactivate
conda env remove -n metasieve
conda env create -f environment.yml
```