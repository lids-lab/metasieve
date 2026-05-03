# MetaSieve: SQL-Based Metapath Selection

MetaSieve selects informative edge types per hop using NMI-based scoring, LCB pruning, and GMM clustering, producing a per-hop sampling config (`.json`) for use with `NeighborLoader`.

---

## Scripts

### MetaSieve pipeline

| Script | What it does |
|---|---|
| `generate_statistics.py` | Walks the graph hop-by-hop and computes metapath statistics for seed nodes, in parallel batches|
| `compute_nmi_batches.py` | Computes the NMI and cost based score between each candidate extension statistic and the task labels, in parallel batches |
| `lcb_score_generation.py` | Applies a lower-confidence-bound to the score and coverage to penalize high-variance estimates |
| `clustering_candidates.py` | Runs GMM clustering on Q-scores to separate informative candidate extensions from uninformative candidate extensions |
| `generate_rules.py` | Converts clustered candidates into per-hop pruning rules |
| `generate_sampling_config.py` | Translates the pruning rules into a `NumNeighbors`-compatible `.json` sampling config |

### Supporting modules

| Script | What it does |
|---|---|
| `candidate_generation_utils.py` | Shared utilities for statistics computation and candidate tracking |
| `feature_components.py` | Definitions for the components used to generate the statistics |
| `embeddings.py` | Text embedding wrapper (used when building the graph) |

### GNN training

| Script | What it does |
|---|---|
| `train_stepwise.py` | Trains HeteroGraphSAGE or HGT on a Relbench task using a given sampling config |
| `model.py` | HeteroGraphSAGE and HGT model definitions |
| `hgt_gnn.py` | HGT layer implementation |

---

## Running the pipeline

All six MetaSieve steps are combined into a single script:

```bash
bash run_pipeline.sh
```

### Parameters in `run_pipeline.sh`

| Variable | What it controls | Example |
|---|---|---|
| `DATASET` | Relbench dataset name | `rel-stack` |
| `TASK` | Task name | `user-badge` |
| `DELTA` | LCB pruning threshold | `0.1 ~ 0.4` |
| `HOPS_SUFFIX` | Label for the hop depth used in output filenames | `3hops` |
| `NUM_LAYERS` | Number of hops in the output sampling config | `3` |
| `BASE_NEIGHBORS` | Fanout assigned to selected edge types in the config | `64` |
| `MAX_HOP` | Maximum hop depth explored during statistics generation | `3` |
| `HOPS_TO_EXPORT` | Which hops to score and include (space-separated) | `"1 2"` |
| `N_BATCHES` | Number of batches for parallelized statistics generation | `32` |
| `AGG_WORKERS` | CPU workers for the statistics generation step | `4` |
| `SCORE_WORKERS` | CPU workers for the NMI scoring step | `16` |
| `LABEL_MODE` | `classification` or `regression` | `classification` |


---

## Training a GNN

To train HeteroGraphSAGE or HGT with a sampling config:

```bash
bash run_training.sh
```

### Parameters in `run_training.sh`

| Variable | What it controls | Example |
|---|---|---|
| `DATASET_NAME` | Relbench dataset name | `rel-ratebeer` |
| `TASK_NAME` | Task name | `user-churn` |
| `TASK_KIND` | `classification` or `regression` | `classification` |
| `NUM_LAYERS` | Number of GNN layers / hop depth | `3` |
| `EPOCHS` | Maximum training epochs | `20` |
| `BATCH_SIZE` | Training batch size | `128` |
| `EARLY_STOP_PATIENCE` | Epochs without improvement before stopping | `10` |
| `MAX_STEPS_PER_EPOCH` | Cap on gradient steps per epoch | `1500` |
| `FANOUT` | Neighbors per edge type per hop (for random sampling) | `64` |
| `MODEL_TYPE` | `hgs` (HeteroGraphSAGE) or `hgt` (HGT) | `hgt` |
| `ATTENTION_HEAD` | Number of attention heads (HGT only) | `4` |
| `SAMPLING_CFG_PATH` | Path to sampling config — leave empty for random sampling | see below |

**Sampling mode** is controlled entirely by `SAMPLING_CFG_PATH`:

- **Random sampling** — leave `SAMPLING_CFG_PATH` empty:
  ```bash
  SAMPLING_CFG_PATH=""
  ```
- **MetaSieve sampling** — point to a `.json` config:
  ```bash
  SAMPLING_CFG_PATH="./outputs/rel-ratebeer/user-churn/user_churn_pruned_3hops_0.1.json"
  ```
- **MPS-GNN sampling** — point to a `.py` config:
  ```bash
  SAMPLING_CFG_PATH="./outputs/rel-ratebeer/user-churn/user_churn_3hops.py"
  ```

---

## Output directory structure

All pipeline outputs are written to `outputs/<dataset>/<task>/`:

```
outputs/
└── <dataset>/
    └── <task>/
        ├── <dataset>_<task>_pipeline_timing.txt                 # per-step timing report
        ├── log_count/train/batch_<N>/hop_<N>/<metapath>.csv    # count statistics
        ├── log_rate/train/batch_<N>/hop_<N>/<metapath>.csv     # rate statistics
        ├── mi_summary_per_aggregate_with_lcb_<hops>_<delta>.csv  # NMI + LCB scores
        ├── extension_q_scores_lcb_<hops>_<delta>.csv           # Q-scores per candidate extension
        ├── extension_dead_paths_lcb_<hops>_<delta>.csv         # dead extensions
        ├── gmm_good_global_lcb_<hops>_<delta>.csv              # GMM informative cluster
        ├── gmm_bad_global_lcb_<hops>_<delta>.csv               # GMM uninformative cluster
        ├── rules_<hops>_<delta>.csv                             # final pruning rules
        └── <task_prefix>_pruned_<N>hops_<delta>.json           # ← MetaSieve sampling config
```

The bulk of the disk usage is the `log_count/` and `log_rate/` trees — one `.csv` per candidate extension per batch per hop. These are the raw statistics produced by `generate_statistics.py`. All downstream scoring steps read from these files and write compact `.csv` summaries at the task root. The final `.json` is the only file needed for training.

---

## Pre-generated configs

The `outputs/` directory already contains pre-generated configs for all datasets and tasks used in the paper:

```
outputs/
├── rel-avito/     ad-ctr, user-clicks, user-visits
├── rel-f1/        driver-dnf, driver-position, driver-top3
├── rel-ratebeer/  beer-churn, user-churn, user-count
├── rel-stack/     post-votes, user-badge, user-engagement
└── rel-trial/     site-success, study-adverse, study-outcome
```

Each task folder contains:
- A `.json` MetaSieve sampling config (generated by `run_pipeline.sh`)
- A `.py` MPS-GNN sampling config (generated by the MPS-GNN pipeline — see `../MPS-GNN`)

You can use these directly with `run_training.sh` without re-running either pipeline.
