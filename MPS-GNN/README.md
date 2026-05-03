# How to run MPS-GNN on Relbench datasets to generate sampling configs

## Adapting MPS-GNN for Relbench

The changes made to the original MPS-GNN code in order to run the metapath searching algorithm for Relbench are listed below:

1. **Sample seed nodes.** A small, balanced subset of train and validation seed nodes is drawn from the task tables (controlled by `N_POS_TRAIN`, `N_NEG_TRAIN`, `N_POS_VAL`, `N_NEG_VAL`). This keeps the search feasible.

2. **Sample a single temporal subgraph.** A single `NeighborLoader` pass is run over all train + val seeds jointly using disjoint temporal sampling, producing the subgraph used by the search.

3. **Run the MPS search.** The subgraph is passed to `run_mps_metapath_search()`, which performs a beam search over candidate metapaths. The full search trace and the final ranked list of metapaths are written to `search_results/<dataset>/mps_search_<task>.log` and a clean ranked-only `.txt` file.

4. **Decode the results.** The ranked `.txt` file is decoded into human-readable edge type names and written to `search_results/<dataset>/mps_search_<task>_decoded.log`.

5. **Generate the sampling config.** The decoded log is parsed to extract the top-K metapaths and merged into a `NumNeighbors` sampling config, saved as a `.py` file in `search_results/<dataset>/`.

---

## Running the full pipeline

Steps 1–3 are implemented in `run_search.py`, step 4 in `decode_search_results.py`, and step 5 in `generate_sampling_config.py`. All three are combined into a single script:

```bash
cd code/
bash run_pipeline.sh
```

### What to configure in `run_pipeline.sh`

| Variable | What it controls | Example |
|---|---|---|
| `DATASET` | Relbench dataset name | `rel-stack` |
| `TASK` | Task name within that dataset | `user-badge` |
| `TASK_KIND` | `classification` or `regression` | `classification` |
| `LAYERS` | **Hop depth** — controls both how deep the search explores and how many hops the output config covers | `3` |
| `K` | Top-K ranked metapaths to merge into the output sampling config | `20` |
| `FANOUT` | Neighbors sampled per edge type per hop in the output config | `64` |
| `CHANNELS` | Hidden embedding dimension used when converting the graph to homogeneous form | `256` |
| `NUM_NEIGHBORS` | Neighbors per hop used during the search subgraph sampling | `32` |
| `N_POS_TRAIN` / `N_NEG_TRAIN` | Positive / negative seed nodes sampled from the train table for the search | `250` / `250` |
| `N_POS_VAL` / `N_NEG_VAL` | Positive / negative seed nodes sampled from the val table for the search | `60` / `60` |
| `SEARCH_K` | Top-K edge types retained at each beam expansion step | `3` |
| `SEARCH_BEAM_WIDTH` | Maximum frontier size after beam pruning | `6` |
| `SEARCH_RANK_METRIC` | Metric used to rank and prune metapaths (`auc_roc` or `f1`) | `auc_roc` |
| `DATA_DIR` | Directory where the materialized graph cache is stored and reused | `./data` |
| `DEVICE` | GPU device | `cuda:0` |

> **Note on `LAYERS`:** setting `LAYERS=3` means the search explores up to 3-hop metapaths and the output config contains per-hop fanouts for exactly 3 hops. Setting `LAYERS=4` does the same for 4 hops. The two values are always kept in sync automatically by the pipeline script.

> **Note on `CHANNELS`:** this must match the value used when the graph cache in `DATA_DIR` was first created. If you change it, delete the cache and let it regenerate.

---

## Inputs and outputs

### Inputs (downloaded automatically on first run)
- Relbench dataset and task tables — downloaded by `get_dataset` / `get_task` into the default Relbench cache.
- Text embeddings — materialized once by `make_pkey_fkey_graph` and stored in `DATA_DIR/<dataset>_glove_materialized_cache/`. Subsequent runs (including the decode step) load from this cache.

### Outputs (all written inside `code/search_results/`)

```
code/search_results/
└── <dataset>/
    ├── mps_search_<task>.log          # full search trace (beam search details + final ranking)
    ├── mps_search_<task>.txt          # ranked metapaths only (clean input for decoding)
    ├── mps_search_<task>_decoded.log  # ranked metapaths with edge types translated to names
    └── <task_prefix>_<N>hops.py       # NumNeighbors sampling config → use with NeighborLoader
```

The `.py` config file is the final output consumed by downstream training scripts. Follow instuctions in `./MetaSieve` on how to use this MPS-GNN sampling config with models like `HeteroGraphSAGE and HGT`. Similarly, follow instuctions in `./RelGT` on how to use this MPS-GNN sampling config with the `RelGT` architecture.

### Pre-generated configs

The `search_results/` directory already contains pre-generated configs for all datasets and tasks used in the paper. These were produced with `FANOUT=64` and K value = 20. You can use them directly without re-running the search.
