import json
import os
from typing import Dict, List, Tuple

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def parse_prefix(prefix: str) -> List[str]:
    """
    Split the prefix like:
        'users__rev_f2p_UserId__votes'
    into ['users', 'rev_f2p_UserId', 'votes'].
    """
    return prefix.split("__")


def parse_edge_column(edge: str) -> Tuple[str, str]:
    """
    Split the edge column like:
        'f2p_PostId__posts'
    into (relation_name, dst_table):

        ('f2p_PostId', 'posts')
    """
    rel, dst = edge.split("__", maxsplit=1)
    return rel, dst


def build_edge_type(prefix: str, edge: str) -> Tuple[str, str, str]:
    """
    Given (prefix, edge) from the CSV, build the torch-geometric edge_type triple:

      (src_type, relation_name, dst_type)

    Logic:
      - prefix: 'users__rev_f2p_UserId__votes'
        => parts = ['users', 'rev_f2p_UserId', 'votes']
        => src_type = last table in prefix => 'votes'

      - edge: 'f2p_PostId__posts'
        => relation_name = 'f2p_PostId'
        => dst_type      = 'posts'

      - if 'relation_name' has NO 'rev_' prefix in CSV, we ADD 'rev_'.
      - if 'relation_name' ALREADY starts with 'rev_', we STRIP it.
    """
    prefix_parts = parse_prefix(prefix)
    src_type = prefix_parts[-1]

    rel_name, dst_type = parse_edge_column(edge)

    if rel_name.startswith("rev_"):
        relation = rel_name[len("rev_"):]
    else:
        relation = "rev_" + rel_name

    return (dst_type, relation, src_type)


def build_pruned_neighbors_from_csv(
    csv_path: str,
    num_layers: int,
    base_neighbors: int,
) -> Dict[str, List[int]]:
    """
    Read the pruning CSV and produce a dict:

        { "(src, rel, dst)": [128, 0, 128, 0], ... }

    Only rows with expand == False are used.

    IMPORTANT: we map CSV 'hop' -> neighbor index **directly**:

        neighbor_index = hop

    So with num_layers=4 and hops in {1,2,3} you get:

        hop=1 -> index 1
        hop=2 -> index 2
        hop=3 -> index 3

    If the same edge-type triple appears in multiple rows
    at different hops, we combine them via OR on the prune
    indicator (any False at a hop -> that hop is pruned).
    """
    df = pd.read_csv(csv_path)

    required_cols = {"hop", "prefix", "edge", "expand"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Only rows where expand == False => we want to prune
    pruned_rows = df[df["expand"].astype(str).str.lower() == "false"]

    # key = edge_type triple, value = list[bool] of length num_layers
    # prune_mask[etype][i] = True  => hop i is pruned (0 neighbors)
    prune_mask: Dict[Tuple[str, str, str], List[bool]] = {}

    for _, row in pruned_rows.iterrows():
        hop = int(row["hop"])

        idx = hop
        if not (0 <= idx < num_layers):
            continue

        prefix = row["prefix"]
        edge = row["edge"]
        edge_type = build_edge_type(prefix, edge)
        if edge_type not in prune_mask:
            prune_mask[edge_type] = [False] * num_layers

        prune_mask[edge_type][idx] = True

    # Convert prune_mask -> actual neighbor arrays
    neighbors_config: Dict[str, List[int]] = {}
    for etype, mask in prune_mask.items():
        neighbors = [
            0 if mask[h] else base_neighbors
            for h in range(num_layers)
        ]
        neighbors_config[str(etype)] = neighbors

    return neighbors_config


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True,
                        help="e.g. ./outputs/rel-ratebeer/user-count/")
    parser.add_argument("--delta", type=float, default=0.4)
    parser.add_argument("--hops-suffix", default="3hops")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--base-neighbors", type=int, default=64)
    parser.add_argument("--task-name", required=True,
                        help="e.g. user-count (used in output filename)")
    args = parser.parse_args()

    start_time = time.time()

    DELTA = args.delta
    OUT_DIR = args.out_dir
    HSUF = args.hops_suffix
    NUM_LAYERS = args.num_layers
    BASE_NEIGHBORS = args.base_neighbors

    CSV_PATH = os.path.join(OUT_DIR, f"rules_{HSUF}_{DELTA}.csv")
    JSON_OUT_PATH = os.path.join(OUT_DIR, f"{args.task_name}_pruned_{HSUF}_{DELTA}.json")

    neighbors_dict = build_pruned_neighbors_from_csv(
        csv_path=CSV_PATH,
        num_layers=NUM_LAYERS,
        base_neighbors=BASE_NEIGHBORS,
    )

    config = {
        "num_layers": NUM_LAYERS,
        "base_neighbors": BASE_NEIGHBORS,
        "neighbors": neighbors_dict,
    }

    out_dir = os.path.dirname(JSON_OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(JSON_OUT_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote NumNeighbors config to: {JSON_OUT_PATH}")
    print(f"Number of edge types with custom pruning: {len(neighbors_dict)}")
    # show a few entries
    for i, (k, v) in enumerate(neighbors_dict.items()):
        print(f"{k}: {v}")
        if i >= 5:
            break

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()