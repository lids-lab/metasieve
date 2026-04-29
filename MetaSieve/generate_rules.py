from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, Set

import pandas as pd

CNT_TAIL_RE = re.compile(r"__cnt_.*$")


def load_candidates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"hop", "prefix", "feature", "cluster_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hop"] = pd.to_numeric(df["hop"], errors="coerce").astype("Int64")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["hop", "prefix", "feature", "cluster_id"]).copy()
    df["hop"] = df["hop"].astype(int)
    df["cluster_id"] = df["cluster_id"].astype(int)

    df["prefix"] = df["prefix"].astype(str).str.strip()
    df["feature"] = df["feature"].astype(str).str.strip()
    return df


def extract_meta_path(df: pd.DataFrame) -> pd.Series:
    return df["feature"].str.replace(r"__cnt_.*$", "", regex=True)


def extract_edge_from_feature(prefix: str, feature: str) -> str | pd.NA:
    prefix = (prefix or "").strip()
    feature = (feature or "").strip()

    if not feature.startswith(prefix + "__"):
        return pd.NA

    feature_no_cnt = CNT_TAIL_RE.sub("", feature)
    suffix = feature_no_cnt[len(prefix) + 2:]  # after prefix + "__"
    return suffix if suffix else pd.NA


def build_has_good(df: pd.DataFrame, good_cluster_threshold: int) -> Dict[str, bool]:
    df = df.copy()
    df["meta_path"] = extract_meta_path(df)

    meta_depth = df.groupby("meta_path")["hop"].min() + 1
    prefix_depth = df.groupby("prefix")["hop"].min()
    path_depth = pd.concat([meta_depth, prefix_depth]).groupby(level=0).min().to_dict()

    children: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        children[row["prefix"]].add(row["meta_path"])

    direct_max_cluster = df.groupby("meta_path")["cluster_id"].max().to_dict()

    # deepest first
    paths_sorted = sorted(path_depth.items(), key=lambda x: -x[1])

    has_good: Dict[str, bool] = {}
    for path, _depth in paths_sorted:
        direct = direct_max_cluster.get(path, -1)
        direct_good = direct >= good_cluster_threshold
        child_good = any(has_good.get(ch, False) for ch in children.get(path, set()))
        has_good[path] = direct_good or child_good

    return has_good


def generate_decisions(df: pd.DataFrame, has_good: Dict[str, bool]) -> pd.DataFrame:
    df = df.copy()
    df["meta_path"] = extract_meta_path(df)
    df["edge"] = df.apply(lambda r: extract_edge_from_feature(r["prefix"], r["feature"]), axis=1)

    # Warn if edge parsing fails (helps debug missing rules)
    na_edges_by_hop = df["edge"].isna().groupby(df["hop"]).sum()
    if int(na_edges_by_hop.sum()) > 0:
        print("[WARN] Some rows have NA edge (feature didn't match prefix). NA edge counts per hop:")
        print(na_edges_by_hop.to_string())

    decisions = []
    grouped = df.groupby(["hop", "prefix", "edge"], dropna=False)

    for (hop, prefix, edge), group in grouped:
        child_paths = group["meta_path"].unique()
        good = any(has_good.get(cp, False) for cp in child_paths)

        decisions.append(
            {
                "hop": int(hop),
                "prefix": str(prefix),
                "edge": edge,
                "expand": bool(good),
                "reason": "has_good_descendant" if good else "all_descendants_cluster0",
            }
        )

    return pd.DataFrame(decisions)


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True,
                        help="e.g. ./outputs/rel-ratebeer/user-count/")
    parser.add_argument("--delta", type=float, default=0.4)
    parser.add_argument("--hops-suffix", default="3hops")
    parser.add_argument("--good-threshold", type=int, default=1)
    args = parser.parse_args()

    start_time = time.time()

    DELTA = args.delta
    OUT_DIR = args.out_dir
    HSUF = args.hops_suffix
    GOOD_CLUSTER_THRESHOLD = args.good_threshold

    cluster_files = {
        0: os.path.join(OUT_DIR, f"gmm_bad_global_lcb_{HSUF}_{DELTA}.csv"),
        1: os.path.join(OUT_DIR, f"gmm_good_global_lcb_{HSUF}_{DELTA}.csv"),
    }
    OUTPUT = os.path.join(OUT_DIR, f"rules_{HSUF}_{DELTA}.csv")

    # Load both cluster files and combine
    dfs = []
    for _cid, path in cluster_files.items():
        dfs.append(normalize_types(load_candidates(path)))
    df = pd.concat(dfs, ignore_index=True)

    # (1) build has_good + decisions
    has_good = build_has_good(df, good_cluster_threshold=GOOD_CLUSTER_THRESHOLD)
    decisions_df = generate_decisions(df, has_good=has_good)

    # (2) keep only the "bad/prune" ones
    bad_df = decisions_df[
        (decisions_df["expand"] == False) &
        (decisions_df["reason"] == "all_descendants_cluster0")
    ].copy()

    # Edge must exist to be a usable (prefix, edge) rule
    bad_df = bad_df.dropna(subset=["edge"]).copy()

    # Save directly (NO subtree pruning)
    bad_df = bad_df[["hop", "prefix", "edge", "expand", "reason"]]
    bad_df.to_csv(OUTPUT, index=False)

    print(f"[OK] wrote rules_final: {OUTPUT}")
    if not bad_df.empty:
        print("[INFO] rule counts per hop:")
        print(bad_df.groupby("hop").size().to_string())
    else:
        print("[INFO] No expand=False / all_descendants_cluster0 rules were generated.")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()