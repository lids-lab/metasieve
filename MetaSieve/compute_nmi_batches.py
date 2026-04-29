import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import time
from dataclasses import dataclass
from glob import glob
from typing import List, Tuple, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# -------------------------------
# Config
# -------------------------------
@dataclass(frozen=True)
class MIConfig:
    # "auto", "classification", "regression"
    label_mode: str = "auto"

    # Used only when label_mode="auto"
    # If number of unique label values <= this => classification else regression
    classification_max_unique: int = 20

    random_state: int = 42

    # Regression normalization:
    # - False => npmi := MI (raw nats)
    # - True  => npmi := MI / H(bin(Y)) where bin(Y) is quantile-binned
    regression_normalize: bool = True

    # FIXED bin count (used when regression_norm_bins_auto=False)
    regression_norm_bins: int = 20

    # AUTO bin count selection (new)
    # Uses np.histogram_bin_edges(..., bins=regression_norm_bins_auto_method) to propose a bin count,
    # then clamps it to [min,max] and ensures at least regression_min_samples_per_bin per bin.
    regression_norm_bins_auto: bool = True
    regression_norm_bins_auto_method: str = "auto"  # "auto", "fd", "sturges", "scott", ...
    regression_norm_bins_min: int = 5
    regression_norm_bins_max: int = 64
    regression_min_samples_per_bin: int = 200  # clamp bins so each bin has >= this many samples

    # "global" => compute bin edges once for ROOT and reuse (recommended)
    # "per_file" => compute edges per file
    regression_bin_edges_mode: str = "global"

    # Global edges sampling
    regression_edges_max_files: int = 200
    regression_edges_max_rows_per_file: int = 20000

    # Logging
    log_every: int = 50


# -------------------------------
# Helper functions
# -------------------------------
def find_count_columns(columns: List[str]) -> List[str]:
    """All feature columns contain '__cnt_'."""
    return [c for c in columns if "__cnt_" in c]


def infer_hop_from_path(path: str) -> int:
    m = re.search(r"[\\/]+hop_(\d+)[\\/]+", path)
    return int(m.group(1)) if m else -1


def infer_batch_from_path(path: str) -> str:
    m = re.search(r"[\\/]+(batch_\d+)[\\/]+", path)
    return m.group(1) if m else "batch_unknown"


def _natural_batch_key(path: str) -> int:
    m = re.search(r"batch_(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def infer_label_column(df: pd.DataFrame, csv_path: str) -> str:
    if "label" in df.columns:
        return "label"
    if "WillGetBadge" in df.columns:
        return "WillGetBadge"
    raise ValueError(f"Could not find label column ('label' or 'WillGetBadge') in {csv_path}")


def discrete_entropy(y: np.ndarray, eps: float = 1e-12) -> float:
    """Discrete entropy H(Y) in nats."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def quantile_edges(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based bin edges for y (unique + monotonic)."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if n_bins <= 1:
        lo, hi = float(np.min(y)), float(np.max(y))
        return np.array([lo, hi], dtype=np.float64)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y, qs))
    if edges.size < 2:
        lo, hi = float(np.min(y)), float(np.max(y))
        edges = np.array([lo, hi], dtype=np.float64)
    return edges


def bin_with_edges(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Bin y using edges. Returns bins 0..B-1.
    Uses np.digitize on internal cutpoints.
    """
    y = np.asarray(y, dtype=np.float64)
    if edges is None or len(edges) < 2:
        return np.zeros_like(y, dtype=np.int64)

    cut = np.asarray(edges[1:-1], dtype=np.float64)
    if cut.size == 0:
        return np.zeros_like(y, dtype=np.int64)

    # right=True => bins[i-1] < x <= bins[i]
    return np.digitize(y, cut, right=True).astype(np.int64)


def decide_label_mode(y_numeric: np.ndarray, cfg: MIConfig) -> str:
    """Auto-detect: classification if nunique <= threshold else regression."""
    if cfg.label_mode != "auto":
        if cfg.label_mode not in ("classification", "regression"):
            raise ValueError(f"label_mode must be auto/classification/regression, got {cfg.label_mode}")
        return cfg.label_mode

    nunique = int(pd.Series(y_numeric).nunique(dropna=True))
    return "classification" if nunique <= cfg.classification_max_unique else "regression"


def read_csv_minimal_npmi(csv_path: str) -> pd.DataFrame:
    """Read label + feature columns (needed for MI/NPMI)."""
    def _usecol(c: str) -> bool:
        return (c == "label") or (c == "WillGetBadge") or ("__cnt_" in c)
    return pd.read_csv(csv_path, usecols=_usecol)


def read_csv_minimal_cost(csv_path: str) -> pd.DataFrame:
    """Read only feature columns (cost comes from paired log_count)."""
    def _usecol(c: str) -> bool:
        return "__cnt_" in c
    return pd.read_csv(csv_path, usecols=_usecol)


def paired_log_count_path(any_csv_path: str) -> str:
    """Map .../log_rate/... -> .../log_count/... ; log_count stays unchanged."""
    needle = os.sep + "log_rate" + os.sep
    repl = os.sep + "log_count" + os.sep
    return any_csv_path.replace(needle, repl) if needle in any_csv_path else any_csv_path


# -------------------------------
# Auto-select number of bins for regression normalization
# -------------------------------
def choose_regression_n_bins(y: np.ndarray, cfg: MIConfig) -> int:
    """
    Returns an integer bin count for regression label normalization.

    If cfg.regression_norm_bins_auto is False: return cfg.regression_norm_bins.

    If True:
      1) propose k from np.histogram_bin_edges(y, bins=cfg.regression_norm_bins_auto_method)
      2) clamp to [min,max]
      3) enforce at least cfg.regression_min_samples_per_bin per bin
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)

    if not cfg.regression_norm_bins_auto:
        return max(1, int(cfg.regression_norm_bins))

    if n <= 1:
        return 1

    # Proposed bins from NumPy estimator
    try:
        edges = np.histogram_bin_edges(y, bins=cfg.regression_norm_bins_auto_method)
        k = int(max(1, len(edges) - 1))
    except Exception:
        # Safe fallback
        k = int(max(1, cfg.regression_norm_bins))

    # Clamp to user bounds
    k = max(int(cfg.regression_norm_bins_min), min(k, int(cfg.regression_norm_bins_max)))

    # Enforce minimum samples/bin (prevents overly fine binning)
    if cfg.regression_min_samples_per_bin and cfg.regression_min_samples_per_bin > 0:
        max_k_from_samples = max(1, n // int(cfg.regression_min_samples_per_bin))
        k = min(k, max_k_from_samples)

    return max(1, k)


# -------------------------------
# Cost from log_count (optionally only non-zeros)
# -------------------------------
def cost_from_logcount_matrix(X_logcnt: np.ndarray, nonzero_only: bool = True) -> np.ndarray:
    """
    cost_j = mean(expm1(log_count_j)).

    If nonzero_only=True: average over entries where expm1(log_count) != 0.
    Else: average over all entries (including zeros).
    """
    X = np.asarray(X_logcnt, dtype=np.float64, order="C")
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, 0.0, None)
    values = np.expm1(X)

    if not nonzero_only:
        return values.mean(axis=0)

    mask = values != 0.0
    denom = mask.sum(axis=0).astype(np.float64)
    numer = (values * mask).sum(axis=0)

    out = np.zeros_like(numer, dtype=np.float64)
    np.divide(numer, denom, out=out, where=(denom > 0))
    return out


# -------------------------------
# Global regression bin edges (optional)
# -------------------------------
def compute_global_regression_edges(root: str, hops: List[int], cfg: MIConfig) -> np.ndarray:
    """
    Compute global QUANTILE bin edges for regression label normalization.
    We only read label column(s) across a sample of files.
    """
    csvs: List[str] = []
    batch_dirs = sorted(
        [d for d in glob(os.path.join(root, "batch_*")) if os.path.isdir(d)],
        key=_natural_batch_key,
    )
    for bdir in batch_dirs:
        for h in hops:
            csvs.extend(sorted(glob(os.path.join(bdir, f"hop_{h}", "*.csv"))))

    if not csvs:
        return np.array([0.0, 1.0], dtype=np.float64)

    max_files = max(1, int(cfg.regression_edges_max_files))
    if len(csvs) > max_files:
        idx = np.linspace(0, len(csvs) - 1, num=max_files, dtype=int)
        csvs = [csvs[i] for i in idx]

    ys: List[np.ndarray] = []
    for fp in csvs:
        try:
            df = pd.read_csv(fp, usecols=lambda c: c in ("label", "WillGetBadge"))
            if df.shape[1] == 0:
                continue

            label_col = "label" if "label" in df.columns else ("WillGetBadge" if "WillGetBadge" in df.columns else None)
            if label_col is None:
                continue

            y = pd.to_numeric(df[label_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if y.size == 0:
                continue

            if y.size > cfg.regression_edges_max_rows_per_file:
                y = y[: cfg.regression_edges_max_rows_per_file]

            ys.append(y)
        except Exception:
            continue

    if not ys:
        return np.array([0.0, 1.0], dtype=np.float64)

    y_all = np.concatenate(ys)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)

    n_bins = choose_regression_n_bins(y_all, cfg)
    edges = quantile_edges(y_all, n_bins)

    if edges.size < 2:
        edges = np.array([float(np.nanmin(y_all)), float(np.nanmax(y_all))], dtype=np.float64)

    print(
        f"[info] Global regression bins: requested_mode="
        f"{'auto:'+cfg.regression_norm_bins_auto_method if cfg.regression_norm_bins_auto else 'fixed'} "
        f"-> n_bins={n_bins} -> effective_bins={(len(edges) - 1)}",
        flush=True,
    )

    return edges


# -------------------------------
# Core: MI/NPMI (classification OR regression) + cost + score
# -------------------------------
def compute_npmi_cost_score_for_file(
    csv_path_npmi: str,
    cfg: MIConfig,
    regression_edges: Optional[np.ndarray],
    cost_nonzero_only: bool = True,
    eps: float = 1e-12,
) -> pd.DataFrame:
    hop = infer_hop_from_path(csv_path_npmi)
    batch = infer_batch_from_path(csv_path_npmi)
    prefix = os.path.splitext(os.path.basename(csv_path_npmi))[0]

    out_cols = [
        "batch", "hop", "prefix", "feature",
        "npmi", "cost", "score",
        "n_samples", "n_nonzero",
    ]

    # 1) Load NPMI-side data (label + features) from ROOT (log_count OR log_rate)
    df_npmi = read_csv_minimal_npmi(csv_path_npmi)
    label_col = infer_label_column(df_npmi, csv_path_npmi)
    feature_cols_npmi = find_count_columns(df_npmi.columns)

    if not feature_cols_npmi:
        return pd.DataFrame(columns=out_cols)

    # Drop NaN labels (keep X aligned)
    y_series = pd.to_numeric(df_npmi[label_col], errors="coerce")
    valid = y_series.notna()
    if not valid.all():
        df_npmi = df_npmi.loc[valid].reset_index(drop=True)
        y_series = y_series.loc[valid].reset_index(drop=True)

    if len(df_npmi) == 0:
        return pd.DataFrame(columns=out_cols)

    y_numeric = y_series.to_numpy(dtype=np.float64, copy=False)
    mode = decide_label_mode(y_numeric, cfg)

    # X for MI/NPMI (from npmi-side file)
    Xn = df_npmi[feature_cols_npmi].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xn_np = Xn.to_numpy(dtype=np.float64, copy=False)
    np.nan_to_num(Xn_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Load COST-side data from paired log_count file
    csv_path_cost = paired_log_count_path(csv_path_npmi)
    if not os.path.exists(csv_path_cost):
        raise FileNotFoundError(f"Missing paired log_count file for cost: {csv_path_cost}")

    df_cost = read_csv_minimal_cost(csv_path_cost)

    # Keep row alignment with any dropped labels
    if not valid.all():
        df_cost = df_cost.loc[valid.to_numpy()].reset_index(drop=True)

    feature_cols_cost = list(df_cost.columns)

    # 3) Align columns
    common_cols = [c for c in feature_cols_npmi if c in feature_cols_cost]
    if not common_cols:
        return pd.DataFrame(columns=out_cols)

    # Restrict in NPMI order
    Xn = Xn[common_cols]
    Xn_np = Xn.to_numpy(dtype=np.float64, copy=False)
    np.nan_to_num(Xn_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    Xc = df_cost[common_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xc_np = Xc.to_numpy(dtype=np.float64, copy=False)
    np.nan_to_num(Xc_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # n_samples + n_nonzero come from log_count side
    n_samples = int(Xc_np.shape[0])
    n_nonzero = (Xc_np != 0.0).sum(axis=0).astype(np.int32)

    # 4) Compute MI (skip all-zero or constant features)
    n_features = Xn_np.shape[1]
    mi_full = np.zeros(n_features, dtype=np.float64)

    keep_mask = (n_nonzero > 0) & (Xn_np.min(axis=0) != Xn_np.max(axis=0))

    if mode == "classification":
        y_cls = y_numeric.astype(np.int64, copy=False)

        if np.unique(y_cls).size < 2:
            npmi = np.zeros_like(mi_full)
        else:
            if keep_mask.any():
                mi_keep = mutual_info_classif(
                    Xn_np[:, keep_mask],
                    y_cls,
                    discrete_features=False,
                    random_state=cfg.random_state,
                )
                mi_keep = np.nan_to_num(mi_keep, nan=0.0)
                mi_keep = np.where(mi_keep < 0, 0.0, mi_keep)
                mi_full[keep_mask] = mi_keep

            H = discrete_entropy(y_cls)  # >= 0
            if H <= 0.0:
                npmi = np.zeros_like(mi_full)
            else:
                npmi = np.clip(mi_full / H, 0.0, 1.0)

    else:
        y_reg = y_numeric.astype(np.float64, copy=False)

        if float(np.nanmin(y_reg)) == float(np.nanmax(y_reg)):
            npmi = np.zeros_like(mi_full)
        else:
            if keep_mask.any():
                mi_keep = mutual_info_regression(
                    Xn_np[:, keep_mask],
                    y_reg,
                    discrete_features=False,
                    random_state=cfg.random_state,
                )
                mi_keep = np.nan_to_num(mi_keep, nan=0.0)
                mi_keep = np.where(mi_keep < 0, 0.0, mi_keep)
                mi_full[keep_mask] = mi_keep

            if not cfg.regression_normalize:
                npmi = mi_full.copy()
            else:
                if cfg.regression_bin_edges_mode == "global":
                    edges = regression_edges
                    if edges is None:
                        n_bins = choose_regression_n_bins(y_reg, cfg)
                        edges = quantile_edges(y_reg, n_bins)
                elif cfg.regression_bin_edges_mode == "per_file":
                    n_bins = choose_regression_n_bins(y_reg, cfg)
                    edges = quantile_edges(y_reg, n_bins)
                else:
                    raise ValueError("regression_bin_edges_mode must be 'global' or 'per_file'")

                yb = bin_with_edges(
                    y_reg,
                    edges if edges is not None else np.array([0.0, 1.0], dtype=np.float64)
                )
                H = discrete_entropy(yb)  # >= 0
                if H <= 0.0:
                    npmi = np.zeros_like(mi_full)
                else:
                    npmi = np.clip(mi_full / H, 0.0, 1.0)

    # 5) COST from log_count
    cost = cost_from_logcount_matrix(Xc_np, nonzero_only=cost_nonzero_only)

    # 6) SCORE (no eps): score=0 if cost<=0 else npmi/cost
    score = np.zeros_like(npmi, dtype=np.float64)
    np.divide(npmi, cost, out=score, where=(cost > 0.0))

    out = pd.DataFrame({
        "batch": batch,
        "hop": hop,
        "prefix": prefix,
        "feature": common_cols,
        "npmi": npmi,
        "cost": cost,
        "score": score,
        "n_samples": n_samples,
        "n_nonzero": n_nonzero,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    return out


# -------------------------------
# Per-batch worker (runs in a process)
# -------------------------------
def process_one_batch(
    bdir: str,
    outroot: str,
    hops: List[int],
    cfg: MIConfig,
    regression_edges: Optional[np.ndarray],
    cost_nonzero_only: bool,
) -> Tuple[str, str, int]:
    batch_name = os.path.basename(bdir)
    outdir = os.path.join(outroot, batch_name)
    os.makedirs(outdir, exist_ok=True)

    print(f"[info] START {batch_name}", flush=True)

    batch_rows: List[pd.DataFrame] = []

    for h in hops:
        pattern = os.path.join(bdir, f"hop_{h}", "*.csv")
        files = sorted(glob(pattern))
        print(f"[{batch_name}] Hop {h}: {len(files)} files", flush=True)

        for i, fp in enumerate(files, 1):
            try:
                rows = compute_npmi_cost_score_for_file(
                    fp,
                    cfg=cfg,
                    regression_edges=regression_edges,
                    cost_nonzero_only=cost_nonzero_only,
                )
                if not rows.empty:
                    batch_rows.append(rows)
            except Exception as e:
                print(f"[error] {batch_name} hop_{h} file={fp}: {e}", flush=True)

            if cfg.log_every and (i % cfg.log_every == 0 or i == len(files)):
                print(f"[{batch_name}] Hop {h}: processed {i}/{len(files)}", flush=True)

    if batch_rows:
        batch_df = pd.concat(batch_rows, axis=0, ignore_index=True)
    else:
        batch_df = pd.DataFrame(columns=[
            "batch", "hop", "prefix", "feature",
            "npmi", "cost", "score",
            "n_samples", "n_nonzero",
        ])

    batch_summary_path = os.path.join(outdir, "mi_summary.csv")
    batch_df.to_csv(batch_summary_path, index=False)

    for h in sorted(batch_df["hop"].dropna().unique()):
        hop_df = batch_df[batch_df["hop"] == h].copy()
        hop_path = os.path.join(outdir, f"mi_hop_{int(h)}.csv")
        hop_df.to_csv(hop_path, index=False)

    print(f"[info] DONE {batch_name} ({len(batch_df)} rows)", flush=True)
    return batch_name, batch_summary_path, int(len(batch_df))


# -------------------------------
# Main (parallelize per batch)
# -------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True,
                        help="e.g. ./outputs/rel-ratebeer/user-count")
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--label-mode", default="auto",
                        choices=["auto", "classification", "regression"])
    parser.add_argument("--cost-nonzero-only", action="store_true", default=True)
    parser.add_argument("--hops-suffix", default="3hops")
    args = parser.parse_args()

    start_time = time.time()

    roots = [
        os.path.join(args.out_dir, "log_rate", "train"),
        os.path.join(args.out_dir, "log_count", "train"),
    ]

    cfg = MIConfig(
        label_mode=args.label_mode,
        regression_normalize=True,
        regression_norm_bins_auto=True,
        regression_norm_bins_auto_method="auto",
        regression_norm_bins_min=5,
        regression_norm_bins_max=64,
        regression_min_samples_per_bin=200,
        regression_bin_edges_mode="global",
        regression_edges_max_files=200,
        regression_edges_max_rows_per_file=20000,
        random_state=42,
        log_every=10,
    )

    HOPS = args.hops
    NUM_WORKERS = args.num_workers
    COST_NONZERO_ONLY = args.cost_nonzero_only

    print(f"[info] cfg={cfg}", flush=True)
    print(f"[info] COST_NONZERO_ONLY={COST_NONZERO_ONLY}", flush=True)

    for ROOT in roots:
        root_start = time.time()
        OUTROOT = os.path.join(ROOT, f"mi_results_{args.hops_suffix}")

        print(f"\n{'='*60}", flush=True)
        print(f"[info] ROOT={ROOT}", flush=True)
        print(f"[info] OUTROOT={OUTROOT}", flush=True)
        print(f"{'='*60}", flush=True)

        os.makedirs(OUTROOT, exist_ok=True)

        batch_dirs = [d for d in glob(os.path.join(ROOT, "batch_*")) if os.path.isdir(d)]
        batch_dirs = sorted(batch_dirs, key=_natural_batch_key)

        if not batch_dirs:
            print("[warn] No batch directories found.", flush=True)
            print(f"[warn] Checked pattern: {os.path.join(ROOT, 'batch_*')}", flush=True)
            continue

        regression_edges: Optional[np.ndarray] = None
        needs_global_edges = (
            cfg.regression_normalize
            and cfg.regression_bin_edges_mode == "global"
            and cfg.label_mode in ("auto", "regression")
        )
        if needs_global_edges:
            print("[info] Computing global regression bin edges...", flush=True)
            regression_edges = compute_global_regression_edges(ROOT, HOPS, cfg)
            print(f"[info] Global edges computed: {len(regression_edges)} (=> {len(regression_edges)-1} bins)", flush=True)

        print(f"[info] Found {len(batch_dirs)} batch dirs. Using {NUM_WORKERS} processes.", flush=True)

        summary_paths: List[str] = []

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_one_batch, bdir, OUTROOT, HOPS, cfg, regression_edges, COST_NONZERO_ONLY
                ): bdir
                for bdir in batch_dirs
            }
            print(f"[info] Submitted {len(futures)} batch jobs.", flush=True)

            for fut in as_completed(futures):
                bdir = futures[fut]
                try:
                    batch_name, summary_path, nrows = fut.result()
                    summary_paths.append(summary_path)
                    print(f"[info] Collected {batch_name}: {nrows} rows", flush=True)
                except Exception as e:
                    print(f"[error] batch failed ({bdir}): {e}", flush=True)

        if summary_paths:
            overall = pd.concat((pd.read_csv(p) for p in summary_paths), ignore_index=True)
            overall_path = os.path.join(OUTROOT, "mi_summary_all_batches.csv")
            overall.to_csv(overall_path, index=False)
            print(f"[write] {overall_path} ({len(overall)} rows)", flush=True)

            for h in sorted(overall["hop"].dropna().unique()):
                hop_df = overall[overall["hop"] == h].copy()
                hop_path = os.path.join(OUTROOT, f"mi_hop_{int(h)}_all_batches.csv")
                hop_df.to_csv(hop_path, index=False)
                print(f"[write] {hop_path} ({len(hop_df)} rows)", flush=True)

            print("\nTop features across ALL batches (preview):", flush=True)
            for h in sorted(overall["hop"].dropna().unique()):
                sub = overall[overall["hop"] == h].nlargest(5, "score")
                print(f"\nHop {int(h)}:", flush=True)
                for _, r in sub.iterrows():
                    print(
                        f"  score={r['score']:.6f} npmi={r['npmi']:.6f} cost={r['cost']:.6f} "
                        f"[{r['batch']}] {r['prefix']} :: {r['feature']}",
                        flush=True,
                    )

        root_elapsed = time.time() - root_start
        print(f"\n[info] {ROOT} done in {root_elapsed:.2f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"\nAll roots done in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()