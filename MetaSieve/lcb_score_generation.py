import os
import numpy as np
import pandas as pd
from scipy.stats import t


def load_mi_summary(root_dir: str, aggregate_type: str, hops_suffix: str = "3hops") -> pd.DataFrame:
    path = os.path.join(root_dir, f"mi_results_{hops_suffix}", "mi_summary_all_batches.csv")
    df = pd.read_csv(path)
    df["aggregate_type"] = aggregate_type
    df["frac_nonzero"] = df["n_nonzero"] / df["n_samples"].clip(lower=1)
    return df


def _t_critical_one_sided(delta: float, df: np.ndarray) -> np.ndarray:
    """
    Returns t_{1-delta, df} for one-sided lower bound.
    """
    df = np.asarray(df, dtype=np.float64)
    out = np.zeros_like(df, dtype=np.float64)

    mask = np.isfinite(df) & (df >= 1.0)
    if mask.any():
        out[mask] = t.ppf(1.0 - delta, df[mask])

    return out


def lower_confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    n: np.ndarray,
    *,
    delta: float,
) -> np.ndarray:
    """
    One-sided lower confidence bound:
        LCB = mu - t_{1-delta, n-1} * sigma / sqrt(n)

    Uses sample sigma (ddof=1). For n <= 1, returns mu.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    lcb = mu.copy()
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(n) & (n > 1.0)

    if mask.any():
        df = n - 1.0
        tcrit = _t_critical_one_sided(delta, df)
        se = sigma / np.sqrt(n)
        lcb[mask] = mu[mask] - tcrit[mask] * se[mask]

    return lcb


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True,
                        help="e.g. ./outputs/rel-ratebeer/user-count/")
    parser.add_argument("--delta", type=float, default=0.4)
    parser.add_argument("--hops-suffix", default="3hops")
    args = parser.parse_args()

    start_time = time.time()

    DELTA = args.delta
    OUT_DIR = args.out_dir
    HSUF = args.hops_suffix

    OUT_PATH = os.path.join(
        OUT_DIR,
        f"mi_summary_per_aggregate_with_lcb_{HSUF}_{DELTA}.csv",
    )

    METRIC_COL = "score"
    METRIC_GROUP_COLS = ["hop", "prefix", "feature", "aggregate_type"]
    COVERAGE_GROUP_COLS = ["hop", "prefix", "feature"]
    COVERAGE_SOURCE = "log_rate"

    # -----------------------------
    # Load train summaries
    # -----------------------------
    metric_dfs = [
        load_mi_summary(os.path.join(OUT_DIR, "log_rate", "train"), "log_rate", HSUF),
        load_mi_summary(os.path.join(OUT_DIR, "log_count", "train"), "log_count", HSUF),
    ]
    all_metric_df = pd.concat(metric_dfs, ignore_index=True)

    coverage_df = load_mi_summary(
        os.path.join(OUT_DIR, COVERAGE_SOURCE, "train"),
        COVERAGE_SOURCE,
        HSUF,
    )

    # -----------------------------
    # Metric stats per aggregate_type
    # -----------------------------
    metric_summary = (
        all_metric_df.groupby(METRIC_GROUP_COLS)
        .agg(
            mu_tr=(METRIC_COL, "mean"),
            sigma_tr=(METRIC_COL, "std"),
            n_train_batches=("batch", "nunique"),
        )
        .reset_index()
    )

    # -----------------------------
    # Coverage stats from one source
    # -----------------------------
    coverage_summary = (
        coverage_df.groupby(COVERAGE_GROUP_COLS)
        .agg(
            cov_mean=("frac_nonzero", "mean"),
            cov_std=("frac_nonzero", "std"),
            n_cov_batches=("batch", "nunique"),
        )
        .reset_index()
    )

    # -----------------------------
    # Merge metric + coverage stats
    # -----------------------------
    summary = pd.merge(
        metric_summary,
        coverage_summary,
        on=COVERAGE_GROUP_COLS,
        how="left",
        validate="many_to_one",
    )

    # -----------------------------
    # Fill missing values
    # -----------------------------
    summary["sigma_tr"] = summary["sigma_tr"].fillna(0.0)
    summary["cov_mean"] = summary["cov_mean"].fillna(0.0)
    summary["cov_std"] = summary["cov_std"].fillna(0.0)
    summary["n_cov_batches"] = summary["n_cov_batches"].fillna(0).astype(int)

    summary["cov_cv"] = np.where(
        summary["cov_mean"].to_numpy(dtype=np.float64) > 0.0,
        summary["cov_std"].to_numpy(dtype=np.float64)
        / summary["cov_mean"].to_numpy(dtype=np.float64),
        0.0,
    ).astype(float)

    # -----------------------------
    # Compute train LCBs
    # -----------------------------
    summary["lcb_tr"] = lower_confidence_bound(
        summary["mu_tr"].to_numpy(),
        summary["sigma_tr"].to_numpy(),
        summary["n_train_batches"].to_numpy(),
        delta=DELTA,
    )

    summary["cov_lcb_tr"] = lower_confidence_bound(
        summary["cov_mean"].to_numpy(),
        summary["cov_std"].to_numpy(),
        summary["n_cov_batches"].to_numpy(),
        delta=DELTA,
    )

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - start_time
    print(f"[write] {OUT_PATH} ({len(summary)} rows)")
    print(f"[info] DELTA={DELTA} (one-sided {(1.0 - DELTA) * 100:.1f}% LCB)")
    print(f"[info] coverage source={COVERAGE_SOURCE}")
    print(f"Done in {elapsed:.2f}s")
    
if __name__ == "__main__":
    main()