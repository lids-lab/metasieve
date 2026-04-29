import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


# -----------------------------
# Build wide extension table
# -----------------------------
def build_wide_extension_table_from_lcb(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (hop, prefix, feature), with:
      lcb_tr_log_rate, lcb_tr_log_count, cov_lcb_tr,
      plus coverage diagnostics if present.

    Expects:
      columns: hop, prefix, feature, aggregate_type, lcb_tr, cov_lcb_tr
    """
    key_cols = ["hop", "prefix", "feature"]

    needed = {"hop", "prefix", "feature", "aggregate_type", "lcb_tr", "cov_lcb_tr"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    cov_cols = [c for c in ["cov_mean", "cov_std", "cov_cv", "cov_lcb_tr"] if c in df.columns]
    cov = (
        df.groupby(key_cols)[cov_cols]
        .first()
        .reset_index()
        .copy()
    )

    # Fill any missing coverage fields
    for c in ["cov_mean", "cov_std", "cov_cv", "cov_lcb_tr"]:
        if c not in cov.columns:
            cov[c] = 0.0
        cov[c] = pd.to_numeric(cov[c], errors="coerce").fillna(0.0).astype(float)

    # Pivot metric LCBs
    wide = df.pivot_table(
        index=key_cols,
        columns="aggregate_type",
        values="lcb_tr",
        aggfunc="first",
    ).reset_index()

    if "log_rate" in wide.columns:
        wide = wide.rename(columns={"log_rate": "lcb_tr_log_rate"})
    else:
        wide["lcb_tr_log_rate"] = 0.0

    if "log_count" in wide.columns:
        wide = wide.rename(columns={"log_count": "lcb_tr_log_count"})
    else:
        wide["lcb_tr_log_count"] = 0.0

    out = wide.merge(
        cov[["hop", "prefix", "feature", "cov_mean", "cov_std", "cov_cv", "cov_lcb_tr"]],
        on=key_cols,
        how="left",
    )

    for c in ["lcb_tr_log_rate", "lcb_tr_log_count", "cov_mean", "cov_std", "cov_cv", "cov_lcb_tr"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    keep_cols = [
        "hop", "prefix", "feature",
        "lcb_tr_log_rate", "lcb_tr_log_count",
        "cov_mean", "cov_std", "cov_cv", "cov_lcb_tr"
    ]
    return out[keep_cols]


def save_dead_paths(ext_df: pd.DataFrame, out_path: str):
    """
    Dead if all modeled signals are zero.
    """
    dead_cols = [
        "lcb_tr_log_rate",
        "lcb_tr_log_count",
        "cov_lcb_tr",
    ]

    X = ext_df[dead_cols].copy().fillna(0.0)
    dead_mask = (X == 0.0).all(axis=1)

    dead_df = ext_df.loc[dead_mask].copy()
    live_df = ext_df.loc[~dead_mask].copy()

    dead_df.to_csv(out_path, index=False)
    print(f"[write] {out_path} ({len(dead_df)} rows)")
    print(f"[info] dead={int(dead_mask.sum())} live={int((~dead_mask).sum())}")
    return dead_df, live_df


# -----------------------------
# Q score
# -----------------------------
def percentile_rank_within_hop(values: pd.Series, hop: pd.Series) -> pd.Series:
    v = values.fillna(0.0).astype(float)
    return v.groupby(hop).rank(pct=True, method="average")


def compute_q_scores_from_lcb(live_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses LCBs:
      lcb_tr_log_rate, lcb_tr_log_count, cov_lcb_tr

    Scale-free within hop:
      p_rate  = percentile rank within hop of lcb_tr_log_rate
      p_count = percentile rank within hop of lcb_tr_log_count
      p_cov   = percentile rank within hop of cov_lcb_tr

    Q:
      Q = max(p_rate, p_count) + p_cov
    """
    out = live_df.copy()

    p_rate = percentile_rank_within_hop(out["lcb_tr_log_rate"], out["hop"]).to_numpy()
    p_count = percentile_rank_within_hop(out["lcb_tr_log_count"], out["hop"]).to_numpy()
    p_cov = percentile_rank_within_hop(out["cov_lcb_tr"], out["hop"]).to_numpy()

    out["p_rate"] = p_rate
    out["p_count"] = p_count
    out["p_cov"] = p_cov

    winner = np.where(
        p_rate > p_count, "log_rate",
        np.where(p_count > p_rate, "log_count", "tie")
    )

    strength = np.maximum(p_rate, p_count)  ## Baseline
    out["Q_score"] = strength + p_cov 
    out["winner_agg"] = winner

    scored = out[
        ["hop", "prefix", "feature", "Q_score", "winner_agg", "p_rate", "p_count", "p_cov"]
    ].copy()
    scored = scored.sort_values("Q_score", ascending=False).reset_index(drop=True)
    return scored


# -----------------------------
# GMM split (global)
# -----------------------------
def split_gmm_2component_global(
    df: pd.DataFrame,
    out_good: str,
    out_bad: str,
    *,
    q_col: str = "Q_score",
    random_state: int = 0,
):
    needed = {"hop", "prefix", "feature", q_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[["hop", "prefix", "feature", q_col]].copy()
    work[q_col] = pd.to_numeric(work[q_col], errors="coerce")
    work = work.dropna(subset=[q_col]).reset_index(drop=True)

    if len(work) < 2 or work[q_col].nunique() < 2:
        good_df = work.sort_values(q_col, ascending=False).copy()
        bad_df = work.iloc[0:0].copy()

        good_df["cluster_id"] = 1
        bad_df["cluster_id"] = 0

        good_df = good_df[["hop", "prefix", "feature", q_col, "cluster_id"]]
        bad_df = bad_df[["hop", "prefix", "feature", q_col, "cluster_id"]]

        os.makedirs(os.path.dirname(out_good), exist_ok=True)
        good_df.to_csv(out_good, index=False)
        bad_df.to_csv(out_bad, index=False)
        return good_df, bad_df

    X = work[[q_col]].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=20,
        reg_covar=1e-6,
        random_state=random_state,
    )
    gmm.fit(Xz)

    labels = gmm.predict(Xz)
    means = gmm.means_.reshape(-1)

    bad_comp = int(np.argmin(means))
    good_comp = int(np.argmax(means))

    bad_df = work.loc[labels == bad_comp].copy()
    good_df = work.loc[labels == good_comp].copy()

    good_df["cluster_id"] = 1
    bad_df["cluster_id"] = 0

    good_df = good_df.sort_values(q_col, ascending=False, kind="mergesort").reset_index(drop=True)
    bad_df = bad_df.sort_values(q_col, ascending=True, kind="mergesort").reset_index(drop=True)

    good_df = good_df[["hop", "prefix", "feature", q_col, "cluster_id"]]
    bad_df = bad_df[["hop", "prefix", "feature", q_col, "cluster_id"]]

    os.makedirs(os.path.dirname(out_good), exist_ok=True)
    good_df.to_csv(out_good, index=False)
    bad_df.to_csv(out_bad, index=False)

    return good_df, bad_df


# -----------------------------
# Main
# -----------------------------
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

    INPUT_PATH = os.path.join(
        OUT_DIR,
        f"mi_summary_per_aggregate_with_lcb_{HSUF}_{DELTA}.csv",
    )
    OUTPUT_DEAD = os.path.join(
        OUT_DIR,
        f"extension_dead_paths_lcb_{HSUF}_{DELTA}.csv",
    )
    OUTPUT_Q = os.path.join(
        OUT_DIR,
        f"extension_q_scores_lcb_{HSUF}_{DELTA}.csv",
    )
    OUT_GOOD = os.path.join(
        OUT_DIR,
        f"gmm_good_global_lcb_{HSUF}_{DELTA}.csv",
    )
    OUT_BAD = os.path.join(
        OUT_DIR,
        f"gmm_bad_global_lcb_{HSUF}_{DELTA}.csv",
    )

    for p in [OUTPUT_DEAD, OUTPUT_Q, OUT_GOOD, OUT_BAD]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    ext_df = build_wide_extension_table_from_lcb(df)
    _, live_df = save_dead_paths(ext_df, OUTPUT_DEAD)

    if len(live_df) == 0:
        empty = live_df[["hop", "prefix", "feature"]].copy()
        empty["Q_score"] = pd.Series(dtype=float)
        empty.to_csv(OUTPUT_Q, index=False)
        print(f"[write] {OUTPUT_Q} (0 rows)")
        return

    scored = compute_q_scores_from_lcb(live_df)
    scored.to_csv(OUTPUT_Q, index=False)
    print(f"[write] {OUTPUT_Q} ({len(scored)} rows)")

    good_df, bad_df = split_gmm_2component_global(scored, OUT_GOOD, OUT_BAD)
    print(f"[info] gmm good={len(good_df)} bad={len(bad_df)}")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()