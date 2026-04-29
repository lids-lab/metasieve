import os
import sys
import time
import math
import glob
import torch
import shutil
import duckdb
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from relbench.tasks import get_task
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

from torch_geometric.seed import seed_everything
from torch_frame.config.text_embedder import TextEmbedderConfig

from candidate_generation_utils import *
from embeddings import GloveTextEmbedding

from feature_components import (
    CountDistinctStrategy,
    LogCountStrategy,
    RateStrategy,
    LogRateStrategy,
    FeatureStrategy,
    _prefix_key,
    frontier_name,
    MetaPath,
    EdgeType,
)

USER = os.environ.get("USER") or str(os.getuid())
SPILL_ROOT = f"/dev/shm/{USER}/duckdb_spill"   # <- big mount
os.makedirs(SPILL_ROOT, exist_ok=True)

def cleanup_duckdb_spill_dirs(root_dir: str, prefix: str = "duckdb_spill_b"):
    """
    Deletes per-batch spill dirs like:
      {root_dir}/duckdb_spill_b1_xxx/
      {root_dir}/duckdb_spill_b2_yyy/
    """
    pattern = os.path.join(root_dir, f"{prefix}*")
    for p in glob.glob(pattern):
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

cleanup_duckdb_spill_dirs(SPILL_ROOT)
cleanup_duckdb_spill_dirs("/tmp")       

# -----------------------
# Global config
# -----------------------

seed_everything(42)

def _parse_early_args():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset", default="rel-ratebeer")
    p.add_argument("--task", default="user-count")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-hop", type=int, default=3)
    p.add_argument("--hops-to-export", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n-batches", type=int, default=32)
    args, _ = p.parse_known_args()
    return args

_CLI = _parse_early_args()

DATASET_NAME = _CLI.dataset
TASK_NAME = _CLI.task
TARGET_N_BATCHES = _CLI.n_batches

# START_NODE_TYPE = "posts"   # can be overridden by inference below
SEED_ID_COL = "SeedId"        # canonical seed id col in temp/views/tables
PKEY_COL = "Id"               # legacy fallback; true PKs inferred from db.table_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

dataset = get_dataset(DATASET_NAME, download=True)
task = get_task(DATASET_NAME, TASK_NAME, download=True)

train_table = task.get_table("train")
val_table   = task.get_table("val")
test_table  = task.get_table("test")

# -----------------------
# Task schema & sampling
# -----------------------

def _infer_entity_task_schema_from_split_table(tbl) -> Tuple[str, str, str, str]:
    """
    Infer (seed_id_col, label_col, time_col, start_node_type) for RelBench entity tasks
    from the split Table metadata.
    """
    df = tbl.df
    time_col = tbl.time_col
    fk_map = getattr(tbl, "fkey_col_to_pkey_table", {}) or {}

    # Entity tasks typically have exactly one FK: entity id -> entity table
    if len(fk_map) == 1:
        seed_col = next(iter(fk_map.keys()))
        start_node_type = fk_map[seed_col]
    else:
        # Fallback: pick a FK col if any; else error out cleanly.
        if len(fk_map) >= 1:
            seed_col = next(iter(fk_map.keys()))
            start_node_type = fk_map[seed_col]
        else:
            raise ValueError(
                "Could not infer seed/entity id column: split table has no fkey_col_to_pkey_table. "
                "This script currently targets entity (node) tasks."
            )

    candidates = [c for c in df.columns if c not in {seed_col, time_col}]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not uniquely infer label column. "
            f"Columns={list(df.columns)}, inferred seed={seed_col!r}, time={time_col!r}, "
            f"remaining={candidates}."
        )
    label_col = candidates[0]
    return seed_col, label_col, time_col, start_node_type

def balanced_sample(
    df: pd.DataFrame,
    label_col: str,
    take_all_label: int,
    take_n_label: int,
    take_n: int,
    *,
    random_state: int = 42,
    replace: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    if take_all_label == take_n_label:
        raise ValueError("take_all_label and take_n_label must be different.")
    if take_n < 0:
        raise ValueError("take_n must be >= 0.")

    all_part = df[df[label_col] == take_all_label]
    n_part = df[df[label_col] == take_n_label]

    if not replace and take_n > len(n_part):
        if strict:
            raise ValueError(
                f"Requested take_n={take_n} but only {len(n_part)} rows available "
                f"for label {take_n_label}. Set strict=False or replace=True."
            )
        take_n = len(n_part)

    n_sample = n_part.sample(n=take_n, replace=replace, random_state=random_state)

    out = (
        pd.concat([all_part, n_sample], axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return out


def proportional_sample(
    df: pd.DataFrame,
    label_col: str,
    total_n: int = 350_000,
    *,
    labels: Tuple[Union[int, bool], Union[int, bool]] = (0, 1),
    random_state: int = 42,
    replace: bool = False,
    strict: bool = True,
    shuffle: bool = True,
) -> pd.DataFrame:
    """
    Stratified fixed-size sample for binary labels.
    - Chooses n0 ~= total_n * (count0 / total), n1 = total_n - n0
    - Samples exactly n0 rows of label 0 and n1 rows of label 1 (unless strict=False and not enough rows)

    strict=True:
        - errors if df doesn't have both labels
        - errors if requested n exceeds available rows for a label (when replace=False)
    """
    if total_n <= 0:
        raise ValueError("total_n must be > 0")

    y = df[label_col]
    present = set(pd.unique(y.dropna()))
    a, b = labels

    if strict and (a not in present or b not in present):
        raise ValueError(f"Expected both labels {labels} in {label_col}, but found {sorted(present)}")

    total = len(df)
    if not replace and total_n > total:
        if strict:
            raise ValueError(f"Requested total_n={total_n} but df has only {total} rows (replace=False).")
        total_n = total  # best effort

    counts = y.value_counts(dropna=False)
    ca = int(counts.get(a, 0))
    cb = int(counts.get(b, 0))

    if ca + cb == 0:
        raise ValueError(f"No rows found for labels {labels} in column {label_col!r}")

    # Proportion from what actually exists (only considering a/b)
    denom = ca + cb
    na = int(round(total_n * (ca / denom)))
    nb = total_n - na  # ensures exact total_n

    part_a = df[y == a]
    part_b = df[y == b]

    if not replace:
        if strict and (na > len(part_a) or nb > len(part_b)):
            raise ValueError(
                f"Not enough rows for requested stratified sample: "
                f"need {na} of {a} (have {len(part_a)}), need {nb} of {b} (have {len(part_b)}). "
                f"Set replace=True or strict=False."
            )
        # best effort caps (keeps it from crashing when strict=False)
        na = min(na, len(part_a))
        nb = min(nb, len(part_b))

        # If we capped, fill the remainder from the other class if possible
        remainder = total_n - (na + nb)
        if remainder > 0:
            room_a = len(part_a) - na
            room_b = len(part_b) - nb
            add_a = min(remainder, room_a)
            na += add_a
            remainder -= add_a
            add_b = min(remainder, room_b)
            nb += add_b
            remainder -= add_b
            if strict and remainder > 0:
                raise ValueError("Could not reach total_n without replacement; not enough rows overall.")

    sa = part_a.sample(n=na, replace=replace, random_state=random_state) if na > 0 else part_a.iloc[:0]
    sb = part_b.sample(n=nb, replace=replace, random_state=random_state) if nb > 0 else part_b.iloc[:0]

    out = pd.concat([sa, sb], axis=0)
    if shuffle:
        out = out.sample(frac=1.0, random_state=random_state)
    return out.reset_index(drop=True)

def nonzero_plus_zero_sample(
    df: pd.DataFrame,
    y_col: str = "popularity",
    zero_n: int = 250_000,
    random_state: int = 42,
) -> pd.DataFrame:
    nonzero = df[df[y_col] != 0]
    zeros   = df[df[y_col].eq(0)]
    k = min(zero_n, len(zeros))
    zero_sample = zeros.sample(n=k, random_state=random_state)

    out = (
        pd.concat([nonzero, zero_sample], axis=0)
          .sample(frac=1.0, random_state=random_state)
          .reset_index(drop=True)
    )
    return out

def zero_nonzero_sample(
    df: pd.DataFrame,
    y_col: str,            # no default — caller must specify
    zero_n: int = 250_000,
    nonzero_n: int = 250_000,
    random_state: int = 42,
) -> pd.DataFrame:
    nonzero = df[df[y_col] != 0]
    zeros   = df[df[y_col] == 0]

    k_zero    = min(zero_n, len(zeros))
    k_nonzero = min(nonzero_n, len(nonzero))

    zero_sample    = zeros.sample(n=k_zero, random_state=random_state)
    nonzero_sample = nonzero.sample(n=k_nonzero, random_state=random_state)

    out = (
        pd.concat([nonzero_sample, zero_sample], axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return out

# --- NEW: infer schema for any entity/node task ---
id_col, label_col, time_col, inferred_start = _infer_entity_task_schema_from_split_table(train_table)

# If the task tells us the entity table, use it (works across datasets).
# START_NODE_TYPE = inferred_start
START_NODE_TYPE = getattr(task, "entity_table", inferred_start)


if TASK_NAME == "post-votes":
    train_sampled_df = nonzero_plus_zero_sample(train_table.df, y_col=label_col, zero_n=250_000, random_state=42)
elif TASK_NAME == "user-badge":
    train_sampled_df = balanced_sample(train_table.df, label_col=label_col, take_all_label=1, take_n_label=0, take_n=200_000)
elif TASK_NAME == "user-engagement":
    train_sampled_df = balanced_sample(train_table.df, label_col=label_col, take_all_label=1, take_n_label=0, take_n=100_000)
elif TASK_NAME == "driver-dnf":
    train_sampled_df = train_table.df
elif TASK_NAME == "driver-top3":
    train_sampled_df = train_table.df
elif TASK_NAME == "driver-position":
    train_sampled_df = train_table.df
elif TASK_NAME == "study-outcome":
    train_sampled_df = train_table.df
elif TASK_NAME == "study-adverse":
    train_sampled_df = train_table.df
elif TASK_NAME == "site-success":
    train_sampled_df = train_table.df
elif TASK_NAME == "user-clicks":
    train_sampled_df = train_table.df
elif TASK_NAME == "user-visits":
    train_sampled_df = train_table.df
elif TASK_NAME == "ad-ctr":
    train_sampled_df = train_table.df
elif TASK_NAME == "beer-churn":
    train_sampled_df = proportional_sample(train_table.df, label_col=label_col, total_n=300_000, random_state=42)
elif TASK_NAME == "user-churn":
    train_sampled_df = proportional_sample(train_table.df, label_col=label_col, total_n=300_000, random_state=42)
elif TASK_NAME == "user-count":
    train_sampled_df = zero_nonzero_sample(train_table.df, y_col=label_col, zero_n=150_000, nonzero_n=150_000, random_state=42)
else:
    train_sampled_df = train_table.df

# --- Standardize column names for SQL + downstream (fixes your timestamp error) ---
rename_map = {id_col: SEED_ID_COL, label_col: "label"}
if time_col is None:
    raise ValueError(
        f"Task split table has time_col=None for task={TASK_NAME!r}. "
        "This script expects per-example timestamps in train/val/test tables."
    )
if time_col != "timestamp":
    rename_map[time_col] = "timestamp"

train_sample_std = train_sampled_df.rename(columns=rename_map)

# IMPORTANT: IDs should be integer-like (BIGINT), not float.
train_sample_std[SEED_ID_COL] = train_sample_std[SEED_ID_COL].astype("int64")
train_sample_std["label"]     = train_sample_std["label"].astype("float64")

# -----------------------
# DuckDB setup
# -----------------------

def _default_relbench_cache_dir() -> str:
    return os.path.expanduser("~/.cache/relbench")

def _find_cached_duckdb_file(dataset_name: str) -> Optional[str]:
    """
    Try to find a DuckDB DB file for this dataset in the standard RelBench cache.
    RelBench datasets are cached to disk (usually at ~/.cache/relbench). 
    """
    base = os.path.join(_default_relbench_cache_dir(), dataset_name)
    if not os.path.isdir(base):
        return None

    patterns = [
        os.path.join(base, "*.duckdb"),
        os.path.join(base, "*.db"),
    ]
    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    # Prefer larger files (heuristic), in case both metadata db and full db exist.
    candidates = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0] if candidates else None

def _materialize_db_to_duckdb(db_obj, out_path: str) -> str:
    """
    Create a DuckDB file containing all tables from relbench Database object.
    Used as fallback if no cached *.db/*.duckdb exists.
    """
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    con_tmp = duckdb.connect(out_path)
    # write tables into 'main' schema
    for tname, tbl in db_obj.table_dict.items():
        con_tmp.register("_df_tmp", tbl.df)
        con_tmp.execute(f'CREATE OR REPLACE TABLE "{tname}" AS SELECT * FROM _df_tmp')
        con_tmp.unregister("_df_tmp")
    con_tmp.close()
    return out_path

# Prefer cached DB; fallback to building one locally.
stack_db_path = _find_cached_duckdb_file(DATASET_NAME)
if stack_db_path is None:
    # Fallback: build a local DB once
    db_fallback = dataset.get_db()
    stack_db_path = _materialize_db_to_duckdb(db_fallback, f"./datasets_cache/{DATASET_NAME}.duckdb")

work_db_path  = "./sql_features.duckdb"

con = duckdb.connect(work_db_path)

safe_path = stack_db_path.replace("'", "''")
con.execute(f"ATTACH '{safe_path}' AS stack (READ_ONLY)")

con.execute("DROP SCHEMA IF EXISTS frontiers CASCADE;")
con.execute("DROP SCHEMA IF EXISTS features  CASCADE;")
con.execute("CREATE SCHEMA frontiers;")
con.execute("CREATE SCHEMA features;")

con.register("train_sample", train_sample_std)

con.execute(f"""
    CREATE OR REPLACE TABLE seeds_all AS
    SELECT DISTINCT CAST({SEED_ID_COL} AS BIGINT) AS {SEED_ID_COL}
    FROM train_sample
""")

# --- build sample_events first ---
con.execute(f"""
    CREATE OR REPLACE TABLE sample_events AS
    SELECT
      CAST(ts.{SEED_ID_COL} AS BIGINT)   AS {SEED_ID_COL},
      CAST(ts.timestamp AS TIMESTAMP)   AS timestamp,
      CAST(ts.label AS DOUBLE)          AS label
    FROM train_sample ts
""")


## Changed
con.execute(f"""
    CREATE OR REPLACE TABLE seed_batches AS
    SELECT
      se.{SEED_ID_COL} AS {SEED_ID_COL},
      se.timestamp     AS timestamp,
      se.label         AS label,
      NTILE({TARGET_N_BATCHES})
        OVER (ORDER BY se.{SEED_ID_COL}, se.timestamp) AS batch_id
    FROM sample_events se
""")


# -----------------------
# Small helpers
# -----------------------

# These globals will be populated in __main__ after db is loaded:
TS_COL_MAP: Dict[str, Optional[str]] = {}
PKEY_COL_MAP: Dict[str, Optional[str]] = {}

def fk_col(rel: str) -> str:
    return rel.split("f2p_", 1)[1] if "f2p_" in rel else rel

def is_rev(rel: str) -> bool:
    return rel.startswith("rev_")

def path_to_name(path_triplets: MetaPath) -> str:
    parts = [path_triplets[0][0]]
    for (_, rel, dst) in path_triplets:
        parts.append(rel)
        parts.append(dst)
    return "__".join(parts)

def _hop_feature_name(path_triplets: MetaPath, dst_table: str, hop_idx: int) -> str:
    return f"{path_to_name(path_triplets)}__cnt_{dst_table}_h{hop_idx}"

def _split_schema_table(qualified: str) -> Tuple[str, str]:
    if "." not in qualified:
        return "main", qualified
    return qualified.split(".", 1)[0], qualified.split(".", 1)[1]

def _table_exists(con, qualified: str) -> bool:
    schema, tbl = _split_schema_table(qualified)
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        LIMIT 1
        """,
        [schema, tbl]
    ).fetchone()
    return row is not None

def _drop_tables(con, qualified_tables: List[str]):
    for t in qualified_tables:
        con.execute(f"DROP TABLE IF EXISTS {t}")

def _ensure_schemas(con, schemas: Iterable[str] = ("frontiers", "features")):
    for s in schemas:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")

def _forward_fks_for_prefix(record: Dict[str, Any]) -> List[str]:
    last_dst = record["prefix"][-1][2]
    fks = set()
    for cand in record.get("candidates", []):
        if getattr(cand, "kind", None) == "table":
            (src, rel, dst) = cand.edge
            if src == last_dst and not is_rev(rel):
                fks.add(fk_col(rel))
    return sorted(fks)

def hop1_required_seed_fks(
    cands_by_hop: Dict[int, List[Dict[str, Any]]],
    *,
    start_node_type: str,
) -> List[str]:
    fks = set()
    for rec in cands_by_hop.get(1, []):
        step = rec["prefix"][0]
        (src, rel, dst) = step
        if src != start_node_type:
            continue
        if not is_rev(rel):
            fks.add(fk_col(rel))
    return sorted(fks)

# -----------------------
# Frontier SQL builders
# -----------------------

def _time_filter_sql(table_alias: str, dst_table: str, ts_col_map: Dict[str, Optional[str]]) -> str:
    """
    Build the time constraint if dst_table has a time_col; else return "".
    RelBench allows non-temporal tables where time_col=None. :contentReference[oaicite:7]{index=7}
    """
    dst_ts = ts_col_map.get(dst_table, None)
    if dst_ts is None:
        return ""
    return f" AND CAST({table_alias}.{dst_ts} AS TIMESTAMP) < f.timestamp"

def _dst_join_pk(dst_table: str) -> str:
    """
    PK used ONLY for forward-edge joins into dst_table (FK -> PK).
    Must exist for any pkey table.
    """
    pk = PKEY_COL_MAP.get(dst_table, None)
    if pk is None:
        raise ValueError(
            f"Cannot do forward FK->PK join into table {dst_table!r} because it has no pkey_col."
        )
    return pk

def _dst_row_id(dst_table: str) -> str:
    """
    Column used ONLY to uniquely identify a row in dst_table for last_id.
    If table has no PK, use DuckDB's rowid pseudo-column (materialized tables). 
    """
    pk = PKEY_COL_MAP.get(dst_table, None)
    return pk if pk is not None else "rowid"

def _build_frontier_sql_h1(
    seeds_temp: str,
    step: EdgeType,
    ts_col_map: Dict[str, Optional[str]],
    forward_fks: List[str],
    *,
    id_col: str = PKEY_COL,
    start_node_type: str = START_NODE_TYPE,
    seed_id_col: str = SEED_ID_COL,
) -> str:
    (src, rel, dst) = step
    assert src == start_node_type, f"hop1 src={src} expected={start_node_type}"

    fk = fk_col(rel)
    row_id_col = _dst_row_id(dst)

    if is_rev(rel):
        join_pred = f"t1.{fk} = f.last_id"
    else:
        join_pk = _dst_join_pk(dst) 
        join_pred = f"t1.{join_pk} = f.fk__{fk}"

    extra_fk_selects = ", ".join([f"t1.{k} AS fk__{k}" for k in forward_fks]) if forward_fks else ""
    if extra_fk_selects:
        extra_fk_selects = ", " + extra_fk_selects

    time_pred = _time_filter_sql("t1", dst, ts_col_map)

    return f"""
    CREATE OR REPLACE TABLE {frontier_name([step])} AS
    SELECT DISTINCT
      f.{seed_id_col} AS {seed_id_col},
      f.timestamp,
      t1.{row_id_col} AS last_id
      {extra_fk_selects}
    FROM {seeds_temp} f
    JOIN stack.main."{dst}" t1
      ON {join_pred}
     {time_pred}
    """
def _build_extend_frontier_sql(
    prev_prefix: MetaPath,
    new_edge: EdgeType,
    ts_col_map: Dict[str, Optional[str]],
    next_forward_fks: List[str],
    *,
    id_col: str = PKEY_COL,
    seed_id_col: str = SEED_ID_COL,
) -> str:
    prev_front = frontier_name(prev_prefix)
    curr_prefix = prev_prefix + [new_edge]

    (_, rel, dst) = new_edge
    fk = fk_col(rel)
    row_id_col = _dst_row_id(dst)

    if is_rev(rel):
        join_pred = f"t.{fk} = f.last_id"
    else:
        join_pk = _dst_join_pk(dst)
        join_pred = f"t.{join_pk} = f.fk__{fk}"

    extra_fk_selects = ", ".join([f"t.{fk_} AS fk__{fk_}" for fk_ in next_forward_fks]) if next_forward_fks else ""
    if extra_fk_selects:
        extra_fk_selects = ", " + extra_fk_selects

    time_pred = _time_filter_sql("t", dst, ts_col_map)

    return f"""
    CREATE OR REPLACE TABLE {frontier_name(curr_prefix)} AS
    SELECT DISTINCT
      f.{seed_id_col} AS {seed_id_col},
      f.timestamp,
      t.{row_id_col} AS last_id
      {extra_fk_selects}
    FROM {prev_front} f
    JOIN stack.main."{dst}" t
      ON {join_pred}
     {time_pred}
    """

# -----------------------
# FK-carrying helpers
# -----------------------

def _column_exists(con, qualified_table: str, colname: str) -> bool:
    schema, tbl = _split_schema_table(qualified_table)
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? AND column_name = ?
        LIMIT 1
        """,
        [schema, tbl, colname]
    ).fetchone()
    return row is not None

def _list_fk_cols_on_frontier(con, prefix: MetaPath) -> List[str]:
    schema, tbl = _split_schema_table(frontier_name(prefix))
    rows = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? AND column_name LIKE 'fk__%%'
        ORDER BY column_name
    """, [schema, tbl]).fetchall()
    return [r[0].split("fk__", 1)[1] for r in rows]

def _ensure_grandparent_has_join_fk(
    con,
    seeds_temp: str,
    parent_prefix: MetaPath,
    ts_col_map: Dict[str, Optional[str]],
    *,
    id_col: str = PKEY_COL,
    start_node_type: str = START_NODE_TYPE,
    seed_id_col: str = SEED_ID_COL,
):
    if len(parent_prefix) < 1:
        return

    (_, rel, _) = parent_prefix[-1]
    if is_rev(rel):
        return

    gp = parent_prefix[:-1]
    if len(gp) == 0:
        return

    k = fk_col(rel)
    gp_front = frontier_name(gp)
    needed = f"fk__{k}"

    if not _column_exists(con, gp_front, needed):
        existing = set(_list_fk_cols_on_frontier(con, gp))
        union_fks = sorted(existing | {k})

        if len(gp) == 1:
            gp_sql = _build_frontier_sql_h1(
                seeds_temp, gp[-1], ts_col_map, union_fks,
                id_col=id_col, start_node_type=start_node_type, seed_id_col=seed_id_col
            )
        else:
            ggp = gp[:-1]
            gp_last = gp[-1]
            gp_sql = _build_extend_frontier_sql(
                ggp, gp_last, ts_col_map, union_fks,
                id_col=id_col, seed_id_col=seed_id_col
            )
        con.execute(gp_sql)

def _ensure_parent_has_fk(
    con,
    seeds_temp: str,
    parent_prefix: MetaPath,
    required_fk: str,
    ts_col_map: Dict[str, Optional[str]],
    *,
    id_col: str = PKEY_COL,
    start_node_type: str = START_NODE_TYPE,
    seed_id_col: str = SEED_ID_COL,
):
    parent_front = frontier_name(parent_prefix)
    needed_col = f"fk__{required_fk}"
    if _column_exists(con, parent_front, needed_col):
        return

    if len(parent_prefix) > 1:
        _ensure_grandparent_has_join_fk(
            con, seeds_temp, parent_prefix, ts_col_map,
            id_col=id_col, start_node_type=start_node_type, seed_id_col=seed_id_col
        )

    existing = set(_list_fk_cols_on_frontier(con, parent_prefix))
    union_fks = sorted(existing | {required_fk})

    if len(parent_prefix) == 1:
        sql = _build_frontier_sql_h1(
            seeds_temp=seeds_temp,
            step=parent_prefix[-1],
            ts_col_map=ts_col_map,
            forward_fks=union_fks,
            id_col=id_col,
            start_node_type=start_node_type,
            seed_id_col=seed_id_col,
        )
    else:
        grandparent = parent_prefix[:-1]
        last_edge = parent_prefix[-1]
        sql = _build_extend_frontier_sql(
            grandparent, last_edge, ts_col_map, union_fks,
            id_col=id_col, seed_id_col=seed_id_col
        )

    con.execute(sql)

# -----------------------
# Feature strategy hook
# -----------------------

def _materialize_feature_for_prefix(con, prefix: MetaPath, strategy: FeatureStrategy):
    tbl = strategy.feature_table_name(prefix)
    if not _table_exists(con, tbl):
        con.execute(strategy.build_sql(prefix))
        
def export_next_hop_features_to_csv(
    con,
    *,
    seeds_temp: str,          # e.g., "seeds_frontier0"
    prefix_record: dict,
    strategy: FeatureStrategy,
    out_path: str,
    seed_id_col: str = SEED_ID_COL,
) -> str:
    """
    Writes one CSV per prefix WITHOUT Pandas:
    COPY (SELECT ... LEFT JOIN feature tables ...) TO 'file.csv' (FORMAT CSV, HEADER true)

    Returns basename (for logging if you want).
    """
    prefix: MetaPath = prefix_record["prefix"]
    basename = path_to_name(prefix)
    next_hop_idx = len(prefix) + 1

    table_cands = [
        c for c in prefix_record.get("candidates", [])
        if getattr(c, "kind", None) == "table"
    ]

    # Base columns (keep stable order)
    select_exprs = [
        f"u.{seed_id_col} AS {seed_id_col}",
        "CAST(u.timestamp AS TIMESTAMP) AS timestamp",
        "u.label AS label",
    ]
    join_clauses = []

    # If your strategy output should default to 0/0.0 when missing:
    default_literal = "0.0"
    # If you *really* need int zeros for CountDistinctStrategy:
    if isinstance(strategy, CountDistinctStrategy):
        default_literal = "0"

    feat_idx = 0
    for cand in table_cands:
        path_next: MetaPath = prefix + [cand.edge]
        feat_tbl = strategy.feature_table_name(path_next)
        if not _table_exists(con, feat_tbl):
            continue

        feat_col = strategy.export_column_name(path_next)
        last_dst = path_next[-1][2]
        out_col = _hop_feature_name(path_next, last_dst, next_hop_idx)

        alias = f"f{feat_idx}"
        feat_idx += 1

        # COALESCE to avoid NULLs in the CSV
        select_exprs.append(
            f"COALESCE({alias}.{feat_col}, {default_literal}) AS \"{out_col}\""
        )

        join_clauses.append(
            f"""
            LEFT JOIN {feat_tbl} {alias}
              ON {alias}.{seed_id_col} = u.{seed_id_col}
             AND {alias}.timestamp = u.timestamp
            """.strip()
        )

    sql = f"""
    COPY (
        SELECT
            {",\n            ".join(select_exprs)}
        FROM {seeds_temp} u
        {" ".join(join_clauses)}
    )
    TO '{out_path.replace("'", "''")}'
    (FORMAT CSV, HEADER true);
    """

    # DuckDB COPY supports exporting query results to CSV. :contentReference[oaicite:1]{index=1}
    con.execute(sql)
    return basename


def materialize_all_frontiers(
    con,
    feature_strategies,
    seeds_temp: str,
    cands_by_hop: Dict[int, List[Dict[str, Any]]],
    ts_col_map: Dict[str, Optional[str]],
    *,
    id_col: str = PKEY_COL,
    start_node_type: str = START_NODE_TYPE,
    seed_id_col: str = SEED_ID_COL,
    max_hop: int = 4,
    tqdm_prefix: str = "",
    tqdm_position: Optional[int] = None,
    tqdm_enabled: bool = True,
):
    _ensure_schemas(con, ("frontiers", "features"))

    hop_iter = tqdm(
        range(1, max_hop + 1),
        desc=f"{tqdm_prefix} materialize hops",
        unit="hop",
        position=tqdm_position,
        leave=False,
        disable=not tqdm_enabled,
    )
    frontiers_by_hop = defaultdict(list) 
    for hop in hop_iter:
        recs = cands_by_hop.get(hop, [])
        rec_iter = tqdm(
            recs,
            total=len(recs),
            desc=f"{tqdm_prefix} h{hop} prefixes",
            unit="prefix",
            position=None if tqdm_position is None else (tqdm_position + 1),
            leave=False,
            disable=not tqdm_enabled,
        )

        for rec in rec_iter:
            prefix: MetaPath = rec["prefix"]
            last_edge = prefix[-1]
            fwd_fks = _forward_fks_for_prefix(rec)

            if hop == 1:
                sql = _build_frontier_sql_h1(
                    seeds_temp, last_edge, ts_col_map, fwd_fks,
                    id_col=id_col, start_node_type=start_node_type, seed_id_col=seed_id_col
                )
                con.execute(sql)
            else:
                parent_prefix = prefix[:-1]
                (_, rel, _) = last_edge
                if not is_rev(rel):
                    _ensure_parent_has_fk(
                        con=con,
                        seeds_temp=seeds_temp,
                        parent_prefix=parent_prefix,
                        required_fk=fk_col(rel),
                        ts_col_map=ts_col_map,
                        id_col=id_col,
                        start_node_type=start_node_type,
                        seed_id_col=seed_id_col,
                    )

                sql = _build_extend_frontier_sql(
                    parent_prefix, last_edge, ts_col_map, fwd_fks,
                    id_col=id_col, seed_id_col=seed_id_col
                )
                con.execute(sql)

            frontiers_by_hop[hop].append(frontier_name(prefix))
            
            for strat in feature_strategies:
                _materialize_feature_for_prefix(con, prefix, strat)
        
        old_hop = hop - 2
        if old_hop >= 1 and frontiers_by_hop.get(old_hop):
            _drop_tables(con, frontiers_by_hop[old_hop])
            frontiers_by_hop[old_hop].clear()

        rec_iter.close()
    for h in list(frontiers_by_hop.keys()):
        _drop_tables(con, frontiers_by_hop[h])
        frontiers_by_hop[h].clear()
    

def assemble_next_hop_features_from_persisted(
    con,
    seeds_temp: str,
    prefix_record: dict,
    strategy: FeatureStrategy,
    *,
    seed_id_col: str = SEED_ID_COL,
) -> Tuple[pd.DataFrame, str]:
    key_cols = [seed_id_col, "timestamp", "label"]

    base_df = con.execute(f"""
        SELECT {seed_id_col}, CAST(timestamp AS TIMESTAMP) AS timestamp, label
        FROM {seeds_temp}
    """).fetch_df()

    out = base_df.copy()
    prefix: MetaPath = prefix_record["prefix"]
    basename = path_to_name(prefix)
    next_hop_idx = len(prefix) + 1

    table_cands = [
        c for c in prefix_record.get("candidates", [])
        if getattr(c, "kind", None) == "table"
    ]

    for cand in table_cands:
        path_next: MetaPath = prefix + [cand.edge]
        feat_tbl = strategy.feature_table_name(path_next)
        feat_col = strategy.export_column_name(path_next)
        last_dst = path_next[-1][2]
        out_col = _hop_feature_name(path_next, last_dst, next_hop_idx)

        if _table_exists(con, feat_tbl):
            df = con.execute(f"""
                SELECT u.{seed_id_col},
                       CAST(u.timestamp AS TIMESTAMP) AS timestamp,
                       u.label,
                       f.{feat_col}
                FROM {seeds_temp} u
                LEFT JOIN {feat_tbl} f
                  ON f.{seed_id_col} = u.{seed_id_col}
                 AND f.timestamp = u.timestamp
            """).fetch_df()

            df = df[key_cols + [feat_col]].rename(columns={feat_col: out_col})
            out = out.merge(df, on=key_cols, how="left")

            if isinstance(strategy, CountDistinctStrategy):
                out[out_col] = out[out_col].fillna(0).astype("int64")
            else:
                out[out_col] = out[out_col].fillna(0.0).astype("float64")

    return out, basename

def run_one_batch(
    batch_id: int,
    *,
    con,
    cands_by_hop,
    ts_col_map,
    out_root: str,
    feature_strategies,
    id_col: str = PKEY_COL,
    start_node_type: str = START_NODE_TYPE,
    seed_id_col: str = SEED_ID_COL,
    max_hop: int = 4,
    hops_to_export=(1, 2, 3),
    progress: bool = True,
    bar_position: Optional[int] = None,
):
    con.execute("DROP SCHEMA IF EXISTS frontiers CASCADE;")
    con.execute("DROP SCHEMA IF EXISTS features  CASCADE;")
    con.execute("CREATE SCHEMA frontiers;")
    con.execute("CREATE SCHEMA features;")

    ## Changed
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW seeds_temp_batch AS
        SELECT {seed_id_col}, timestamp, label
        FROM seed_batches
        WHERE batch_id = {batch_id}
    """)

    seed_fks = hop1_required_seed_fks(cands_by_hop, start_node_type=start_node_type)

    start_pk = _dst_join_pk(start_node_type)
    start_ts = ts_col_map.get(start_node_type, None)

    if seed_fks:
        fk_selects = ",\n                ".join([f"s0.{k} AS fk__{k}" for k in seed_fks])

        # If start table is non-temporal (time_col=None), don't filter by time.
        time_where = ""
        if start_ts is not None:
            time_where = f"WHERE CAST(s0.{start_ts} AS TIMESTAMP) < se.timestamp"

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE seeds_frontier0 AS
            SELECT
                se.{seed_id_col} AS {seed_id_col},
                se.timestamp,
                se.label,
                se.{seed_id_col} AS last_id,
                {fk_selects}
            FROM seeds_temp_batch se
            JOIN stack.main."{start_node_type}" s0
              ON s0.{start_pk} = se.{seed_id_col}
            {time_where}
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW seeds_frontier0 AS
            SELECT
                {seed_id_col},
                timestamp,
                label,
                {seed_id_col} AS last_id
            FROM seeds_temp_batch
        """)

    start_time = time.time()

    materialize_all_frontiers(
        con=con,
        feature_strategies=feature_strategies,
        seeds_temp="seeds_frontier0",
        cands_by_hop=cands_by_hop,
        ts_col_map=ts_col_map,
        id_col=id_col,
        start_node_type=start_node_type,
        seed_id_col=seed_id_col,
        max_hop=max_hop,
        tqdm_prefix=f"[b{batch_id}]",
        tqdm_position=bar_position,
        tqdm_enabled=progress,
    )

    strategy = feature_strategies[0]

    hop_bar = tqdm(
        hops_to_export,
        desc=f"[b{batch_id}] export hops",
        unit="hop",
        position=bar_position,
        leave=False,
        disable=not progress,
    )

    for hop in hop_bar:
        batch_dir = os.path.join(out_root, f"batch_{batch_id}", f"hop_{hop}")
        os.makedirs(batch_dir, exist_ok=True)

        recs = cands_by_hop.get(hop, [])
        rec_bar = tqdm(
            recs,
            total=len(recs),
            desc=f"[b{batch_id}] h{hop} export prefixes",
            unit="prefix",
            position=None if bar_position is None else (bar_position + 1),
            leave=False,
            disable=not progress,
        )

        for rec in rec_bar:
            out_path = os.path.join(batch_dir, f"{path_to_name(rec['prefix'])}.csv")
            export_next_hop_features_to_csv(
                con,
                seeds_temp="seeds_frontier0",
                prefix_record=rec,
                strategy=strategy,
                out_path=out_path,
                seed_id_col=SEED_ID_COL,
            )

        rec_bar.close()

    elapsed = time.time() - start_time
    # n_seeds = con.execute(
    #     f"SELECT COUNT(DISTINCT {seed_id_col}) FROM seeds_temp_batch"
    # ).fetchone()[0]
    # tqdm.write(f"[Batch {batch_id}] seeds={n_seeds}  time={elapsed:.2f}s")
    n_items = con.execute("SELECT COUNT(*) FROM seeds_temp_batch").fetchone()[0]
    n_unique_seeds = con.execute(
        f"SELECT COUNT(DISTINCT {seed_id_col}) FROM seeds_temp_batch"
    ).fetchone()[0]

    tqdm.write(f"[Batch {batch_id}] items={n_items} seeds={n_unique_seeds} time={elapsed:.2f}s")
    


def _run_batch_in_isolated_connection(
    batch_id: int,
    *,
    stack_db_path: str,
    shared_db_path: str,
    out_root: str,
    cands_by_hop,
    ts_col_map,
    feature_strategies,
    id_col: str = PKEY_COL,
    start_node_type: str = START_NODE_TYPE,
    seed_id_col: str = SEED_ID_COL,
    max_hop: int = 4,
    hops_to_export=(1, 2, 3),
    worker_slots: int = 8,
) -> int:
    import os
    import shutil
    import tempfile

    # -----------------------
    # Compute per-worker CPU + memory budgets
    # -----------------------
    total_cpus = os.cpu_count() or 1
    # Target: evenly split cores across NUM_WORKERS (=worker_slots)
    threads_per_worker = max(1, total_cpus // max(1, worker_slots))
    # Practical clamp (avoid crazy oversubscription overhead)
    threads_per_worker = min(threads_per_worker, 32)

    # Read system memory (Linux) and split across workers (keep headroom).
    def _read_memtotal_bytes() -> int | None:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            return None
        return None

    mem_total = _read_memtotal_bytes()
    # Give each worker ~85%/worker_slots of RAM. If unknown, skip memory_limit.
    mem_limit_bytes = None
    if mem_total is not None:
        mem_limit_bytes = int(mem_total * 0.85 / max(1, worker_slots))
        # Don't set absurdly low limits
        # mem_limit_bytes = max(mem_limit_bytes, 4 * 1024**3)  # >= 4GB
        mem_limit_bytes = max(mem_limit_bytes, 1 * 1024**3)

    # Per-worker spill dir (important when DuckDB spills to disk during GROUP BY/JOIN/etc.)
    # spill_dir = tempfile.mkdtemp(prefix=f"duckdb_spill_b{batch_id}_")
    spill_dir = tempfile.mkdtemp(prefix=f"duckdb_spill_b{batch_id}_", dir=SPILL_ROOT)

    # con_local = duckdb.connect(":memory:")
    db_path = os.path.join(spill_dir, f"worker_{batch_id}.duckdb")
    con_local = duckdb.connect(db_path)
    try:
        # -----------------------
        # DuckDB knobs
        # -----------------------
        con_local.execute(f"SET threads = {threads_per_worker}")
        con_local.execute("SET preserve_insertion_order = false")
        con_local.execute(f"SET temp_directory = '{spill_dir}'")
        con_local.execute("SET enable_progress_bar = false")

        if mem_limit_bytes is not None:
            mem_gb = max(1, mem_limit_bytes // (1024**3))
            con_local.execute(f"SET memory_limit = '{mem_gb}GB'")

        # -----------------------
        # Attach DBs
        # -----------------------
        safe_stack = stack_db_path.replace("'", "''")
        safe_shared = shared_db_path.replace("'", "''")

        # Smaller row groups can improve parallelism for queries that scan "not that many" rows per operator.
        # (Optional but often helpful when you increase threads.) :contentReference[oaicite:1]{index=1}
        con_local.execute(
            f"ATTACH '{safe_stack}'  AS stack  (READ_ONLY, ROW_GROUP_SIZE 16384)"
        )
        con_local.execute(
            f"ATTACH '{safe_shared}' AS shared (READ_ONLY)"
        )

        con_local.execute(
            "CREATE OR REPLACE TEMP VIEW sample_events AS SELECT * FROM shared.sample_events"
        )
        con_local.execute(
            "CREATE OR REPLACE TEMP VIEW seed_batches  AS SELECT * FROM shared.seed_batches"
        )

        base_pos = 2 + ((batch_id - 1) % max(1, worker_slots)) * 2

        run_one_batch(
            batch_id=batch_id,
            con=con_local,
            cands_by_hop=cands_by_hop,
            ts_col_map=ts_col_map,
            out_root=out_root,
            feature_strategies=feature_strategies,
            id_col=id_col,
            start_node_type=start_node_type,
            seed_id_col=seed_id_col,
            max_hop=max_hop,
            hops_to_export=hops_to_export,
            progress=True,
            bar_position=base_pos,
        )
        return batch_id
    finally:
        try:
            con_local.close()
        finally:
            shutil.rmtree(spill_dir, ignore_errors=True)


# -----------------------
# Candidate generation
# -----------------------

def gather_candidates_by_parent_hop(
    mp_by_hop,
    data,
    voc=None,
    hops=(1, 2, 3),
    allowed_tables: Optional[set] = None,
    as_mapping_per_hop: bool = False,
    avoid_immediate_backtrack: bool = True,
):
    if as_mapping_per_hop:
        out = {h: {} for h in hops}
    else:
        out = {h: [] for h in hops}

    for h in hops:
        for prefix in mp_by_hop.get(h, []):
            cands = generate_candidates_for_prefix(
                prefix,
                data,
                voc=voc,
                allowed_tables=allowed_tables,
                avoid_immediate_backtrack=avoid_immediate_backtrack,
            )
            record = {"prefix": prefix, "candidates": cands}

            if as_mapping_per_hop:
                out[h][_prefix_key(prefix)] = record
            else:
                out[h].append(record)

    return out

# -----------------------
# Script entry point
# -----------------------

if __name__ == "__main__":
    db = dataset.get_db()
    # --- Populate global PK/time maps (general across all datasets) ---
    TS_COL_MAP = {tname: tbl.time_col for tname, tbl in db.table_dict.items()}
    PKEY_COL_MAP = {tname: tbl.pkey_col for tname, tbl in db.table_dict.items()}
    
    col_to_stype_dict = get_stype_proposal(db)
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256,
    )

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=os.path.join(root_dir, f"{DATASET_NAME}_glove_materialized_cache"),
    )

    mp_by_hop = enumerate_metapaths_from(
        data,
        start_node_type=START_NODE_TYPE,
        max_hops=_CLI.max_hop,
        avoid_immediate_backtrack=True,
        dedupe_by_base_rel=False,
    )

    voc = build_schema_vocabs(data)

    # --- Generalize: allow all tables in this dataset (instead of rel-stack-only list) ---
    allowed_tables = set(db.table_dict.keys())

    cands_by_hop = gather_candidates_by_parent_hop(
        mp_by_hop, data, voc, hops=tuple(range(1, _CLI.max_hop + 1)), allowed_tables=allowed_tables
    )


    n_batches = con.execute("SELECT MAX(batch_id) FROM seed_batches").fetchone()[0]

    total_items = con.execute("SELECT COUNT(*) FROM sample_events").fetchone()[0]
    total_unique_seeds = con.execute(
        f"SELECT COUNT(DISTINCT {SEED_ID_COL}) FROM sample_events"
    ).fetchone()[0]

    print(
        f"Prepared {n_batches} batches for {total_items} items "
        f"across {total_unique_seeds} seeds "
        f"(~{math.ceil(total_items / n_batches)} items/batch)."
    )

    con.close()

    grand_start = time.time()

    NUM_WORKERS = _CLI.num_workers

    strategy_runs = [
        (
            (LogCountStrategy(seed_id_col=SEED_ID_COL),),
            f"./outputs/{DATASET_NAME}/{TASK_NAME}/log_count/train",
        ),
        (
            (LogRateStrategy(seed_id_col=SEED_ID_COL),),
            f"./outputs/{DATASET_NAME}/{TASK_NAME}/log_rate/train",
        ),
    ]

    for feature_strategies, out_root in strategy_runs:
        strat_name = feature_strategies[0].name
        print(f"\n{'='*60}")
        print(f"Running strategy: {strat_name}")
        print(f"Output: {out_root}")
        print(f"{'='*60}")

        Path(out_root).mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = [
                pool.submit(
                    _run_batch_in_isolated_connection,
                    b,
                    stack_db_path=stack_db_path,
                    shared_db_path=work_db_path,
                    out_root=out_root,
                    cands_by_hop=cands_by_hop,
                    ts_col_map=TS_COL_MAP,
                    feature_strategies=feature_strategies,
                    id_col=PKEY_COL,
                    start_node_type=START_NODE_TYPE,
                    seed_id_col=SEED_ID_COL,
                    max_hop=_CLI.max_hop,
                    hops_to_export=tuple(_CLI.hops_to_export),
                    worker_slots=NUM_WORKERS,
                )
                for b in range(1, n_batches + 1)
            ]

            with tqdm(total=len(futures), desc=f"Batches ({strat_name})", unit="batch", dynamic_ncols=True) as pbar:
                for f in as_completed(futures):
                    try:
                        done_b = f.result()
                        tqdm.write(f"[{strat_name}] Batch {done_b} finished.")
                    except Exception as e:
                        tqdm.write(f"[{strat_name}] Worker failed: {e}")
                    finally:
                        pbar.update(1)

        print(f"Strategy {strat_name} done.")

    agg_elapsed = time.time() - grand_start
    print(f"\nAll strategies done in {agg_elapsed:.2f}s")
    print(f"AGGREGATE_TIME_SECONDS={agg_elapsed:.2f}")