import os
import re
import torch
import argparse
import datetime as dt
import pandas as pd
from pathlib import Path
from pandas.api.types import is_bool_dtype

from mps_bridge import _build_type_offsets, relbench_hetero_to_homo_for_mps
from embeddings import GloveTextEmbedding
from mps_wrapper import run_mps_metapath_search

from relbench.tasks import get_task
from relbench.datasets import get_dataset
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from torch_geometric.loader import NeighborLoader
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input


def bool_label_to_int_inplace(table, label_col):
    if label_col not in table.df.columns:
        return
    s = table.df[label_col]
    if is_bool_dtype(s) or s.dropna().isin([True, False]).all():
        table.df[label_col] = s.map({False: 0, True: 1}).astype("Int8")


def make_temporal_mps_objects(
    data_homo,
    data_hetero,
    task,
    train_table, train_sampled_df,
    val_table,   val_sampled_df,
    num_neighbors,
):
    offsets = _build_type_offsets(data_hetero, data_homo)

    tr_in = get_node_train_table_input(train_table, task)
    va_in = get_node_train_table_input(val_table, task)

    entity_type = tr_in.nodes[0]
    assert entity_type == va_in.nodes[0]

    tr_idx = torch.tensor(train_sampled_df.index.to_numpy(), dtype=torch.long)
    va_idx = torch.tensor(val_sampled_df.index.to_numpy(),   dtype=torch.long)

    tr_seed = tr_in.nodes[1][tr_idx].to(torch.long)
    va_seed = va_in.nodes[1][va_idx].to(torch.long)

    tr_time = tr_in.time[tr_idx]
    va_time = va_in.time[va_idx]

    tr_y = torch.tensor(train_sampled_df[task.target_col].to_numpy(), dtype=torch.long)
    va_y = torch.tensor(val_sampled_df[task.target_col].to_numpy(),   dtype=torch.long)

    tr_gidx = tr_seed + offsets[entity_type]
    va_gidx = va_seed + offsets[entity_type]

    seed_nodes = torch.cat([tr_gidx, va_gidx], dim=0)
    seed_times = torch.cat([tr_time, va_time], dim=0)

    n_train = tr_y.numel()
    n_total = n_train + va_y.numel()

    # Disjoint temporal sampling: each row instance gets its own component
    loader = NeighborLoader(
        data_homo,
        input_nodes=seed_nodes,
        input_time=seed_times,
        time_attr="time",
        temporal_strategy="uniform",
        num_neighbors=num_neighbors,
        batch_size=n_total,
        shuffle=False,
        disjoint=True,
        subgraph_type="directional",     # avoid induced+time bug
        num_workers=0,
        persistent_workers=False,
    )
    batch = next(iter(loader))
    batch = batch.cpu()

    assert batch.batch_size == n_total
    seed_local = torch.arange(n_total, dtype=torch.long)

    batch.train_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
    batch.val_mask   = torch.zeros(batch.num_nodes, dtype=torch.bool)
    batch.test_mask  = torch.zeros(batch.num_nodes, dtype=torch.bool)

    batch.train_mask[seed_local[:n_train]] = True
    batch.val_mask[seed_local[n_train:]]   = True

    y_all = torch.cat([tr_y, va_y], dim=0)
    batch.y = torch.zeros(batch.num_nodes, dtype=torch.long)
    batch.y[seed_local] = y_all
    batch.mpgnn_y = batch.y.clone()

    train_targets = seed_local[:n_train]
    bags  = [[int(i)] for i in train_targets.tolist()]
    alpha = {(int(i), (int(i),)): 1.0 for i in train_targets.tolist()}

    data_score = Data(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_type=batch.edge_type,
        y=tr_y.cpu(),
        num_nodes=batch.num_nodes,
        bags=bags,
        alpha=alpha,
        target_ids=train_targets.tolist()
    )

    data_score.edge_type_names = getattr(data_homo, "edge_type_names", None)
    batch.edge_type_names      = getattr(data_homo, "edge_type_names", None)
    data_score.node_type_names = getattr(data_homo, "node_type_names", None)
    batch.node_type_names      = getattr(data_homo, "node_type_names", None)

    return data_score, batch


def sample_n(
    df: pd.DataFrame,
    label_col: str,
    pos_label: int,
    neg_label: int,
    n_pos: int,
    n_neg: int,
    *,
    random_state: int = 42,
    replace_pos: bool = False,
    replace_neg: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    if pos_label == neg_label:
        raise ValueError("pos_label and neg_label must be different.")
    if n_pos < 0 or n_neg < 0:
        raise ValueError("n_pos and n_neg must be >= 0.")

    pos_part = df[df[label_col] == pos_label]
    neg_part = df[df[label_col] == neg_label]

    if not replace_pos and n_pos > len(pos_part):
        if strict:
            raise ValueError(
                f"Requested n_pos={n_pos} but only {len(pos_part)} rows available "
                f"for label {pos_label}. Set strict=False or replace_pos=True."
            )
        n_pos = len(pos_part)

    if not replace_neg and n_neg > len(neg_part):
        if strict:
            raise ValueError(
                f"Requested n_neg={n_neg} but only {len(neg_part)} rows available "
                f"for label {neg_label}. Set strict=False or replace_neg=True."
            )
        n_neg = len(neg_part)

    pos_sample = pos_part.sample(n=n_pos, replace=replace_pos, random_state=random_state) if n_pos else pos_part.iloc[0:0]
    neg_sample = neg_part.sample(n=n_neg, replace=replace_neg, random_state=random_state) if n_neg else neg_part.iloc[0:0]

    out = pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1.0, random_state=random_state)
    return out


def seed_ids_from_sampled_df(split_table, sampled_df, task):
    table_input = get_node_train_table_input(split_table, task)
    entity_type, seed_ids_all = table_input.nodes

    idx = torch.tensor(sampled_df.index.to_numpy(), dtype=torch.long)
    seed_ids = seed_ids_all[idx].to(torch.long).cpu()
    return entity_type, seed_ids


def extract_ranked_results(log_path: Path, txt_path: Path) -> int:
    pat = re.compile(r"\(path=\[.*?\],\s*hop=\d+,\s*sort_by=[^)]+\)\s*rank=\d+\s+\S+=\S+")
    matched = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if pat.search(line):
                matched.append(line)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(matched)
    return len(matched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPS-GNN metapath search")
    parser.add_argument("--dataset",              type=str,   required=True,                   help="Relbench dataset name (e.g. rel-stack)")
    parser.add_argument("--task",                 type=str,   required=True,                   help="Task name (e.g. user-badge)")
    parser.add_argument("--task_kind",            type=str,   default="classification",         choices=["classification", "regression"])
    parser.add_argument("--channels",             type=int,   default=256,                     help="Hidden embedding dimension")
    parser.add_argument("--num_layers",           type=int,   default=4,                       help="Number of GNN layers / max hop depth")
    parser.add_argument("--num_neighbors",        type=int,   default=32,                      help="Neighbors sampled per layer")
    parser.add_argument("--text_emb_batch_size",  type=int,   default=256)
    parser.add_argument("--n_pos_train",          type=int,   default=250,                     help="Positive training samples")
    parser.add_argument("--n_neg_train",          type=int,   default=250,                     help="Negative training samples")
    parser.add_argument("--n_pos_val",            type=int,   default=60,                      help="Positive validation samples")
    parser.add_argument("--n_neg_val",            type=int,   default=60,                      help="Negative validation samples")
    parser.add_argument("--search_l_max",         type=int,   default=3,                       help="Maximum metapath length")
    parser.add_argument("--search_k",             type=int,   default=3,                       help="Top-K metapaths to keep per hop")
    parser.add_argument("--search_beam_width",    type=int,   default=6,                       help="Beam width for metapath search")
    parser.add_argument("--search_sampling_size", type=int,   default=1,                       help="Sampling size for MPS evaluation")
    parser.add_argument("--search_rank_metric",   type=str,   default="auc_roc",               help="Metric used to rank metapaths")
    parser.add_argument("--device",               type=str,   default="cuda:0")
    parser.add_argument("--data_dir",             type=str,   default="./data",                help="Directory for dataset cache")
    args = parser.parse_args()

    seed_everything(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log_dir = Path("search_results") / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / f"mps_search_{args.task}.log").resolve()
    txt_path = (log_dir / f"mps_search_{args.task}.txt").resolve()

    dataset = get_dataset(args.dataset, download=True)
    task    = get_task(args.dataset, args.task, download=False)

    train_table = task.get_table("train")
    val_table   = task.get_table("val")
    label_col   = task.target_col

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=args.text_emb_batch_size,
    )

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=os.path.join(args.data_dir, f"{args.dataset}_glove_materialized_cache"),
    )

    data_homo = relbench_hetero_to_homo_for_mps(
        data_hetero=data,
        col_stats_dict=col_stats_dict,
        channels=args.channels,
        device=torch.device("cpu"),
    )

    train_sampled_df = sample_n(
        train_table.df, label_col,
        pos_label=1, neg_label=0,
        n_pos=args.n_pos_train, n_neg=args.n_neg_train,
    )
    val_sampled_df = sample_n(
        val_table.df, label_col,
        pos_label=1, neg_label=0,
        n_pos=args.n_pos_val, n_neg=args.n_neg_val,
    )

    num_neighbors = [args.num_neighbors] * args.num_layers

    data_score, data_mpgnn = make_temporal_mps_objects(
        data_homo=data_homo,
        data_hetero=data,
        task=task,
        train_table=train_table,
        train_sampled_df=train_sampled_df,
        val_table=val_table,
        val_sampled_df=val_sampled_df,
        num_neighbors=num_neighbors,
    )

    print("train bags:", len(data_score.target_ids), "val seeds:", int(data_mpgnn.val_mask.sum()))
    print("data_score y pos/neg:",
          int((data_score.y == 1).sum()), int((data_score.y == 0).sum()))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[RUN_START] {dt.datetime.now().isoformat()}\n")
        f.write(f"DATASET_NAME={args.dataset}\n")
        f.write(f"TASK_NAME={args.task}\n")
        f.write(f"NUM_LAYERS={args.num_layers}\n")
        f.write(f"num_neighbors={num_neighbors}\n")
        f.write(f"train_sample: n_pos={args.n_pos_train}, n_neg={args.n_neg_train}\n")
        f.write(f"val_sample:   n_pos={args.n_pos_val}, n_neg={args.n_neg_val}\n")
        f.write(f"SEARCH_L_MAX={args.search_l_max}\n")
        f.write(f"SEARCH_K={args.search_k}\n")
        f.write(f"SEARCH_BEAM_WIDTH={args.search_beam_width}\n")
        f.write(f"SEARCH_SAMPLING_SIZE={args.search_sampling_size}\n")
        f.write(f"SEARCH_RANK_METRIC={args.search_rank_metric}\n")
        f.write("-" * 60 + "\n")

    best_mps = run_mps_metapath_search(
        data_score=data_score,
        data_mpgnn=data_mpgnn,
        hidden_dim=args.channels,
        eval_split="val",
        rank_metric=args.search_rank_metric,
        device=device,
        log_path=str(log_path),
        log_to_console=False,
        L_MAX=args.search_l_max,
        K=args.search_k,
        beam_width=args.search_beam_width,
        sampling_size=args.search_sampling_size,
    )

    print(best_mps)

    n = extract_ranked_results(log_path, txt_path)
    print(f"Ranked results ({n} metapaths) saved to: {txt_path}")
