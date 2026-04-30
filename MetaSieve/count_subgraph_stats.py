# count_subgraph_stats.py

import os
import ast
import json
import argparse
import importlib.util
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import warnings
from torch_geometric.seed import seed_everything
from torch_geometric.sampler.base import NumNeighbors
from torch_geometric.loader import NeighborLoader

from embeddings import GloveTextEmbedding
from relbench.tasks import get_task
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from torch_frame.config.text_embedder import TextEmbedderConfig

warnings.filterwarnings("once")
seed_everything(42)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Subgraph node/edge counter")
    p.add_argument("--dataset_name",        type=str,   default="rel-stack")
    p.add_argument("--task_name",           type=str,   default="user-engagement")
    p.add_argument("--num_layers",          type=int,   default=4)
    p.add_argument("--batch_size",          type=int,   default=512)
    p.add_argument("--text_emb_batch_size", type=int,   default=256)
    p.add_argument("--fanout",              type=int,   default=64)
    p.add_argument("--sampling_cfg_path",   type=str,   default="")
    p.add_argument("--output_root",         type=str,   default="outputs")
    p.add_argument("--device",              type=str,   default="cuda:0")
    return p.parse_args()


# ── Sampling config loaders (copied from train_stepwise.py) ───────────────────

def num_neighbors_from_json(cfg_path) -> NumNeighbors:
    cfg = json.loads(Path(cfg_path).read_text())
    L    = cfg["num_layers"]
    base = cfg["base_neighbors"]
    neighbors = {ast.literal_eval(k): v for k, v in cfg["neighbors"].items()}
    return NumNeighbors(neighbors, default=[base] * L)


def num_neighbors_from_py(cfg_path) -> NumNeighbors:
    cfg_path = Path(cfg_path)
    module_name = f"sampling_cfg_{cfg_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(cfg_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "num_neighbors"):
        raise AttributeError(f"{cfg_path} must define a top-level variable `num_neighbors`")
    nn = module.num_neighbors
    if isinstance(nn, NumNeighbors):
        return nn
    if isinstance(nn, dict):
        return NumNeighbors(nn)
    if isinstance(nn, (list, tuple)):
        return list(nn)
    raise TypeError(f"`num_neighbors` must be NumNeighbors/dict/list/tuple; got {type(nn)}")


# ── Subgraph stat counting ────────────────────────────────────────────────────

def count_nodes_edges(batch) -> tuple[int, int]:
    """Sum nodes and edges across all types in a HeteroData batch."""
    total_nodes = sum(
        batch[nt].num_nodes
        for nt in batch.node_types
        if hasattr(batch[nt], "num_nodes") and batch[nt].num_nodes is not None
    )
    total_edges = sum(
        batch[et].num_edges
        for et in batch.edge_types
        if hasattr(batch[et], "num_edges") and batch[et].num_edges is not None
    )
    return total_nodes, total_edges


# ── Output filename logic ─────────────────────────────────────────────────────

def build_output_filename(dataset_name, task_name, num_layers, sampling_cfg_path, output_root) -> Path:
    hops_tag = f"{num_layers}hops"
    cfg = sampling_cfg_path.strip()

    if not cfg:
        suffix_tag = "random"
    else:
        ext = Path(cfg).suffix.lower()
        if ext == ".json":
            suffix_tag = "metasieve"
        elif ext == ".py":
            suffix_tag = "mpsgnn"
        else:
            raise ValueError(f"Unsupported sampling config extension: {ext}")

    filename = f"{dataset_name}_{task_name}_{hops_tag}_{suffix_tag}.txt"
    out_dir = Path(output_root) / dataset_name / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = (
        torch.device("cpu")
        if args.device == "cpu" or not torch.cuda.is_available()
        else torch.device("cuda:0")
    )

    # ── Load dataset / task ───────────────────────────────────────────────────
    print(f"Loading dataset '{args.dataset_name}' / task '{args.task_name}' ...")
    dataset = get_dataset(args.dataset_name, download=True)
    task    = get_task(args.dataset_name, args.task_name, download=True)

    train_table = task.get_table("train")
    db          = dataset.get_db()

    col_to_stype_dict = get_stype_proposal(db)
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=args.text_emb_batch_size,
    )

    data, _ = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=os.path.join("./data", f"{args.dataset_name}_glove_materialized_cache"),
    )

    # ── Build num_neighbors ───────────────────────────────────────────────────
    cfg_path = args.sampling_cfg_path.strip()
    if not cfg_path:
        num_neighbors = [args.fanout] * args.num_layers
    else:
        ext = Path(cfg_path).suffix.lower()
        if ext == ".json":
            num_neighbors = num_neighbors_from_json(cfg_path)
        elif ext == ".py":
            num_neighbors = num_neighbors_from_py(cfg_path)
        else:
            raise ValueError(f"Unsupported config extension: {ext}")

    # ── Build train loader ────────────────────────────────────────────────────
    table_input  = get_node_train_table_input(table=train_table, task=task)
    entity_table = table_input.nodes[0]

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=getattr(table_input, "time", None),
        transform=getattr(table_input, "transform", None),
        batch_size=args.batch_size,
        temporal_strategy="uniform",
        shuffle=False,          # deterministic for counting
        num_workers=0,
        persistent_workers=False,
    )

    # ── Iterate and collect stats ─────────────────────────────────────────────
    node_counts = []
    edge_counts = []

    print(f"Iterating over {len(train_loader)} training batches ...")
    for batch in tqdm(train_loader, desc="Counting subgraph stats"):
        n, e = count_nodes_edges(batch)
        node_counts.append(n)
        edge_counts.append(e)

    avg_nodes = float(np.mean(node_counts))
    avg_edges = float(np.mean(edge_counts))
    std_nodes = float(np.std(node_counts))
    std_edges = float(np.std(edge_counts))

    print(f"\nResults over {len(node_counts)} batches:")
    print(f"  Avg nodes : {avg_nodes:.2f}  (std {std_nodes:.2f})")
    print(f"  Avg edges : {avg_edges:.2f}  (std {std_edges:.2f})")

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = build_output_filename(
        args.dataset_name,
        args.task_name,
        args.num_layers,
        args.sampling_cfg_path,
        args.output_root,
    )

    with open(out_path, "w") as f:
        f.write(f"Dataset      : {args.dataset_name}\n")
        f.write(f"Task         : {args.task_name}\n")
        f.write(f"Num layers   : {args.num_layers}\n")
        f.write(f"Batch size   : {args.batch_size}\n")
        f.write(f"Fanout       : {args.fanout}\n")
        f.write(f"Sampling cfg : {args.sampling_cfg_path or '(none — random)'}\n")
        f.write(f"Num batches  : {len(node_counts)}\n")
        f.write(f"\nAvg nodes per subgraph : {avg_nodes:.4f}\n")
        f.write(f"Std nodes per subgraph : {std_nodes:.4f}\n")
        f.write(f"Avg edges per subgraph : {avg_edges:.4f}\n")
        f.write(f"Std edges per subgraph : {std_edges:.4f}\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()