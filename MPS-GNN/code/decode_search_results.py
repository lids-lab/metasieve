import os
import re
import ast
import torch
import argparse
from pathlib import Path

from mps_bridge import relbench_hetero_to_homo_for_mps
from embeddings import GloveTextEmbedding

from relbench.datasets import get_dataset
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph


def _decode_rel(rel_id: int, names):
    if isinstance(names, (list, tuple)) and 0 <= rel_id < len(names):
        return names[rel_id]
    if isinstance(names, dict):
        return names.get(rel_id)
    return None


def decode_all_paths_from_file(names, in_path: Path, out_path: Path):
    pat = re.compile(
        r"\(path=(\[[^\]]*\])\s*,\s*hop=(\d+),\s*sort_by=([^)]+)\)\s*rank=(\d+)\s+(\w+)=([0-9.]+)"
    )

    lines_out = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue

            path_str, hop_str, sort_by, rank_str, metric_name, metric_str = m.groups()
            path = ast.literal_eval(path_str)
            hop = int(hop_str)
            rank = int(rank_str)
            metric_val = float(metric_str)

            lines_out.append(
                f"rank={rank}  {metric_name}={metric_val:.6f}  hop={hop}  sort_by={sort_by.strip()}  path={path}\n"
            )

            lines_out.append("  decoded (logged order):\n")
            for r in path:
                rel = _decode_rel(int(r), names)
                lines_out.append(f"    {int(r)} {rel}\n")

            fwd = list(reversed(path))
            lines_out.append(f"  decoded (forward order = reverse(path)): {fwd}\n")
            for r in fwd:
                rel = _decode_rel(int(r), names)
                lines_out.append(f"    {int(r)} {rel}\n")

            lines_out.append("\n")

    with open(out_path, "w", encoding="utf-8") as g:
        g.writelines(lines_out)

    print(f"Wrote {len(lines_out)} lines to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode MPS-GNN search results")
    parser.add_argument("--dataset",             type=str, required=True,  help="Relbench dataset name (e.g. rel-stack)")
    parser.add_argument("--task",                type=str, required=True,  help="Task name (e.g. user-badge)")
    parser.add_argument("--channels",            type=int, default=256,    help="Hidden embedding dimension (must match run_search.py)")
    parser.add_argument("--text_emb_batch_size", type=int, default=256)
    parser.add_argument("--device",              type=str, default="cuda:0")
    parser.add_argument("--data_dir",            type=str, default="./data")
    args = parser.parse_args()

    seed_everything(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    txt_path     = Path("search_results") / args.dataset / f"mps_search_{args.task}.txt"
    decoded_path = Path("search_results") / args.dataset / f"mps_search_{args.task}_decoded.log"

    dataset = get_dataset(args.dataset, download=True)
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

    names = data_homo.edge_type_names

    decode_all_paths_from_file(names, txt_path, decoded_path)
