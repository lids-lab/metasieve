# #!/usr/bin/env python3
# import argparse
# import re
# from collections import defaultdict
# from typing import Dict, List, Tuple

# EdgeType = Tuple[str, str, str]

# RANK_RE = re.compile(r"^rank=(\d+)\b")
# FWD_HDR_RE = re.compile(r"^\s*decoded\s+\(forward order = reverse\(path\)\):")
# EDGE_LINE_RE = re.compile(r"^\s*\d+\s+\('([^']+)',\s*'([^']+)',\s*'([^']+)'\)\s*$")


# def reverse_rel(rel: str) -> str:
#     return rel[4:] if rel.startswith("rev_") else "rev_" + rel


# def reverse_edge(et: EdgeType) -> EdgeType:
#     src, rel, dst = et
#     return (dst, reverse_rel(rel), src)


# def parse_topk_forward_edges(log_path: str, k: int) -> Dict[int, List[EdgeType]]:
#     """
#     Returns: rank -> list of edge tuples in *forward order* (seed -> outward).
#     """
#     with open(log_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     rank_to_edges: Dict[int, List[EdgeType]] = {}
#     i = 0
#     while i < len(lines):
#         m = RANK_RE.match(lines[i].strip())
#         if not m:
#             i += 1
#             continue

#         rank = int(m.group(1))
#         if rank > k:
#             break

#         # Find the "decoded (forward order ...)" header for this rank block
#         i += 1
#         while i < len(lines) and not FWD_HDR_RE.match(lines[i]):
#             # stop early if next rank starts (malformed block)
#             if RANK_RE.match(lines[i].strip()):
#                 break
#             i += 1

#         # If header not found, skip this rank
#         if i >= len(lines) or not FWD_HDR_RE.match(lines[i]):
#             continue

#         # Read subsequent edge lines
#         i += 1
#         edges: List[EdgeType] = []
#         while i < len(lines):
#             line = lines[i].rstrip("\n")
#             m2 = EDGE_LINE_RE.match(line)
#             if not m2:
#                 break
#             edges.append((m2.group(1), m2.group(2), m2.group(3)))
#             i += 1

#         rank_to_edges[rank] = edges

#     return rank_to_edges


# def merge_num_neighbors(
#     rank_to_forward_edges: Dict[int, List[EdgeType]],
#     fanout: int,
#     num_layers: int,
# ) -> Dict[EdgeType, List[int]]:
#     """
#     Builds dict: reverse(edge_type) -> [fanout_per_hop]
#     Default is [0]*num_layers, and we take max() when merging overlaps.
#     """
#     out: Dict[EdgeType, List[int]] = defaultdict(lambda: [0] * num_layers)

#     for rank in sorted(rank_to_forward_edges.keys()):
#         forward_edges = rank_to_forward_edges[rank]
#         for hop_idx, et in enumerate(forward_edges):
#             if hop_idx >= num_layers:
#                 # ignore hops beyond config length (or you can extend here if you want)
#                 break
#             sampled_edge = reverse_edge(et)  # your rule: sample reverse(edge) at that hop
#             out[sampled_edge][hop_idx] = max(out[sampled_edge][hop_idx], fanout)

#     return dict(out)


# def format_config(num_neighbors_dict: Dict[EdgeType, List[int]]) -> str:
#     def fmt_et(et: EdgeType) -> str:
#         return f"({et[0]!r}, {et[1]!r}, {et[2]!r})"

#     lines = []
#     lines.append("from torch_geometric.sampler.base import NumNeighbors\n")
#     lines.append("num_neighbors = NumNeighbors({")
#     for et in sorted(num_neighbors_dict.keys()):
#         lines.append(f"    {fmt_et(et)}: {num_neighbors_dict[et]},")
#     lines.append("})\n")
#     return "\n".join(lines)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--log", default="./mps_search_results/rel-stack/mps_search_user-engagement_decoded.log")
#     ap.add_argument("--k", type=int, default=15)
#     ap.add_argument("--fanout", type=int, default=64)
#     ap.add_argument("--layers", type=int, default=3)  # default [0]*4 as you requested
#     ap.add_argument("--out", default="./mps_search_results/rel-stack/user_engagement_3hops.py")
#     args = ap.parse_args()

#     rank_to_edges = parse_topk_forward_edges(args.log, args.k)
#     merged = merge_num_neighbors(rank_to_edges, args.fanout, args.layers)
#     cfg = format_config(merged)

#     with open(args.out, "w", encoding="utf-8") as f:
#         f.write(cfg)

#     print(f"Wrote merged NumNeighbors config for top-{args.k} to: {args.out}")
#     print(f"Edge-types in merged config: {len(merged)}")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple

EdgeType = Tuple[str, str, str]

RANK_RE = re.compile(r"^rank=(\d+)\b")
HOP_RE = re.compile(r"\bhop=(\d+)\b")
FWD_HDR_RE = re.compile(r"^\s*decoded\s+\(forward order = reverse\(path\)\):")
EDGE_LINE_RE = re.compile(r"^\s*\d+\s+\('([^']+)',\s*'([^']+)',\s*'([^']+)'\)\s*$")


def reverse_rel(rel: str) -> str:
    return rel[4:] if rel.startswith("rev_") else "rev_" + rel


def reverse_edge(et: EdgeType) -> EdgeType:
    src, rel, dst = et
    return (dst, reverse_rel(rel), src)


def parse_forward_entries(log_path: str) -> List[dict]:
    """
    Parse the whole log.

    Returns:
        [
          {"rank": int, "hop": int, "edges": List[EdgeType]},
          ...
        ]
    """
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries: List[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        m_rank = RANK_RE.match(line)
        if not m_rank:
            i += 1
            continue

        rank = int(m_rank.group(1))

        m_hop = HOP_RE.search(line)
        if m_hop is None:
            i += 1
            continue

        hop = int(m_hop.group(1))

        # Find the forward-order header for this rank block.
        i += 1
        while i < len(lines) and not FWD_HDR_RE.match(lines[i]):
            if RANK_RE.match(lines[i].strip()):
                break
            i += 1

        # Skip malformed block if forward-order header not found.
        if i >= len(lines) or not FWD_HDR_RE.match(lines[i]):
            continue

        # Read forward-order edge lines.
        i += 1
        edges: List[EdgeType] = []
        while i < len(lines):
            m_edge = EDGE_LINE_RE.match(lines[i].rstrip("\n"))
            if not m_edge:
                break
            edges.append((m_edge.group(1), m_edge.group(2), m_edge.group(3)))
            i += 1

        entries.append({
            "rank": rank,
            "hop": hop,
            "edges": edges,
        })

    return entries


def select_entries(entries: List[dict], k: int, num_layers: int) -> List[dict]:
    """
    Filter by num_layers first, then take top-k by rank.
    """
    filtered = [e for e in entries if e["hop"] <= num_layers]
    filtered.sort(key=lambda e: e["rank"])
    return filtered[:k]


def merge_num_neighbors(
    selected_entries: List[dict],
    fanout: int,
    num_layers: int,
) -> Dict[EdgeType, List[int]]:
    """
    Builds:
        reverse(edge_type) -> [fanout_per_hop]

    Default is [0] * num_layers.
    If an edge type appears multiple times at the same hop across selected paths,
    we keep the max fanout.
    """
    out: Dict[EdgeType, List[int]] = defaultdict(lambda: [0] * num_layers)

    for entry in sorted(selected_entries, key=lambda e: e["rank"]):
        forward_edges = entry["edges"]

        for hop_idx, et in enumerate(forward_edges):
            if hop_idx >= num_layers:
                break
            sampled_edge = reverse_edge(et)
            out[sampled_edge][hop_idx] = max(out[sampled_edge][hop_idx], fanout)

    return dict(out)


def format_config(num_neighbors_dict: Dict[EdgeType, List[int]]) -> str:
    def fmt_et(et: EdgeType) -> str:
        return f"({et[0]!r}, {et[1]!r}, {et[2]!r})"

    lines = []
    lines.append("from torch_geometric.sampler.base import NumNeighbors\n")
    lines.append("num_neighbors = NumNeighbors({")
    for et in sorted(num_neighbors_dict.keys()):
        lines.append(f"    {fmt_et(et)}: {num_neighbors_dict[et]},")
    lines.append("})\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="./search_results/rel-f1/mps_search_driver-top3_decoded.log")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--fanout", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default="./search_results/rel-f1/driver_top3_2hops.py")
    args = ap.parse_args()

    # 1) parse all entries
    entries = parse_forward_entries(args.log)

    # 2) filter by --layers first, then take top --k
    selected_entries = select_entries(entries, k=args.k, num_layers=args.layers)

    # 3) merge into NumNeighbors config
    merged = merge_num_neighbors(
        selected_entries=selected_entries,
        fanout=args.fanout,
        num_layers=args.layers,
    )

    cfg = format_config(merged)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(cfg)

    print(f"Wrote config to: {args.out}")
    print(f"layers filter: hop <= {args.layers}")
    print(f"top-k after filtering: {args.k}")
    print(f"selected ranks: {[e['rank'] for e in selected_entries]}")
    print(f"edge-types in merged config: {len(merged)}")


if __name__ == "__main__":
    main()