import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]



def hetero_adjacency(data) -> Dict[str, List[EdgeType]]:
    """
    Build a simple adjacency from the heterogeneous graph:
      node_type -> list of outgoing (src, rel, dst) triples.
    """
    node_types, edge_types = data.metadata()
    adj: Dict[str, List[EdgeType]] = defaultdict(list)
    for (src, rel, dst) in edge_types:
        adj[src].append((src, rel, dst))
    return adj

def base_rel(rel_name: str) -> str:
    """
    Normalize relation names by removing the 'rev_' prefix so that:
      - 'f2p_OwnerUserId'  and
      - 'rev_f2p_OwnerUserId'
    map to the same *base* relation family
    """
    return rel_name[4:] if rel_name.startswith("rev_") else rel_name
           

def is_trivial_backtrack(prev_edge: EdgeType, next_edge: EdgeType) -> bool:
    (src1, rel1, dst1) = prev_edge
    (src2, rel2, dst2) = next_edge

    # must be an inverse pair in schema
    inverse_pair = (
        (dst1 == src2) and (src1 == dst2)
        and (base_rel(rel1) == base_rel(rel2))
        and (rel1.startswith("rev_") != rel2.startswith("rev_"))
    )
    if not inverse_pair:
        return False

    # TRIVIAL only when we go rev_* then immediately go forward
    return rel1.startswith("rev_") and (not rel2.startswith("rev_"))

def is_immediate_inverse(prev_edge: EdgeType, next_edge: EdgeType) -> bool:
    """
    Implements the non-backtracking rule:
    Do not immediately traverse the exact inverse of the previous step.
    Concretely, (src1, rel1, dst1) followed by (src2, rel2, dst2) is an
    immediate inverse if:
      - we go back to the prior node type (dst1 == src2 and src1 == dst2), and
      - both edges are the same base relation family but in opposite directions.
    """
    (src1, rel1, dst1) = prev_edge
    (src2, rel2, dst2) = next_edge
    return (dst1 == src2) and (src1 == dst2) \
           and (base_rel(rel1) == base_rel(rel2)) \
           and (rel1.startswith("rev_") != rel2.startswith("rev_"))
           

def enumerate_metapaths_from(
    data,
    start_node_type: str = "users",
    max_hops: int = 4,
    avoid_immediate_backtrack: bool = True,
    dedupe_by_base_rel: bool = True,
) -> Dict[int, List[MetaPath]]:
    """
    Enumerate all meta-paths starting from start_node_type

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
    start_node_type : str
    max_hops : int
    avoid_immediate_backtrack : bool
    dedupe_by_base_rel : bool
        If True, treat 'f2p_X' and 'rev_f2p_X' as the same family when deduplicating
        paths (we still keep direction for traversal; this only affects uniqueness).
    """
    adj = hetero_adjacency(data)  # node_type -> outgoing edge triples
    
    # Internal DFS state:
    path: MetaPath = []           # current path (list of edge triples)
    # Accumulator: per hop length, keep a set of canonical keys for dedup
    seen_by_hop: Dict[int, Set[Tuple]] = {h: set() for h in range(1, max_hops + 1)}
    paths_by_hop: Dict[int, List[MetaPath]] = {h: [] for h in range(1, max_hops + 1)}

    def path_key(p: MetaPath) -> Tuple:
        """
        - If dedupe_by_base_rel=True, strip 'rev_' from each relation name.
        - Include src/dst node types to preserve the typed schema sequence.
        """
        if dedupe_by_base_rel:
            return tuple((s, base_rel(r), d) for (s, r, d) in p)
        else:
            return tuple(p)

    def dfs(curr_type: str, depth: int):
        """
        Depth-first traversal over the typed schema. At each step, enumerate all
        outgoing typed edges (src=curr_type) and extend the path. Stop at max_hops.
        """
        if depth == max_hops:
            return
        for step in adj.get(curr_type, []):
            # Enforce non-backtracking if requested:
            if avoid_immediate_backtrack and path:
                if is_trivial_backtrack(path[-1], step):
                    continue
                # if is_immediate_inverse(path[-1], step):
                #     continue

            # Take the step:
            path.append(step)

            # Record this path under its hop length if not seen already:
            key = path_key(path)
            hop = len(path)
            if key not in seen_by_hop[hop]:
                seen_by_hop[hop].add(key)
                # Store the *exact* edge sequence (with original 'rev_' kept for clarity)
                paths_by_hop[hop].append(path.copy())

            # Recurse from the new dst type:
            _, _, nxt = step
            dfs(nxt, depth + 1)

            # Backtrack in DFS:
            path.pop()

    # Kick off DFS from the requested start node type:
    dfs(start_node_type, 0)

    # Sort for stable, readable output: primarily by node-type sequence, secondarily by rel names
    def node_seq(mp: MetaPath) -> List[str]:
        return [mp[0][0]] + [e[2] for e in mp]
    def rel_seq(mp: MetaPath) -> List[str]:
        return [e[1] for e in mp]

    for h in paths_by_hop:
        paths_by_hop[h].sort(key=lambda mp: (node_seq(mp), rel_seq(mp)))

    return paths_by_hop

def pretty_print_metapaths(mp_by_hop: Dict[int, List[MetaPath]]):
    """
    Nicely print meta-paths grouped by hop length.
    For each path we show: node-type chain and the relation-name chain.
    """
    def node_path(mp: MetaPath) -> str:
        return " -> ".join([mp[0][0]] + [e[2] for e in mp])
    def rel_path(mp: MetaPath) -> str:
        return ", ".join([e[1] for e in mp])

    for h in sorted(mp_by_hop):
        print(f"\n# Hop {h} (count={len(mp_by_hop[h])})")
        for mp in mp_by_hop[h]:
            print(f"  {node_path(mp)}   [{rel_path(mp)}]")


@dataclass
class SchemaVocabs:
    node2id: Dict[str, int]      # node type -> idx
    table2id: Dict[str, int]     # same as node2id (alias)
    node_dim: int


def build_schema_vocabs(data) -> SchemaVocabs:
    """
    Build a minimal, dataset-agnostic schema vocab from the graph metadata.
    No attribute schema, no relation families, no direction encodings.
    """
    node_types, edge_types = data.metadata()
    node2id = {nt: i for i, nt in enumerate(sorted(node_types))}
    table2id = node2id.copy()
    return SchemaVocabs(
        node2id=node2id,
        table2id=table2id,
        node_dim=len(node2id),
    )


@dataclass
class Candidate:
    kind: str          # always "table" for this simplified version
    edge: EdgeType     # (src, rel, dst)
    table: str         # child table (= edge[2])
    # attribute is intentionally removed / unused


def generate_candidates_for_prefix(
    prefix: MetaPath,
    data,
    voc: Optional[SchemaVocabs] = None,
    allowed_tables: Optional[set] = None,
    avoid_immediate_backtrack: bool = True,
) -> List[Candidate]:
    """
    Generate table-level candidates for the next hop from the end node type of `prefix`,
    optionally filtering out trivial immediate backtracks (schema-level).
    """
    node_types, edge_types = data.metadata()

    # adjacency list by source node type (schema-level)
    adj = defaultdict(list)
    for (src, rel, dst) in edge_types:
        adj[src].append((src, rel, dst))

    end_nt = prefix[-1][2]
    next_steps = adj.get(end_nt, [])

    allowed = set(node_types) if allowed_tables is None else allowed_tables

    out: List[Candidate] = []
    prev_edge = prefix[-1] if prefix else None

    for step in next_steps:
        (src, rel, dst) = step

        if dst not in allowed:
            continue
        if voc is not None and dst not in voc.table2id:
            continue

        if avoid_immediate_backtrack and prev_edge is not None:
            if is_trivial_backtrack(prev_edge, step):
                continue
            # if is_immediate_inverse(prev_edge, step):
            #     continue

        out.append(Candidate(kind="table", edge=step, table=dst))

    return out

