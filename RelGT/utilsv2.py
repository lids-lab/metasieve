import gc
import os
import math
import heapq
import random
from itertools import accumulate
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import HeteroData

from relbench.base import EntityTask
from relbench.modeling.graph import get_node_train_table_input

Key = Tuple[str, int]
SECS_PER_DAY = 24.0 * 60.0 * 60.0


class GloveTextEmbedding:
    def __init__(self, device: str):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


def build_adjacency_hetero(hetero_data: HeteroData, undirected: bool = True):
    adjacency = {nt: [set() for _ in range(hetero_data[nt].num_nodes)] for nt in hetero_data.node_types}
    for et in hetero_data.edge_types:
        src_t, _, dst_t = et
        if "edge_index" not in hetero_data[et]:
            continue
        edge_index = hetero_data[et].edge_index
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adjacency[src_t][s].add((dst_t, d))
            if undirected:
                adjacency[dst_t][d].add((src_t, s))
    return adjacency


class RelGTTokens(TorchDataset):
    def __init__(
        self,
        data: HeteroData,
        task: EntityTask,
        K: int,
        split: str = "train",
        undirected: bool = True,
        num_workers: int = None,
        precompute: bool = True,
        precomputed_dir: str = None,
        train_stage: str = "finetune",
    ):
        super().__init__()
        self.data = data
        self.task = task
        self.split = split
        self.K = K
        self.undirected = undirected
        self.num_workers = num_workers
        self.precompute = precompute
        self.precomputed_dir = precomputed_dir
        self.train_stage = train_stage

        self.table = self.task.get_table(split=self.split)
        self.table_input = get_node_train_table_input(self.table, self.task)
        self.node_type, self.node_idxs = self.table_input.nodes
        self.target = self.table_input.target if self.table_input.target is not None else None
        self.time = getattr(self.table_input, "time", None)
        self.transform = getattr(self.table_input, "transform", None)

        self.node_types = self.data.node_types
        self.node_type_to_index = {nt: idx for idx, nt in enumerate(self.node_types)}
        self.index_to_node_type = {idx: nt for idx, nt in enumerate(self.node_types)}
        self.max_neighbor_hop = 3

        self._create_global_mappings()
        self.precomputed_path = self._construct_precomputed_path()

        if self.precompute:
            if not os.path.exists(self.precomputed_path):
                raise FileNotFoundError(f"Missing precomputed file: {self.precomputed_path}")

    def _create_global_mappings(self):
        self.type_local_to_global = {}
        self.global_to_type_local = {}
        global_index = 0
        for type_idx, node_type in self.index_to_node_type.items():
            num_nodes = int(self.data[node_type].num_nodes)
            for local_idx in range(num_nodes):
                self.type_local_to_global[(type_idx, local_idx)] = global_index
                self.global_to_type_local[global_index] = (type_idx, local_idx)
                global_index += 1

    def get_global_index(self, type_idxs: List[int], local_idxs: List[int]) -> List[int]:
        return [self.type_local_to_global[(t_i, l_i)] for t_i, l_i in zip(type_idxs, local_idxs)]

    def _construct_precomputed_path(self) -> str:
        if not self.precomputed_dir:
            raise ValueError("must provide 'precomputed_dir'")
        path = os.path.join(self.precomputed_dir, str(self.K), f"{self.split}.h5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def __len__(self):
        return len(self.node_idxs)

    def __getitem__(self, idx: int):
        with h5py.File(self.precomputed_path, "r") as hf:
            sample = {
                "types": torch.from_numpy(hf["types"][idx]).long(),
                "indices": torch.from_numpy(hf["indices"][idx]).long(),
                "hops": torch.from_numpy(hf["hops"][idx]).long(),
                "times": torch.from_numpy(hf["times"][idx]),
            }
            offsets = hf["edges_offsets"]
            edges_dset = hf["edges"]
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            if start == end:
                eidx = torch.zeros((2, 0), dtype=torch.long)
            else:
                eidx = torch.from_numpy(edges_dset[:, start:end]).long()
            sample["edge_index"] = eidx

        label = self.target[idx] if self.target is not None else None
        sample["first_type"] = int(sample["types"][0].item())
        sample["first_index"] = int(sample["indices"][0].item())

        sample["tfs"] = [
            self.data[self.index_to_node_type[t.item()]].tf[i.item()]
            for t, i in zip(sample["types"], sample["indices"])
        ]
        sample["global_idx"] = idx
        return sample, label

    def collate(self, batch: List[Tuple[dict, Optional[torch.Tensor]]]):
        samples, labels = zip(*batch)

        neighbor_types = torch.stack([s["types"] for s in samples], dim=0)
        neighbor_indices = torch.stack([s["indices"] for s in samples], dim=0)
        neighbor_hops = torch.stack([s["hops"] for s in samples], dim=0)
        neighbor_times = torch.stack([s["times"] for s in samples], dim=0)

        out = {
            "neighbor_types": neighbor_types,
            "neighbor_indices": neighbor_indices,
            "neighbor_hops": neighbor_hops,
            "neighbor_times": neighbor_times,
        }

        out["labels"] = torch.stack(labels, dim=0) if self.target is not None else None

        first_types = [s["first_type"] for s in samples]
        first_indices = [s["first_index"] for s in samples]
        out["node_indices"] = torch.tensor(self.get_global_index(first_types, first_indices), dtype=torch.long)

        B, K = neighbor_types.shape
        grouped_tfs = {}
        grouped_positions = {}
        for t_id in range(len(self.node_types)):
            mask = neighbor_types == t_id
            if not bool(mask.any()):
                continue
            local_idxs = neighbor_indices[mask]
            type_str = self.index_to_node_type[t_id]

            pos_2d = torch.nonzero(mask, as_tuple=False)
            grouped_positions[t_id] = [int(bb) * K + int(kk) for bb, kk in pos_2d.tolist()]
            grouped_tfs[t_id] = self.data[type_str].tf[local_idxs]

        out["grouped_tfs"] = grouped_tfs
        out["grouped_indices"] = grouped_positions
        out["flat_batch_idx"] = torch.arange(B).unsqueeze(1).expand(B, K).reshape(-1).tolist()
        out["flat_nbr_idx"] = torch.arange(K).repeat(B).tolist()
        out["global_idx"] = torch.tensor([s["global_idx"] for s in samples], dtype=torch.long)

        batched_edges = []
        batch_vec = []
        node_offset = 0
        for i, s in enumerate(samples):
            eidx = s["edge_index"]
            shifted = eidx + node_offset
            batched_edges.append(shifted)
            batch_vec.append(torch.full((K,), i, dtype=torch.long))
            node_offset += K

        out["edge_index"] = torch.cat(batched_edges, dim=1) if batched_edges else torch.zeros((2, 0), dtype=torch.long)
        out["batch"] = torch.cat(batch_vec, dim=0) if batch_vec else torch.zeros((0,), dtype=torch.long)
        return out
    
def _choices_from_hops(hop_buckets: List[List[Key]], k: int, max_hop: int) -> List[Key]:
    """Uniform over the union, WITH replacement, without flattening."""
    if k <= 0:
        return []
    sizes = [len(hop_buckets[d]) for d in range(1, max_hop + 1)]
    total = sum(sizes)
    if total == 0:
        return []
    cum = list(accumulate(sizes))  # using cum_weights saves work :contentReference[oaicite:1]{index=1}
    hops = random.choices(range(1, max_hop + 1), cum_weights=cum, k=k)
    return [random.choice(hop_buckets[d]) for d in hops]


def _sample_remaining_without_replacement(
    hop_buckets: List[List[Key]],
    used: set,
    k: int,
    max_hop: int,
) -> List[Key]:
    if k <= 0:
        return []
    heap: List[Tuple[float, Key]] = []  # max-heap via negative key
    for d in range(1, max_hop + 1):
        for x in hop_buckets[d]:
            if x in used:
                continue
            r = random.random()
            if len(heap) < k:
                heapq.heappush(heap, (-r, x))
            else:
                if r < -heap[0][0]:
                    heapq.heapreplace(heap, (-r, x))
    out = [x for _, x in heap]
    random.shuffle(out)
    return out


def pick_proportional_from_hops_fast(hop_buckets: List[List[Key]], budget: int, max_hop: int) -> List[Key]:
    """
    Same intent as your pick_proportional_from_hops, but:
      - no flatten 'rest'
      - uses random.sample instead of torch.randperm
      - returns length == budget if total reachable > 0, else [].
    """
    if budget <= 0:
        return []

    sizes = [len(hop_buckets[d]) for d in range(1, max_hop + 1)]
    total = sum(sizes)
    if total == 0:
        return []

    raw = [budget * (s / total) for s in sizes]
    base = [int(x) for x in raw]
    frac = [raw[i] - base[i] for i in range(max_hop)]

    leftover = budget - sum(base)
    if leftover > 0:
        for i in sorted(range(max_hop), key=lambda i: frac[i], reverse=True)[:leftover]:
            base[i] += 1

    picked: List[Key] = []
    used = set()

    # Per-hop quota sampling (without replacement inside each bucket) :contentReference[oaicite:2]{index=2}
    for d in range(1, max_hop + 1):
        q = base[d - 1]
        bucket = hop_buckets[d]
        if q <= 0 or not bucket:
            continue
        if len(bucket) >= q:
            part = random.sample(bucket, q)
        else:
            part = list(bucket)  # take all unique; we'll fill later
        picked.extend(part)
        used.update(part)

    # Fill missing WITHOUT replacement from remaining union
    if len(picked) < budget:
        need = budget - len(picked)
        extra = _sample_remaining_without_replacement(hop_buckets, used, need, max_hop)
        picked.extend(extra)
        used.update(extra)

    # If still short (not enough unique total), fill WITH replacement from union :contentReference[oaicite:3]{index=3}
    if len(picked) < budget:
        picked.extend(_choices_from_hops(hop_buckets, budget - len(picked), max_hop))

    return picked[:budget]
    
    
def pick_proportional_from_hops(hop_buckets, budget, max_hop):
    """
    hop_buckets[d] is a list of node keys at hop d (d=1..max_hop).
    budget is how many non-seed tokens to pick (K-1).
    Returns a flat list of picked node keys length <= budget.
    """
    counts = [len(hop_buckets[d]) for d in range(1, max_hop + 1)]
    total = sum(counts)
    if total == 0 or budget <= 0:
        return []

    # Real-valued ideal quotas
    raw = [budget * (c / total) for c in counts]

    # Base integer quotas
    base = [int(math.floor(x)) for x in raw]
    rem = [x - b for x, b in zip(raw, base)]

    # Distribute leftovers by largest remainder
    leftover = budget - sum(base)
    order = sorted(range(max_hop), key=lambda i: rem[i], reverse=True)
    quotas = base[:]
    for i in order:
        if leftover <= 0:
            break
        quotas[i] += 1
        leftover -= 1

    picked = []
    used = set()

    # Sample within each hop bucket
    for d in range(1, max_hop + 1):
        q = quotas[d - 1]
        bucket = hop_buckets[d]
        if q <= 0 or not bucket:
            continue

        perm = torch.randperm(len(bucket)).tolist()
        for j in perm:
            key = bucket[j]
            if key in used:
                continue
            picked.append(key)
            used.add(key)
            if len(picked) >= sum(quotas[:d]):  # satisfied this hop's quota cumulatively
                break

    # If some hops were too small, we may still be short. Fill from remaining hop nodes uniformly.
    if len(picked) < budget:
        rest = []
        for d in range(1, max_hop + 1):
            rest.extend([k for k in hop_buckets[d] if k not in used])
        if rest:
            perm = torch.randperm(len(rest)).tolist()
            for j in perm:
                picked.append(rest[j])
                if len(picked) == budget:
                    break

    return picked




def relgt_from_neighborloader_batch(
    nl_batch: HeteroData,
    hetero_data: HeteroData,
    seed_type: str,
    K: int,
    seed_ids_local: torch.Tensor,
    seed_times_local: Optional[torch.Tensor],
    row_ids_local: torch.Tensor,
    targets_local: Optional[torch.Tensor],
    max_hop: int,
    fallback_hop_id: int,
    time_attr: str = "time",
    undirected: bool = True,
) -> Dict:
    node_types = list(hetero_data.node_types)
    type_to_idx = {nt: i for i, nt in enumerate(node_types)}
    num_real_types = len(node_types)
    mask_type_id = num_real_types

    seed_store = nl_batch[seed_type]
    B = int(getattr(seed_store, "batch_size", 0))
    if B <= 0:
        raise RuntimeError("NeighborLoader batch_size is 0; cannot build tokens.")

    input_pos = None
    if hasattr(seed_store, "input_id") and isinstance(seed_store.input_id, torch.Tensor):
        cand = seed_store.input_id.to(torch.long).cpu()
        if cand.numel() == B and cand.min().item() >= 0 and cand.max().item() < int(seed_ids_local.numel()):
            seed_ids_check = seed_store.n_id[:B].to(torch.long).cpu()
            if torch.equal(seed_ids_local[cand].to(torch.long).cpu(), seed_ids_check):
                input_pos = cand

    seed_ids_in_batch = seed_store.n_id[:B].to(torch.long).cpu()

    if input_pos is not None:
        seed_ids = seed_ids_local[input_pos].to(torch.long).cpu()
        global_idx = row_ids_local[input_pos].to(torch.long).cpu()
        labels = None
        if targets_local is not None:
            labels = targets_local[input_pos].cpu()
            if labels.dtype not in (torch.float16, torch.float32, torch.float64):
                labels = labels.to(torch.float32)
        seed_times = seed_times_local[input_pos].cpu() if seed_times_local is not None else None
    else:
        seed_ids = seed_ids_in_batch
        global_idx = None
        labels = None
        seed_times = None

    offsets: Dict[str, int] = {}
    running = 0
    for nt in node_types:
        offsets[nt] = running
        running += int(hetero_data[nt].num_nodes)

    node_indices = offsets[seed_type] + seed_ids.to(torch.long)

    time_tensor_by_type: Dict[str, Optional[torch.Tensor]] = {}
    for nt in node_types:
        if time_attr in hetero_data[nt] and isinstance(hetero_data[nt][time_attr], torch.Tensor):
            time_tensor_by_type[nt] = hetero_data[nt][time_attr]
        else:
            time_tensor_by_type[nt] = None

    adj_by_b: List[Dict[Tuple[str, int], set]] = [dict() for _ in range(B)]

    for et in nl_batch.edge_types:
        if "edge_index" not in nl_batch[et]:
            continue
        src_t, _, dst_t = et
        eidx = nl_batch[et].edge_index
        if not isinstance(eidx, torch.Tensor) or eidx.numel() == 0:
            continue
        if not hasattr(nl_batch[src_t], "batch") or not hasattr(nl_batch[dst_t], "batch"):
            continue
        if not hasattr(nl_batch[src_t], "n_id") or not hasattr(nl_batch[dst_t], "n_id"):
            continue

        src_batch = nl_batch[src_t].batch.to(torch.long).cpu()
        dst_batch = nl_batch[dst_t].batch.to(torch.long).cpu()
        src_nid = nl_batch[src_t].n_id.to(torch.long).cpu()
        dst_nid = nl_batch[dst_t].n_id.to(torch.long).cpu()

        eidx_cpu = eidx.to(torch.long).cpu()
        src_pos = eidx_cpu[0]
        dst_pos = eidx_cpu[1]
        for j in range(int(src_pos.numel())):
            sp = int(src_pos[j].item())
            dp = int(dst_pos[j].item())
            if sp < 0 or dp < 0 or sp >= int(src_batch.numel()) or dp >= int(dst_batch.numel()):
                continue
            b_s = int(src_batch[sp].item())
            b_d = int(dst_batch[dp].item())
            if b_s != b_d or b_s < 0 or b_s >= B:
                continue
            u = (src_t, int(src_nid[sp].item()))
            v = (dst_t, int(dst_nid[dp].item()))
            dct = adj_by_b[b_s]
            if u not in dct:
                dct[u] = set()
            dct[u].add(v)
            if undirected:
                if v not in dct:
                    dct[v] = set()
                dct[v].add(u)

    neighbor_types = torch.full((B, K), mask_type_id, dtype=torch.long)
    neighbor_indices = torch.zeros((B, K), dtype=torch.long)
    neighbor_hops = torch.full((B, K), fallback_hop_id, dtype=torch.long)
    # neighbor_times = torch.full((B, K), -1.0, dtype=torch.float32)
    neighbor_times = torch.zeros((B, K), dtype=torch.float32)

    tokens_by_b: List[List[Tuple[str, int]]] = []

    for b in range(B):
        sid = int(seed_ids[b].item())
        seed_key = (seed_type, sid)

        st = None
        if seed_times is not None:
            st = int(seed_times[b].item())
        else:
            tt = time_tensor_by_type.get(seed_type, None)
            if tt is not None and 0 <= sid < int(tt.numel()):
                st = int(tt[sid].item())

        dist = {seed_key: 0}
        dq = deque([seed_key])
        adj = adj_by_b[b]
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du >= max_hop:
                continue
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = du + 1
                    dq.append(v)

        hop_buckets: List[List[Tuple[str, int]]] = [[] for _ in range(max_hop + 1)]
        for key, d in dist.items():
            if d == 0:
                continue
            if 1 <= d <= max_hop:
                hop_buckets[d].append(key)

        budget = K - 1

        total_reachable = 0
        for d in range(1, max_hop + 1):
            total_reachable += len(hop_buckets[d])

        if budget <= 0:
            picked = []
        else:
            if total_reachable >= budget:
                picked = pick_proportional_from_hops_fast(hop_buckets, budget, max_hop)
            elif total_reachable > 0:
                picked = _choices_from_hops(hop_buckets, budget, max_hop)  # uniform union, with replacement
            else:
                picked = []

        tokens = [seed_key] + picked
        while len(tokens) < K:
            tokens.append(("", -1))

        tokens_by_b.append(tokens)

        for k, (nt, nid) in enumerate(tokens):
            if k == 0:
                neighbor_types[b, k] = type_to_idx[seed_type]
                neighbor_indices[b, k] = sid
                neighbor_hops[b, k] = 0
                neighbor_times[b, k] = 0.0 if st is not None else -1.0
                continue

            if nt == "" or nid < 0:
                neighbor_types[b, k] = mask_type_id
                neighbor_indices[b, k] = 0
                neighbor_hops[b, k] = fallback_hop_id
                neighbor_times[b, k] = -1.0
                continue

            neighbor_types[b, k] = type_to_idx[nt]
            neighbor_indices[b, k] = int(nid)

            d = dist.get((nt, int(nid)), None)
            neighbor_hops[b, k] = int(d) if (d is not None and d <= max_hop) else fallback_hop_id

            # nt_time = time_tensor_by_type.get(nt, None)
            # if st is not None and nt_time is not None and 0 <= int(nid) < int(nt_time.numel()):
            #     neighbor_times[b, k] = float(st - int(nt_time[int(nid)].item()))
            # else:
            #     neighbor_times[b, k] = -1.0
            nt_time = time_tensor_by_type.get(nt, None)
            if st is not None and nt_time is not None and 0 <= int(nid) < int(nt_time.numel()):
                dt_sec = float(st - int(nt_time[int(nid)].item()))
                if dt_sec < 0:
                    dt_sec = 0.0  # optional safety if NL ever includes future nodes
                neighbor_times[b, k] = dt_sec / SECS_PER_DAY
            else:
                neighbor_times[b, k] = 0.0

    edge_src: List[int] = []
    edge_dst: List[int] = []
    for b in range(B):
        base = b * K
        tokens = tokens_by_b[b]
        key_to_pos: Dict[Tuple[str, int], int] = {}
        for k, (nt, nid) in enumerate(tokens):
            if nt != "" and nid >= 0:
                key_to_pos[(nt, int(nid))] = k

        adj = adj_by_b[b]
        for u_key, u_pos in key_to_pos.items():
            for v_key in adj.get(u_key, ()):
                v_pos = key_to_pos.get(v_key, None)
                if v_pos is None:
                    continue
                edge_src.append(base + u_pos)
                edge_dst.append(base + v_pos)

    edge_index = torch.zeros((2, 0), dtype=torch.long) if len(edge_src) == 0 else torch.tensor([edge_src, edge_dst], dtype=torch.long)
    batch_vec = torch.arange(B, dtype=torch.long).repeat_interleave(K)
    flat_batch_idx = torch.arange(B).unsqueeze(1).expand(B, K).reshape(-1).tolist()
    flat_nbr_idx = torch.arange(K).repeat(B).tolist()

    grouped_tfs: Dict[int, object] = {}
    grouped_positions: Dict[int, List[int]] = {}
    for t_id in range(num_real_types):
        mask = neighbor_types == t_id
        if not bool(mask.any()):
            continue
        local_idxs = neighbor_indices[mask].to(torch.long)
        type_str = node_types[t_id]
        pos_2d = torch.nonzero(mask, as_tuple=False)
        grouped_positions[t_id] = [int(bb) * K + int(kk) for bb, kk in pos_2d.tolist()]
        grouped_tfs[t_id] = hetero_data[type_str].tf[local_idxs]

    out = {
        "neighbor_types": neighbor_types,
        "neighbor_hops": neighbor_hops,
        "neighbor_times": neighbor_times,
        "node_indices": node_indices.to(torch.long),
        "grouped_tfs": grouped_tfs,
        "grouped_indices": grouped_positions,
        "flat_batch_idx": flat_batch_idx,
        "flat_nbr_idx": flat_nbr_idx,
        "edge_index": edge_index,
        "batch": batch_vec,
        "global_idx": global_idx if global_idx is not None else torch.full((B,), -1, dtype=torch.long),
    }

    if targets_local is not None and labels is not None:
        out["labels"] = labels
    elif targets_local is None:
        out["labels"] = None
    else:
        raise RuntimeError(
            "targets_local is provided but labels could not be mapped. "
            "Ensure NeighborLoader exposes per-seed input positions via input_id."
        )

    return out

