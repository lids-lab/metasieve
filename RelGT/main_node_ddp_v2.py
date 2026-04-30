import argparse
import copy
import json
import math
import ast
import sys
import os
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import relbench.modeling.graph as rb_graph
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.sampler import NumNeighbors
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from tqdm import tqdm

import wandb
import pynvml

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from model import RelGT
from utilsv2 import GloveTextEmbedding, RelGTTokens, relgt_from_neighborloader_batch


torch.autograd.set_detect_anomaly(True)


def to_unix_time_safe(time_ser):
    arr = time_ser.to_numpy(dtype="datetime64[ns]", copy=True)
    unix_ns = arr.astype("int64", copy=False)
    return unix_ns // (10**9)


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def init_gpu_utilization(device_index: int):
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(device_index)


def get_gpu_stats(handle, device_: torch.device):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu
    mem_allocated = torch.cuda.memory_allocated(device_) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(device_) / 1024**2
    return gpu_util, mem_allocated, mem_reserved


def cast_time_tensors_inplace(hetero_data, time_attr: str = "time"):
    for nt in hetero_data.node_types:
        store = hetero_data[nt]
        if time_attr in store and isinstance(store[time_attr], torch.Tensor):
            if store[time_attr].dtype != torch.long:
                store[time_attr] = store[time_attr].to(torch.long)

    for et in hetero_data.edge_types:
        store = hetero_data[et]
        if time_attr in store and isinstance(store[time_attr], torch.Tensor):
            if store[time_attr].dtype != torch.long:
                store[time_attr] = store[time_attr].to(torch.long)


def infer_time_dtype(hetero_data, seed_type: str, time_attr: str = "time"):
    if seed_type in hetero_data.node_types:
        store = hetero_data[seed_type]
        if time_attr in store and isinstance(store[time_attr], torch.Tensor):
            return store[time_attr].dtype
    return torch.long


def ddp_padded_perm_indices(n: int, epoch: int, seed: int, rank: int, world_size: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    perm = torch.randperm(n, generator=g)
    rem = n % world_size
    if rem != 0:
        pad = perm[: (world_size - rem)]
        perm = torch.cat([perm, pad], dim=0)
    return perm[rank::world_size]


def build_epoch_train_inputs(
    seed_ids_all: torch.Tensor,
    seed_times_all: Optional[torch.Tensor],
    row_ids_all: torch.Tensor,
    targets_all: Optional[torch.Tensor],
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    idx = ddp_padded_perm_indices(len(seed_ids_all), epoch, seed, rank, world_size)
    seed_ids_local = seed_ids_all[idx].to(torch.long)
    row_ids_local = row_ids_all[idx].to(torch.long)
    targets_local = targets_all[idx] if targets_all is not None else None
    if seed_times_all is None:
        return seed_ids_local, None, row_ids_local, targets_local
    seed_times_local = seed_times_all[idx]
    return seed_ids_local, seed_times_local, row_ids_local, targets_local


def build_seed_id_to_row(seed_ids_local: torch.Tensor, row_ids_local: torch.Tensor) -> Dict[int, int]:
    return {int(n): int(r) for n, r in zip(seed_ids_local.tolist(), row_ids_local.tolist())}


def setup_experiment_logger(log_path: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# def parse_nl_num_neighbors(spec: str):
#     """
#     Backward compatible:
#       - "64,64"  -> List[int]
#       - "@file.json" -> NumNeighbors(values=..., default=...)
#       - "{...json...}" -> NumNeighbors(...)
#     JSON schema:
#       { "default": [..], "values": { "src|rel|dst": [..], ... } }
#     """
#     s = spec.strip()

#     # Old behavior: comma list
#     if not (s.startswith("@") or s.startswith("{")):
#         return parse_int_list(s)

#     # Load JSON (from file or inline)
#     if s.startswith("@"):
#         path = s[1:]
#         with open(path, "r") as f:
#             cfg = json.load(f)
#     else:
#         cfg = json.loads(s)

#     default = cfg.get("default", None)
#     values_in = cfg.get("values", {})

#     values = {}
#     for k, v in values_in.items():
#         # "posts|rev_f2p_ParentId|posts" -> ("posts","rev_f2p_ParentId","posts")
#         parts = k.split("|")
#         if len(parts) != 3:
#             raise ValueError(f"Bad edge_type key '{k}'. Expected 'src|rel|dst'.")
#         values[(parts[0], parts[1], parts[2])] = v

#     return NumNeighbors(values, default=default)

def num_neighbors_from_py(cfg_path: str | Path):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Sampling config not found: {cfg_path}")

    # NOTE: This executes the Python file. Only load trusted configs.
    module_name = f"nl_cfg_{cfg_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(cfg_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {cfg_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "num_neighbors"):
        raise AttributeError(f"{cfg_path} must define a top-level variable named `num_neighbors`")

    nn = getattr(module, "num_neighbors")

    # Accept common forms:
    if isinstance(nn, NumNeighbors):
        return nn
    if isinstance(nn, dict):
        return NumNeighbors(nn)
    if isinstance(nn, (list, tuple)):
        return list(nn)

    raise TypeError(
        f"`num_neighbors` in {cfg_path} must be a NumNeighbors, dict, list, or tuple; got {type(nn)}"
    )


def parse_nl_num_neighbors(spec: str):
    s = spec.strip()

    # Old behavior: comma list
    if not (s.startswith("@") or s.startswith("{")):
        return parse_int_list(s)

    # Load from file
    if s.startswith("@"):
        path = Path(s[1:]).expanduser()
        suffix = path.suffix.lower()

        if suffix == ".py":
            return num_neighbors_from_py(path)

        # default: JSON file
        with open(path, "r") as f:
            cfg = json.load(f)

    # Inline JSON
    else:
        cfg = json.loads(s)

    # -------- NEW SCHEMA: num_layers/base_neighbors/neighbors --------
    if "neighbors" in cfg and "base_neighbors" in cfg and "num_layers" in cfg:
        L = int(cfg["num_layers"])
        base = int(cfg["base_neighbors"])
        neighbors_in = cfg.get("neighbors", {})
        neighbors = {ast.literal_eval(k): v for k, v in neighbors_in.items()}
        return NumNeighbors(neighbors, default=[base] * L)

    # -------- OLD SCHEMA: default/values with "src|rel|dst" keys --------
    default = cfg.get("default", None)
    values_in = cfg.get("values", {})

    values = {}
    for k, v in values_in.items():
        parts = k.split("|")
        if len(parts) != 3:
            raise ValueError(f"Bad edge_type key '{k}'. Expected 'src|rel|dst'.")
        values[(parts[0], parts[1], parts[2])] = v
    return NumNeighbors(values, default=default)



## Debugging print functions
DEBUG_STEPS = 2
def _fmt_int(x):
    return f"{int(x):,}"

def _print_hop_debug(batch, *, step: int, max_hop: int, fallback_hop_id: int, show_n: int = 50):
    hops = batch["neighbor_hops"]          # [B,K]
    types = batch["neighbor_types"]        # [B,K]
    B, K = hops.shape
    budget = K - 1

    mask_type_id = int(batch.get("mask_type_id", types.max().item()))
    hops_ns = hops[:, 1:]                  # [B, budget]
    types_ns = types[:, 1:]                # [B, budget]

    fb_mask = (hops_ns == fallback_hop_id)
    pad_mask = (types_ns == mask_type_id)
    realfb_mask = fb_mask & (~pad_mask)

    pad_per_seed = pad_mask.sum(dim=1)
    realfb_per_seed = realfb_mask.sum(dim=1)

    dbg_bucket = batch.get("debug_bucket_counts", None)   # [B, max_hop+1]
    dbg_total = batch.get("debug_reachable_total", None)  # [B]

    print("\n" + "=" * 90)
    print(f"[DEBUG] step={step}  (picked token hop histogram + pad vs real-fallback + available NL hops)")
    print("-" * 90)

    # ---- A) Available hop buckets in the NL-sampled disjoint subgraph ----
    if dbg_bucket is None:
        print("[Available NL hops] debug_bucket_counts not found (add it in relgt_from_neighborloader_batch_new).")
        avail_cols = None
    else:
        avail_cols = dbg_bucket[:, 1 : max_hop + 1]  # [B, max_hop]
        print(f"[Available NL hops] B={B}")
        if dbg_total is not None:
            print(f"  reachable_total: mean={dbg_total.float().mean().item():.2f}  "
                  f"min={int(dbg_total.min().item())}  max={int(dbg_total.max().item())}")
        for d in range(1, max_hop + 1):
            col = avail_cols[:, d - 1]
            print(f"  avail hop{d}: total={_fmt_int(col.sum().item())}  "
                  f"mean/seed={col.float().mean().item():.2f}  "
                  f"min={int(col.min().item())}  max={int(col.max().item())}")

        avail_total = avail_cols.sum(dim=1)
        zero_reach = int((avail_total == 0).sum().item())
        lt_budget = int(((avail_total > 0) & (avail_total < budget)).sum().item())
        ge_budget = int((avail_total >= budget).sum().item())
        print(f"  seeds w/ reachable_total==0: {zero_reach}/{B}")
        print(f"  seeds w/ 0<reachable_total<budget (replacement likely used): {lt_budget}/{B}")
        print(f"  seeds w/ reachable_total>=budget: {ge_budget}/{B}")

    print("")

    # ---- B) Picked hop histogram + pad vs real-fallback ----
    print(f"[Picked K tokens] B={B}  K={K}  budget={budget}")
    for d in range(1, max_hop + 1):
        c = int((hops_ns == d).sum().item())
        print(f"  picked hop{d}: total={_fmt_int(c)}  mean/seed={c / float(B):.2f}")

    c_fb = int(fb_mask.sum().item())
    c_realfb = int(realfb_mask.sum().item())
    c_pad = int(pad_mask.sum().item())
    print(f"  picked fallback(hop_id={fallback_hop_id}): total={_fmt_int(c_fb)}  mean/seed={c_fb / float(B):.2f}")
    print(f"    ├─ real fallback nodes (type!=mask): total={_fmt_int(c_realfb)}  mean/seed={c_realfb / float(B):.2f}")
    print(f"    └─ mask padding tokens (type==mask): total={_fmt_int(c_pad)}  mean/seed={c_pad / float(B):.2f}")

    # ---- C) Seed-level case counts ----
    seeds_any_pad = int((pad_per_seed > 0).sum().item())
    seeds_all_pad = int((pad_per_seed == budget).sum().item())
    seeds_any_realfb = int((realfb_per_seed > 0).sum().item())
    seeds_all_realfb = int((realfb_per_seed == budget).sum().item())

    print("\n[Seed-level case counts]")
    print(f"  seeds with ANY padding: {seeds_any_pad}/{B}")
    print(f"  seeds with ALL padding (pad==budget): {seeds_all_pad}/{B}")
    print(f"  seeds with ANY real-fallback: {seeds_any_realfb}/{B}")
    print(f"  seeds with ALL real-fallback (realfb==budget): {seeds_all_realfb}/{B}")

    # ---- D) First few seeds: available vs picked ----
    show_n = min(show_n, B)
    print("\n  first few seeds breakdown:")
    for i in range(show_n):
        picked_h = [int((hops_ns[i] == d).sum().item()) for d in range(1, max_hop + 1)]
        real_fb = int(realfb_per_seed[i].item())
        pad = int(pad_per_seed[i].item())

        if avail_cols is not None:
            avail_h = [int(avail_cols[i, d - 1].item()) for d in range(1, max_hop + 1)]
            avail_str = ", ".join([f"avail_h{d}={avail_h[d-1]}" for d in range(1, max_hop + 1)])
            picked_str = ", ".join([f"h{d}={picked_h[d-1]}" for d in range(1, max_hop + 1)])
            print(f"    seed[{i}]: {avail_str} | {picked_str}, real_fb={real_fb}, pad={pad}")
        else:
            picked_str = ", ".join([f"h{d}={picked_h[d-1]}" for d in range(1, max_hop + 1)])
            print(f"    seed[{i}]: {picked_str}, real_fb={real_fb}, pad={pad}")

    print("=" * 90 + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-top3")
    parser.add_argument("--precompute", action="store_true", default=False)
    parser.add_argument("--sampling_backend", type=str, default="precompute", choices=["precompute", "neighborloader"])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--gt_conv_type", type=str, default="full")
    parser.add_argument("--ablate", type=str, default="none")
    parser.add_argument("--gnn_pe_dim", type=int, default=0)
    parser.add_argument("--num_neighbors", type=int, default=300)
    parser.add_argument("--num_centroids", type=int, default=4096)
    parser.add_argument("--ff_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--temporal_strategy", type=str, default="uniform")
    parser.add_argument("--pos_enc", type=str, default="none")
    parser.add_argument("--max_degree", type=int, default=10000)
    parser.add_argument("--pos_enc_dim", type=int, default=128)
    parser.add_argument("--max_steps_per_epoch", type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/debug")
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--model_parameters", type=int, default=0)
    parser.add_argument("--nl_num_neighbors", type=str, default="32,32,32,32")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples"))
    parser.add_argument("--train_stage", type=str, default="finetune", choices=["finetune"])
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    parser.add_argument("--log_file", type=str, default="experiment.log")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    # args.nl_num_neighbors = parse_int_list(args.nl_num_neighbors)
    # args.nl_max_hop = len(args.nl_num_neighbors)
    # args.fallback_hop_id = args.nl_max_hop + 1
    # args.max_neighbor_hop = args.fallback_hop_id
    # print(args.nl_num_neighbors)
    # exit()
    
    args.nl_num_neighbors_obj = parse_nl_num_neighbors(args.nl_num_neighbors)
    if isinstance(args.nl_num_neighbors_obj, NumNeighbors):
        args.nl_max_hop = int(args.nl_num_neighbors_obj.num_hops)
    else:
        args.nl_max_hop = len(args.nl_num_neighbors_obj)
        
    args.fallback_hop_id = args.nl_max_hop + 1
    args.max_neighbor_hop = args.fallback_hop_id
    nl_num_neighbors_for_loader = args.nl_num_neighbors_obj

    dist.init_process_group(backend="nccl")

    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Build the same directory you already use for checkpoints:
    output_path = os.path.join(args.out_dir, args.dataset, args.task)
    os.makedirs(output_path, exist_ok=True)

    logger = setup_experiment_logger(Path(output_path) / args.log_file, rank)
    # logger = setup_experiment_logger(Path(args.out_dir) / "experiment.log", rank)

    if torch.cuda.is_available():
        torch.set_num_threads(1)

    seed_everything(args.seed)
    gpu_handle = init_gpu_utilization(local_rank)

    if local_rank == 0:
        args.run_name = f"{args.dataset}-{args.task}-{args.run_name}"

    if rank == 0:
        logger.info(f"Run starting. run_name={args.run_name} out_dir={args.out_dir} world_size={world_size} seed={args.seed}")

    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)

    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    rb_graph.to_unix_time = to_unix_time_safe

    hetero_data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=f"cuda:{local_rank}"),
            batch_size=256,
        ),
        cache_dir=os.path.join(args.cache_dir, f"{args.dataset}_glove_materialized_cache"),
    )

    cast_time_tensors_inplace(hetero_data, time_attr="time")

    token_precompute = args.precompute and (args.sampling_backend == "precompute")
    data = {
        split: RelGTTokens(
            data=hetero_data,
            task=task,
            K=args.num_neighbors,
            split=split,
            undirected=True,
            precompute=token_precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
            num_workers=args.num_workers,
            train_stage=args.train_stage,
        )
        for split in ["train", "val", "test"]
    }

    clamp_min, clamp_max = None, None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "multilabel_auprc_macro"
        higher_is_better = True
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")

    model = RelGT(
        num_nodes=data["train"].data.num_nodes,
        max_neighbor_hop=args.max_neighbor_hop,
        node_type_map=data["train"].node_type_to_index,
        col_names_dict={nt: data["train"].data[nt].tf.col_names_dict for nt in data["train"].data.node_types},
        col_stats_dict=col_stats_dict,
        local_num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        global_dim=args.channels // 2,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        conv_type=args.gt_conv_type,
        ablate=args.ablate,
        gnn_pe_dim=args.gnn_pe_dim,
        num_centroids=args.num_centroids,
        sample_node_len=args.num_neighbors,
        args=args,
    ).to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.model_parameters = total_params

    # if local_rank == 0:
    #     wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))

    # output_path = os.path.join(args.out_dir, args.dataset, args.task)
    # os.makedirs(output_path, exist_ok=True)
    
    # best_ckpt_path = os.path.join(output_path, "finetuned.pt")
    # best_ckpt_path = os.path.join(output_path, "finetuned.pt")

    # base_lr = args.lr * world_size
    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    # loader_dict: Dict[str, DataLoader] = {}
    
    best_ckpt_path = os.path.join(output_path, "finetuned.pt")

    base_lr = args.lr * world_size
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    # ---- resume state ----
    start_epoch = 1
    best_val_metric = -math.inf if higher_is_better else math.inf
    epochs_no_improve = 0

    if args.resume_ckpt is not None and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(ckpt["epoch"]) + 1
        best_val_metric = float(ckpt.get("best_val_metric", best_val_metric))

        if rank == 0:
            logger.info(
                f"Resumed from {args.resume_ckpt} "
                f"(saved_epoch={ckpt['epoch']}, start_epoch={start_epoch}, "
                f"best_val_metric={best_val_metric})"
            )

    loader_dict: Dict[str, DataLoader] = {}
    
    
    split_len_dict: Dict[str, int] = {}
    seed_type_dict: Dict[str, str] = {}
    seed_id_to_row_dict: Dict[str, Dict[int, int]] = {}

    seed_ids_local_dict: Dict[str, torch.Tensor] = {}
    seed_times_local_dict: Dict[str, Optional[torch.Tensor]] = {}
    row_ids_local_dict: Dict[str, torch.Tensor] = {}
    targets_local_dict: Dict[str, Optional[torch.Tensor]] = {}

    train_pack = {}
    if args.sampling_backend == "precompute":
        train_sampler = DistributedSampler(data["train"], shuffle=True, seed=args.seed)
        loader_dict["train"] = DataLoader(
            data["train"],
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=data["train"].collate,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )

        val_sampler = DistributedSampler(data["val"], shuffle=False, seed=args.seed, drop_last=False)
        loader_dict["val"] = DataLoader(
            data["val"],
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=data["val"].collate,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )

        test_sampler = DistributedSampler(data["test"], shuffle=False, seed=args.seed, drop_last=False)
        loader_dict["test"] = DataLoader(
            data["test"],
            batch_size=args.batch_size,
            sampler=test_sampler,
            collate_fn=data["test"].collate,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )
    else:
        for split in ["train", "val", "test"]:
            table = task.get_table(split)
            table_input = get_node_train_table_input(table=table, task=task)
            seed_type, seed_ids_all = table_input.nodes
            seed_times_all = getattr(table_input, "time", None)
            targets_all = getattr(table_input, "target", None)
            transform = getattr(table_input, "transform", None)

            seed_ids_all = seed_ids_all.to(torch.long)
            if seed_times_all is not None:
                seed_times_all = seed_times_all.to(dtype=infer_time_dtype(hetero_data, seed_type, time_attr="time"))

            split_len_dict[split] = int(len(seed_ids_all))
            seed_type_dict[split] = seed_type

            global_row_ids_all = torch.arange(len(seed_ids_all), dtype=torch.long)

            if split == "train":
                train_pack = {
                    "seed_type": seed_type,
                    "seed_ids_all": seed_ids_all,
                    "seed_times_all": seed_times_all,
                    "row_ids_all": global_row_ids_all,
                    "targets_all": targets_all,
                    "transform": transform,
                }
                continue

            seed_ids_local = seed_ids_all[local_rank::world_size]
            seed_times_local = None
            if seed_times_all is not None:
                seed_times_local = seed_times_all[local_rank::world_size]
            row_ids_local = global_row_ids_all[local_rank::world_size]
            targets_local = targets_all[local_rank::world_size] if targets_all is not None else None

            seed_ids_local_dict[split] = seed_ids_local
            seed_times_local_dict[split] = seed_times_local
            row_ids_local_dict[split] = row_ids_local
            targets_local_dict[split] = targets_local

            loader_dict[split] = NeighborLoader(
                hetero_data,
                input_nodes=(seed_type, seed_ids_local),
                input_time=seed_times_local,
                time_attr="time",
                temporal_strategy=args.temporal_strategy,
                transform=transform,
                # num_neighbors=args.nl_num_neighbors,
                num_neighbors=nl_num_neighbors_for_loader,
                disjoint=True,
                subgraph_type="directional",
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
            )

    global_step = 0

    def make_train_neighborloader(epoch: int):
        seed_type = train_pack["seed_type"]
        seed_ids_all = train_pack["seed_ids_all"]
        seed_times_all = train_pack["seed_times_all"]
        row_ids_all = train_pack["row_ids_all"]
        targets_all = train_pack["targets_all"]
        transform = train_pack["transform"]

        seed_ids_local, seed_times_local, row_ids_local, targets_local = build_epoch_train_inputs(
            seed_ids_all=seed_ids_all,
            seed_times_all=seed_times_all,
            row_ids_all=row_ids_all,
            targets_all=targets_all,
            epoch=epoch,
            seed=args.seed,
            rank=local_rank,
            world_size=world_size,
        )

        loader = NeighborLoader(
            hetero_data,
            input_nodes=(seed_type, seed_ids_local),
            input_time=seed_times_local,
            time_attr="time",
            temporal_strategy=args.temporal_strategy,
            transform=transform,
            # num_neighbors=args.nl_num_neighbors,
            num_neighbors=nl_num_neighbors_for_loader,
            disjoint=True,
            subgraph_type="directional",
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        pack = {
            "seed_ids_local": seed_ids_local,
            "seed_times_local": seed_times_local,
            "row_ids_local": row_ids_local,
            "targets_local": targets_local,
        }
        return loader, seed_type, pack

    def train_supervised(epoch: int) -> float:
        nonlocal global_step
        model.train()
        loss_accum = 0.0
        count_accum = 0

        if args.sampling_backend == "precompute":
            assert isinstance(loader_dict["train"].sampler, DistributedSampler)
            loader_dict["train"].sampler.set_epoch(epoch)  # type: ignore[union-attr]
            train_loader = loader_dict["train"]
            seed_type = None
            seed_pack = None
        else:
            train_loader, seed_type, seed_pack = make_train_neighborloader(epoch)

        total_steps = min(len(train_loader), args.max_steps_per_epoch)

        it = tqdm(
            train_loader,
            total=total_steps,
            desc=f"Train (epoch {epoch})",
            disable=(local_rank != 0),
        )

        for step, batch in enumerate(it, start=1):
            if args.sampling_backend == "neighborloader":
                if local_rank == 0 and step == 1:
                    seed_store = batch[seed_type]
                    B = int(seed_store.batch_size)

                    inp = seed_store.input_id[:B].cpu().long()
                    seeds_from_batch = seed_store.n_id[:B].cpu().long()

                    seeds_from_inputs = seed_pack["seed_ids_local"][inp].cpu().long()

                    assert torch.equal(seeds_from_inputs, seeds_from_batch), (
                        "Seed mapping mismatch -> labels/global_idx likely wrong. "
                        "NeighborLoader guarantees first batch_size nodes are seeds; "
                        "input_id should map back to input_nodes indices."
                    )
                
                batch = relgt_from_neighborloader_batch(
                    nl_batch=batch,
                    hetero_data=hetero_data,
                    seed_type=seed_type,  # type: ignore[arg-type]
                    K=args.num_neighbors,
                    seed_ids_local=seed_pack["seed_ids_local"],  # type: ignore[index]
                    seed_times_local=seed_pack["seed_times_local"],  # type: ignore[index]
                    row_ids_local=seed_pack["row_ids_local"],  # type: ignore[index]
                    targets_local=seed_pack["targets_local"],  # type: ignore[index]
                    max_hop=args.nl_max_hop,
                    fallback_hop_id=args.fallback_hop_id,
                    time_attr="time",
                    undirected=True,
                )
                
                # # --- DEBUG PRINTS for step 1-2 (rank 0 only) ---
                # if local_rank == 0 and step <= DEBUG_STEPS:
                #     _print_hop_debug(
                #         batch,
                #         step=step,
                #         max_hop=args.nl_max_hop,
                #         fallback_hop_id=args.fallback_hop_id,
                #     )
                # # --- EXIT after step 2 (ALL ranks exit to avoid DDP hangs) ---
                # if step == DEBUG_STEPS:
                #     if dist.is_initialized():
                #         dist.barrier()
                #         dist.destroy_process_group()
                #     raise SystemExit(0)


            neighbor_types = batch["neighbor_types"].to(device)
            node_indices = batch["node_indices"].to(device)
            neighbor_hops = batch["neighbor_hops"].to(device)
            neighbor_times = batch["neighbor_times"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec = batch["batch"].to(device)

            grouped_tf_dict = {
                "grouped_tfs": batch["grouped_tfs"],
                "grouped_indices": batch["grouped_indices"],
                "flat_batch_idx": batch["flat_batch_idx"],
                "flat_nbr_idx": batch["flat_nbr_idx"],
            }
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            pred = model(
                neighbor_types,
                node_indices,
                neighbor_hops,
                neighbor_times,
                grouped_tf_dict,
                edge_index=edge_index,
                batch=batch_vec,
            )
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            loss = loss_fn(pred.float(), labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = float(loss.detach().item())
            gpu_util, mem_allocated, mem_reserved = get_gpu_stats(gpu_handle, device)

            # if local_rank == 0:
            #     wandb.log(
            #         {
            #             "train_loss": loss_value,
            #             "global_step": global_step,
            #             "lr": optimizer.param_groups[0]["lr"],
            #             "gpu_util_percent": gpu_util,
            #             "gpu_mem_allocated_MB": mem_allocated,
            #             "gpu_mem_reserved_MB": mem_reserved,
            #             "epoch": epoch,
            #         }
            #     )

            loss_accum += loss_value * labels.size(0)
            count_accum += labels.size(0)
            global_step += 1

            if step >= args.max_steps_per_epoch:
                break

        return loss_accum / max(count_accum, 1)

    @torch.no_grad()
    def evaluate(split: str, eval_model) -> Optional[np.ndarray]:
        eval_model.eval()
        loader = loader_dict[split] if args.sampling_backend == "precompute" else loader_dict[split]
        seed_type = seed_type_dict.get(split, None)

        pred_list = []
        idx_list = []

        for batch in tqdm(loader, desc=split.capitalize(), disable=(local_rank != 0)):
            if args.sampling_backend == "neighborloader":
                batch = relgt_from_neighborloader_batch(
                    nl_batch=batch,
                    hetero_data=hetero_data,
                    seed_type=seed_type,
                    K=args.num_neighbors,
                    seed_ids_local=seed_ids_local_dict[split],
                    seed_times_local=seed_times_local_dict[split],
                    row_ids_local=row_ids_local_dict[split],
                    targets_local=targets_local_dict[split],
                    max_hop=args.nl_max_hop,
                    fallback_hop_id=args.fallback_hop_id,
                    time_attr="time",
                    undirected=True,
                )

            neighbor_types = batch["neighbor_types"].to(device)
            node_indices = batch["node_indices"].to(device)
            neighbor_hops = batch["neighbor_hops"].to(device)
            neighbor_times = batch["neighbor_times"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec = batch["batch"].to(device)

            grouped_tf_dict = {
                "grouped_tfs": batch["grouped_tfs"],
                "grouped_indices": batch["grouped_indices"],
                "flat_batch_idx": batch["flat_batch_idx"],
                "flat_nbr_idx": batch["flat_nbr_idx"],
            }

            pred = eval_model(
                neighbor_types,
                node_indices,
                neighbor_hops,
                neighbor_times,
                grouped_tf_dict,
                edge_index=edge_index,
                batch=batch_vec,
            )

            if task.task_type == TaskType.REGRESSION:
                pred = torch.clamp(pred, clamp_min, clamp_max)
            if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
                pred = torch.sigmoid(pred)

            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu().numpy())
            idx_list.append(batch["global_idx"].cpu().numpy())

        local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
        local_idxs = np.concatenate(idx_list, axis=0) if idx_list else np.array([], dtype=np.int64)

        gathered = [None for _ in range(world_size)] if local_rank == 0 else None
        dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)

        if local_rank != 0:
            return None
        

        total_len = split_len_dict[split] if args.sampling_backend == "neighborloader" else len(loader.dataset)

        first_pred = None
        for i in range(world_size):
            _, gp = gathered[i]
            if gp is not None and np.size(gp) > 0:
                first_pred = gp
                break

        if first_pred is None:
            return np.array([])

        if first_pred.ndim == 1:
            all_preds = np.full((total_len,), -100.0, dtype=np.float32)
            for i in range(world_size):
                g_idx, g_pred = gathered[i]
                for idx, p in zip(g_idx, g_pred):
                    all_preds[int(idx)] = float(p)
            return all_preds

        all_preds = np.full((total_len, first_pred.shape[1]), -100.0, dtype=np.float32)
        for i in range(world_size):
            g_idx, g_pred = gathered[i]
            for idx, p in zip(g_idx, g_pred):
                all_preds[int(idx)] = p
        return all_preds

    if args.train_stage == "finetune":
        best_state = None

        if rank == 0:
            logger.info(
                f"Training about to start (right before epoch {start_epoch}). "
                f"Timestamp={datetime.now().isoformat(timespec='seconds')}"
            )

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_supervised(epoch)
            dist.barrier()

            val_pred = evaluate("val", eval_model=model.module)
            if local_rank == 0:
                val_metrics = task.evaluate(val_pred, task.get_table("val"))
                val_metrics = {k: float(v) for k, v in val_metrics.items()}

                logger.info(f"Epoch: {epoch:02d}, Train loss: {train_loss:.6f}, Val metrics: {val_metrics}")

                # wandb.log({"epoch_train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

                current = float(val_metrics[tune_metric])

                # STRICT improvement
                improved = (current > best_val_metric) if higher_is_better else (current < best_val_metric)

                should_stop = False
                if improved:
                    logger.info(f"{tune_metric} improved. Saving finetuned.pt to {output_path}")
                    best_val_metric = current
                    epochs_no_improve = 0

                    best_state = copy.deepcopy(model.module.state_dict())
                    args_to_save = vars(args).copy()
                    args_to_save.pop("nl_num_neighbors_obj", None)
                    ckpt = {
                        "epoch": epoch,
                        "best_val_metric": best_val_metric,
                        "tune_metric": tune_metric,
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": args_to_save,
                    }
                    torch.save(ckpt, best_ckpt_path)
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement for {epochs_no_improve}/{args.early_stop_patience} epoch(s).")

                    if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch}. Best {tune_metric}={best_val_metric:.6f}"
                        )
                        should_stop = True

                # ---- make all ranks stop together ----
                stop_tensor = torch.tensor([1 if (rank == 0 and should_stop) else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                if int(stop_tensor.item()) == 1:
                    break
            dist.barrier()

        # if local_rank == 0 and best_state is not None:
        #     model.module.load_state_dict(best_state)
        # for p in model.parameters():
        #     dist.broadcast(p.data, src=0)
        # for b in model.buffers():
        #     dist.broadcast(b.data, src=0)
        # dist.barrier()
        if local_rank == 0:
            ckpt = torch.load(best_ckpt_path, map_location="cpu") 
            
            model.module.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Loaded best checkpoint from disk: {best_ckpt_path} "
                f"(epoch={ckpt.get('epoch')}, best_val_metric={ckpt.get('best_val_metric')}, tune_metric={ckpt.get('tune_metric')})"
            )
        dist.barrier()
        

        final_val_preds = evaluate("val", eval_model=model.module)
        final_test_preds = evaluate("test", eval_model=model.module)

        if local_rank == 0:
            val_metrics = task.evaluate(final_val_preds, task.get_table("val"))
            test_metrics = task.evaluate(final_test_preds)

            out = {"val_metrics": val_metrics, "test_metrics": test_metrics}
            with open(os.path.join(output_path, f"{args.seed}.json"), "w") as f:
                json.dump(out, f, indent=4)

            logger.info(f"Final val_metrics={ {k: float(v) for k, v in val_metrics.items()} }")
            logger.info(f"Final test_metrics={ {k: float(v) for k, v in test_metrics.items()} }")
            logger.info("Run finished.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
