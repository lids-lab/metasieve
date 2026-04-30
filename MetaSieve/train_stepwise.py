import os
import ast
import sys 
import json
import math
import copy
import torch
import logging
import warnings
import argparse
import numpy as np
import importlib.util
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import torch_geometric.backend

from model import Model, HGTModel
from embeddings import GloveTextEmbedding

from relbench.tasks import get_task
from relbench.datasets import get_dataset
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from torch_geometric.sampler.base import NumNeighbors
from torch_geometric.loader import NeighborLoader
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input

# Disable segment_matmul for rel-trial / rel-ratebeer (HGT incompatibility)
if any(d in sys.argv for d in ("rel-ratebeer", "rel-trial")):
    torch_geometric.backend.use_segment_matmul = False


warnings.filterwarnings("once")
seed_everything(42)


def parse_args():
    p = argparse.ArgumentParser("RelBench step-wise training")

    p.add_argument("--dataset_name", type=str, default="rel-stack")
    p.add_argument("--output_root", type=str, default="outputs_final")
    p.add_argument("--task_name", type=str, default="user-engagement")
    p.add_argument("--task_kind", type=str, choices=["classification", "regression"], default="classification")
    p.add_argument("--num_layers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--text_emb_batch_size", type=int, default=256)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--max_steps_per_epoch", type=int, default=1000)  # <=0 means full epoch

    p.add_argument("--fanout", type=int, default=64)
    p.add_argument("--sampling_cfg_path", type=str, default="")

    p.add_argument("--model_type", type=str, choices=["hgs", "hgt"], default="hgt")

    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="none")

    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", type=str, default="cuda:0")  # always use cuda:0 if available

    return p.parse_args()


def num_neighbors_from_json(cfg_path: str | Path) -> tuple[int, int, NumNeighbors]:
    cfg = json.loads(Path(cfg_path).read_text())
    L = cfg["num_layers"]
    base = cfg["base_neighbors"]
    neighbors = {ast.literal_eval(k): v for k, v in cfg["neighbors"].items()}
    return L, base, NumNeighbors(neighbors, default=[base] * L)


def num_neighbors_from_py(cfg_path: str | Path) -> NumNeighbors | list[int] | dict:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Sampling config not found: {cfg_path}")

    # NOTE: This executes the Python file. Only load trusted configs.
    module_name = f"sampling_cfg_{cfg_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(cfg_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {cfg_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "num_neighbors"):
        raise AttributeError(
            f"{cfg_path} must define a top-level variable named `num_neighbors`"
        )

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

def data_loader(data, task, train_table, val_table, test_table, num_neighbors, batch_size: int):
    loader_dict = {}
    entity_table = None

    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        table_input = get_node_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0] if entity_table is None else entity_table

        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=getattr(table_input, "time", None),
            transform=getattr(table_input, "transform", None),
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=(split == "train"),
            num_workers=0,
            persistent_workers=False,
        )

    assert entity_table is not None
    return loader_dict, entity_table


def train(
    model,
    optimizer,
    data_loader_instance,
    task,
    loss_fn,
    device,
    max_steps_per_epoch: Optional[int] = None,
    epoch: Optional[int] = None,
) -> float:
    model.train()

    loss_accum = 0.0
    count_accum = 0

    train_loader = data_loader_instance["train"]
    total_steps = len(train_loader) if max_steps_per_epoch is None else min(len(train_loader), max_steps_per_epoch)
    desc = "Training" if epoch is None else f"Training (epoch {epoch})"

    for step, batch in enumerate(tqdm(train_loader, desc=desc, total=total_steps), start=1):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch, task.entity_table)
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.view(-1)

        loss = loss_fn(pred.float(), batch[task.entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += float(loss.detach().item()) * pred.size(0)
        count_accum += pred.size(0)

        if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
            break

    return loss_accum / max(count_accum, 1)


@torch.no_grad()
def test(model, loader, task, device, task_kind: str, clamp_min=None, clamp_max=None) -> np.ndarray:
    model.eval()
    pred_list = []

    for batch in tqdm(loader, desc="Testing"):
        batch = batch.to(device)
        pred = model(batch, task.entity_table)

        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.view(-1)

        if task_kind == "classification":
            pred = torch.sigmoid(pred)
        elif task_kind == "regression" and clamp_min is not None and clamp_max is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)

        pred_list.append(pred.detach().cpu())

    return torch.cat(pred_list, dim=0).numpy()


if __name__ == "__main__":
    args = parse_args()
    

    DATASET_NAME = args.dataset_name
    OUTPUT_ROOT = Path(args.output_root)
    TASK_NAME = args.task_name
    TASK_KIND = args.task_kind
    NUM_LAYERS = args.num_layers

    epochs = args.epochs
    BATCH_SIZE = args.batch_size
    TEXT_EMB_BATCH_SIZE = args.text_emb_batch_size
    EARLY_STOP_PATIENCE = args.early_stop_patience
    MAX_STEPS_PER_EPOCH = None if args.max_steps_per_epoch <= 0 else int(args.max_steps_per_epoch)

    FANOUT = args.fanout
    SAMPLING_CFG_PATH = args.sampling_cfg_path.strip()

    model_type = args.model_type.lower()
    # sampling_tag = "pruning" if SAMPLING_CFG_PATH else "random"
    if not SAMPLING_CFG_PATH:
        sampling_tag = "random"
    else:
        suffix = Path(SAMPLING_CFG_PATH).suffix.lower()
        if suffix == ".json":
            sampling_tag = "pruning"
        elif suffix == ".py":
            sampling_tag = "mpspruning"
        else:
            raise ValueError(
                f"Unsupported sampling config type: {suffix} (expected .json or .py)"
            )

    if args.device == "cpu" or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    root_dir = "./data"

    if TASK_KIND == "classification":
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
        metric_display_name = "AUROC"
    elif TASK_KIND == "regression":
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        metric_display_name = "MAE"
    else:
        raise ValueError("TASK_KIND must be 'classification' or 'regression'")

    dataset = get_dataset(DATASET_NAME, download=True)
    task = get_task(DATASET_NAME, TASK_NAME, download=True)

    run_dir = OUTPUT_ROOT / DATASET_NAME / TASK_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    # model_tag = "hgt" if model_type == "hgt" else "hgs"
    # log_file = run_dir / f"{DATASET_NAME}_{TASK_NAME}_{model_tag}_{sampling_tag}_fanout_{FANOUT}_batchsize_{BATCH_SIZE}.log"
    
    model_tag = "hgt" if model_type == "hgt" else "hgs"

    layers_tag = f"layers_{NUM_LAYERS}"
    heads_tag = f"head_{args.heads}" if model_type == "hgt" else None  # only for attention model

    parts = [
        DATASET_NAME,
        TASK_NAME,
        model_tag,
        layers_tag,
    ]

    if heads_tag is not None:
        parts.append(heads_tag)

    parts += [
        sampling_tag,
        f"fanout_{FANOUT}",
        f"batchsize_{BATCH_SIZE}",
    ]

    log_file = run_dir / ("_".join(parts) + ".log")

    base_name = log_file.stem
    model_save_path = run_dir / "checkpoints" / base_name
    model_save_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=TEXT_EMB_BATCH_SIZE,
    )

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=os.path.join(root_dir, f"{DATASET_NAME}_glove_materialized_cache"),
    )

    # if SAMPLING_CFG_PATH:
    #     _, _, num_neighbors = num_neighbors_from_json(SAMPLING_CFG_PATH)
    # else:
    #     num_neighbors = [FANOUT for _ in range(NUM_LAYERS)]
    if not SAMPLING_CFG_PATH:
        num_neighbors = [FANOUT for _ in range(NUM_LAYERS)]
    else:
        suffix = Path(SAMPLING_CFG_PATH).suffix.lower()
        if suffix == ".json":
            _, _, num_neighbors = num_neighbors_from_json(SAMPLING_CFG_PATH)
        elif suffix == ".py":
            num_neighbors = num_neighbors_from_py(SAMPLING_CFG_PATH)
        else:
            raise ValueError(
                f"Unsupported sampling config type: {suffix} (expected .json or .py)"
            )
    

    data_loader_instance, entity_table = data_loader(
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
        num_neighbors=num_neighbors,
        batch_size=BATCH_SIZE,
    )

    clamp_min = clamp_max = None
    if TASK_KIND == "regression":
        y_np = train_table.df[task.target_col].to_numpy()
        clamp_min, clamp_max = np.percentile(y_np, [2, 98])

    if model_type == "hgt":
        model = HGTModel(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=NUM_LAYERS,
            channels=256,
            out_channels=out_channels,
            aggr="sum",
            norm="batch_norm",
            heads=args.heads,
            dropout=args.dropout,
            act=args.act,
        ).to(device)
    else:
        model = Model(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=NUM_LAYERS,
            channels=256,
            out_channels=out_channels,
            aggr="sum",
            norm="batch_norm",
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_metric = -math.inf if higher_is_better else math.inf
    best_epoch = 0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss = train(
            model=model,
            optimizer=optimizer,
            data_loader_instance=data_loader_instance,
            task=task,
            loss_fn=loss_fn,
            device=device,
            max_steps_per_epoch=MAX_STEPS_PER_EPOCH,
            epoch=epoch,
        )

        val_pred = test(
            model,
            data_loader_instance["val"],
            task=task,
            device=device,
            task_kind=TASK_KIND,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        val_metrics = task.evaluate(val_pred, val_table)
        val_metrics = {k: float(v) for k, v in val_metrics.items()}

        print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
        logging.info(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

        current_metric = float(val_metrics[tune_metric])
        improved = (current_metric > best_val_metric) if higher_is_better else (current_metric < best_val_metric)

        if improved:
            print(f"{metric_display_name} improved from {best_val_metric:.4f} to {current_metric:.4f}. Saving model...")
            logging.info(f"{metric_display_name} improved from {best_val_metric:.4f} to {current_metric:.4f}. Saving model...")

            best_val_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_metric": best_val_metric,
            }
            torch.save(
                checkpoint,
                str(Path(model_save_path) / f"model_{tune_metric}:{current_metric:.6f}_epoch:{epoch}.pth"),
            )
        else:
            epochs_no_improve += 1
            print(f"No improvement in {metric_display_name} for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epoch(s).")
            logging.info(f"No improvement in {metric_display_name} for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epoch(s).")

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered. Best {metric_display_name}={best_val_metric:.4f} at epoch {best_epoch}.")
                logging.info(f"Early stopping triggered. Best {metric_display_name}={best_val_metric:.4f} at epoch {best_epoch}.")
                break

    model.load_state_dict(best_model_wts)

    test_pred = test(
        model,
        data_loader_instance["test"],
        task=task,
        device=device,
        task_kind=TASK_KIND,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    # test_metrics = task.evaluate(test_pred, test_table)
    test_metrics = task.evaluate(test_pred)
    test_metrics = {k: float(v) for k, v in test_metrics.items()}

    print(f"Best Epoch={best_epoch}, Best Val {metric_display_name}={best_val_metric:.4f}, Test metrics={test_metrics}")
    logging.info(f"Best Epoch={best_epoch}, Best Val {metric_display_name}={best_val_metric:.4f}, Test metrics={test_metrics}")
