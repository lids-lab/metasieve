# mps_wrapper.py
import ast
import copy
import logging
import importlib
import torch
from typing import Any, Iterable, Optional, Sequence, Tuple, List
from tqdm.auto import tqdm 
from utils import mpgnn  

def _eval_mpgnn_on_split(data_mpgnn, metapaths, hidden_dim, split="val"):
    assert split in ("val", "test")
    orig_test_mask = data_mpgnn.test_mask.clone()
    try:
        if split == "val":
            data_mpgnn.test_mask = data_mpgnn.val_mask.clone()
        f1, auc = mpgnn(data_mpgnn, metapaths, pre_trained_model=[], hidden_dim=hidden_dim)
    finally:
        data_mpgnn.test_mask = orig_test_mask
    return float(f1), float(auc)


def _top_bottom(scored: Sequence[Tuple[Any, float, Any]], k_best=5, k_worst=5):
    if not scored:
        return [], []
    scored_sorted = sorted(scored, key=lambda t: float(t[1]))  # lower loss is better
    best = scored_sorted[: min(k_best, len(scored_sorted))]
    worst = scored_sorted[-min(k_worst, len(scored_sorted)) :] if k_worst > 0 else []
    return best, worst


def _topk_by_loss(scored: Sequence[Tuple[Any, float, Any]], k: int):
    if not scored or k <= 0:
        return []
    return sorted(scored, key=lambda t: float(t[1]))[: min(k, len(scored))]


def _decode_rel(rel_id: int, edge_type_names):
    if isinstance(edge_type_names, (list, tuple)) and 0 <= rel_id < len(edge_type_names):
        return edge_type_names[rel_id]
    if isinstance(edge_type_names, dict):
        return edge_type_names.get(rel_id)
    return None


def _get_or_make_logger(
    logger: Optional[logging.Logger],
    log_path: Optional[str],
    level: int = logging.INFO,
    name: str = "mps_search",
) -> Optional[logging.Logger]:
    if logger is not None:
        return logger
    if not log_path:
        return None

    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False

    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path for h in lg.handlers):
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        lg.addHandler(fh)

    return lg


def run_mps_metapath_search(
    data_score,
    data_mpgnn,
    hidden_dim: int,
    K: int = 2,
    L_MAX: int = 3,
    sampling_size: float = 0.7,
    eval_split: str = "val",
    device: torch.device | None = None,
    *,
    rank_metric: str = "auc_roc",          # "f1" or "auc_roc"
    logger: Optional[logging.Logger] = None,
    log_path: Optional[str] = None,
    log_to_console: bool = False,
    k_best_log: int = 10,
    k_worst_log: int = 10,
    max_frontier_logs: int = 50,
    beam_width: Optional[int] = None,
):
    """
    Top-K expansion MPS search WITHOUT cloning/storing Data objects.
    Stores only metapath + theta sequence (+ metrics) and reconstructs needed Data
    states on-the-fly via create_new_data replay.

    rank_metric:
      - "f1"     -> decisions by F1
      - "auc_roc" / "roc_auc" / "auroc" / "auc" -> decisions by ROC-AUC
    """
    import importlib
    import copy
    import torch
    from tqdm.auto import tqdm

    mps_main = importlib.import_module("main")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mps_main.device = device

    # ---- Normalize metric choice ----
    rank_metric_norm = (rank_metric or "f1").strip().lower()
    if rank_metric_norm in {"auc_roc", "roc_auc", "auroc", "auc"}:
        metric_key = "auc_roc"
        metric_display = "auc_roc"
        higher_is_better = True
    elif rank_metric_norm == "f1":
        metric_key = "f1"
        metric_display = "f1"
        higher_is_better = True
    else:
        raise ValueError(f"Unknown rank_metric='{rank_metric}'. Use 'f1' or 'auc_roc'.")

    # ---- Logger ----
    edge_type_names = getattr(data_score, "edge_type_names", None) or getattr(data_mpgnn, "edge_type_names", None)
    lg = _get_or_make_logger(logger, log_path)

    def emit(msg: str):
        if lg is not None:
            lg.info(msg)
        if log_to_console or lg is None:
            tqdm.write(msg)

    def log_rel_list(prefix: str, rels: Sequence[Tuple[Any, float, Any]]):
        emit(prefix)
        for rel, loss, _theta in rels:
            rid = int(rel)
            name = _decode_rel(rid, edge_type_names)
            emit(f"  rel={rid:<4d} loss={float(loss):.6g} name={name}")

    def _safe_metric(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return float("-inf")
        if not (x == x):  # NaN
            return float("-inf")
        if x == float("inf") or x == float("-inf"):
            return float("-inf")
        return x

    # ---- Robust theta handling ----
    def _to_cpu_theta(theta):
        # handles Tensor/Parameter; keeps CPU so replay is CPU-stable
        if isinstance(theta, torch.Tensor):
            return theta.detach().to("cpu")
        # very defensive fallback:
        if hasattr(theta, "detach"):
            return theta.detach().to("cpu")
        return theta

    def _theta_on_x_device(data_obj, theta):
        """Ensure theta is a Tensor on the same device as data_obj.x."""
        if not isinstance(theta, torch.Tensor):
            # should not happen in your setup, but keeps it safe
            theta = torch.as_tensor(theta)
        dev = data_obj.x.device if hasattr(data_obj, "x") and isinstance(data_obj.x, torch.Tensor) else torch.device("cpu")
        return theta.detach().to(dev)

    # ---- Keep an immutable CPU base for replay ----
    # (score() may call Data.to(cuda) internally; keep a CPU base to replay from)
    base_data_score = data_score.cpu()
    if not hasattr(base_data_score, "edge_dict_rel"):
        base_data_score.edge_dict_rel = {}

    # ---- Rebuild suffix state (CPU replay) ----
    def _rebuild_suffix_data(base_data, meta: List[int], theta_seq: List[Any]):
        """
        For meta=[r_k, ..., r_1] with theta_seq aligned [t_k, ..., t_1],
        rebuild suffix-state after applying [r_{k-1},...,r_1] in chronological order.
        """
        d = copy.copy(base_data)  # shallow ok; create_new_data clones anyway
        suffix_rels = list(reversed(meta[1:]))        # r_1 ... r_{k-1}
        suffix_thetas = list(reversed(theta_seq[1:]))

        for rel_i, theta_i in zip(suffix_rels, suffix_thetas):
            theta_i = _theta_on_x_device(d, theta_i)      # <<< FIX: align devices
            d = mps_main.create_new_data(d, theta_i, rel_i)
            if len(getattr(d, "target_ids", [])) == 0:
                break
        return d

    # Stores evaluated metapaths -> info (NO Data objects stored)
    partial_results: dict[str, dict] = {}
    all_metapaths: List[List[int]] = []

    emit(f"[MPS] rank_metric={metric_display} (used for beam pruning, ranking, and top-k combination checks)")

    # --------------------
    # Hop 1: score candidate relations connected to targets
    # --------------------
    actual_relations = mps_main.extract_relation_types(
        base_data_score.edge_index, base_data_score.edge_type, base_data_score.target_ids
    )
    emit(f"[MPS] #candidate relations (connected to targets): {len(actual_relations)}")

    scored_1 = []
    rel_pbar = tqdm(actual_relations, desc="Scoring candidate relations", unit="rel")
    for rel in rel_pbar:
        rel_id = int(rel)

        # Optional safety: pass a clone to avoid any in-place device moves of base_data_score
        tup = mps_main.score(rel, base_data_score.clone(), sampling_size=sampling_size)

        scored_1.append((int(tup[0]), float(tup[1]), _to_cpu_theta(tup[2])))
        rel_pbar.set_postfix(rel=rel_id, loss=f"{tup[1]:.4g}")

    best_rels, worst_rels = _top_bottom(scored_1, k_best=k_best_log, k_worst=k_worst_log)
    log_rel_list("[MPS][hop=1] best relations (lower score loss is better):", best_rels)
    log_rel_list("[MPS][hop=1] worst relations:", worst_rels)

    seeds = _topk_by_loss(scored_1, K)
    if not seeds:
        emit("[MPS] No relations found; returning empty metapath list.")
        return []

    emit(f"[MPS][hop=1] evaluating top-{len(seeds)} seeds with mpgnn() on split={eval_split}")
    metapaths_frontier: List[List[int]] = []

    seed_pbar = tqdm(seeds, desc="Evaluating 1-hop seeds", unit="seed")
    for rel, loss, theta in seed_pbar:
        meta = [int(rel)]
        theta_seq = [_to_cpu_theta(theta)]

        f1, auc = _eval_mpgnn_on_split(data_mpgnn, [meta], hidden_dim, split=eval_split)
        auc_roc = float(auc)
        chosen = auc_roc if metric_key == "auc_roc" else float(f1)

        seed_pbar.set_postfix(meta=str(meta), f1=f"{float(f1):.4f}", auc=f"{auc_roc:.4f}", chosen=f"{_safe_metric(chosen):.4f}")

        key = str(meta)
        partial_results[key] = {
            "meta": meta,
            "theta_seq": theta_seq,
            "score_loss": float(loss),
            "f1": float(f1),
            "auc": float(auc_roc),
            "auc_roc": float(auc_roc),
        }
        metapaths_frontier.append(meta)
        all_metapaths.append(meta)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_metric_for_meta(meta_list: List[int]) -> float:
        rec = partial_results[str(meta_list)]
        return _safe_metric(rec.get(metric_key, float("-inf")))

    # --------------------
    # Hop 2.. expansions
    # --------------------
    outer_pbar = tqdm(range(L_MAX), desc="Expanding metapaths", unit="iter")
    for iteration in outer_pbar:
        new_frontier: List[List[int]] = []
        created = 0

        mp_pbar = tqdm(
            metapaths_frontier,
            desc=f"Iter {iteration+1}/{L_MAX}: expand (top-{K} per frontier)",
            unit="mp",
            leave=False,
        )

        logged_frontier = 0
        for mp in mp_pbar:
            prev = partial_results[str(mp)]
            theta_seq_prev: List[Any] = prev["theta_seq"]

            data_prev_suffix = _rebuild_suffix_data(base_data_score, mp, theta_seq_prev)

            rel_first = mp[0]
            theta_first = theta_seq_prev[0]
            theta_first = _theta_on_x_device(data_prev_suffix, theta_first)   # <<< FIX: align devices

            data_new = mps_main.create_new_data(data_prev_suffix, theta_first, rel_first)

            if len(getattr(data_new, "target_ids", [])) == 0:
                continue

            actual_relations_new = mps_main.extract_relation_types(
                data_new.edge_index, data_new.edge_type, data_new.target_ids
            )
            if not actual_relations_new:
                continue

            scored_new = []
            for rel in actual_relations_new:
                tup = mps_main.score(rel, data_new, sampling_size=sampling_size)
                scored_new.append((int(tup[0]), float(tup[1]), _to_cpu_theta(tup[2])))

            hop = len(mp) + 1
            if logged_frontier < max_frontier_logs:
                best_new, worst_new = _top_bottom(scored_new, k_best=k_best_log, k_worst=k_worst_log)
                emit(f"[MPS][hop={hop}] extending mp={mp} | #candidates={len(scored_new)} | taking top-{min(K, len(scored_new))}")
                log_rel_list(f"[MPS][hop={hop}] best extensions for mp={mp}:", best_new)
                log_rel_list(f"[MPS][hop={hop}] worst extensions for mp={mp}:", worst_new)
                logged_frontier += 1

            topk_ext = _topk_by_loss(scored_new, K)

            for rel2, loss2, theta2 in topk_ext:
                new_meta = [int(rel2)] + mp[:]
                new_theta_seq = [_to_cpu_theta(theta2)] + theta_seq_prev[:]

                f1, auc = _eval_mpgnn_on_split(data_mpgnn, [new_meta], hidden_dim, split=eval_split)
                auc_roc = float(auc)

                key = str(new_meta)
                partial_results[key] = {
                    "meta": new_meta,
                    "theta_seq": new_theta_seq,
                    "score_loss": float(loss2),
                    "f1": float(f1),
                    "auc": float(auc_roc),
                    "auc_roc": float(auc_roc),
                }

                new_frontier.append(new_meta)
                all_metapaths.append(new_meta)
                created += 1

                chosen = auc_roc if metric_key == "auc_roc" else float(f1)
                mp_pbar.set_postfix(last=str(new_meta), f1=f"{float(f1):.4f}", auc=f"{auc_roc:.4f}", chosen=f"{_safe_metric(chosen):.4f}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del data_prev_suffix, data_new, scored_new

        # ---- Beam pruning (by rank_metric) ----
        if beam_width is not None and len(new_frontier) > beam_width:
            new_frontier = sorted(new_frontier, key=lambda m: _get_metric_for_meta(m), reverse=higher_is_better)[:beam_width]
            emit(f"[MPS] beam prune -> frontier={len(new_frontier)} (beam_width={beam_width}, by {metric_display})")

        metapaths_frontier = new_frontier
        outer_pbar.set_postfix(frontier=len(metapaths_frontier), total_candidates=len(all_metapaths), created=created)

        if not metapaths_frontier:
            emit("[MPS] frontier empty; stopping early.")
            break

    # --------------------
    # Final ranking (by rank_metric) + logging
    # --------------------
    seen = set()
    unique = []
    for mp in all_metapaths:
        k = str(mp)
        if k not in seen:
            unique.append(mp)
            seen.add(k)

    ranked = sorted(unique, key=lambda m: _get_metric_for_meta(m), reverse=higher_is_better)
    if not ranked:
        emit("[MPS] No metapaths survived ranking; returning empty.")
        return []

    emit(f"[MPS] ALL_METAPATHS_SORTED split={eval_split} sort_by={metric_display} total={len(ranked)}")
    for rank, mp in enumerate(ranked, start=1):
        metric_val = _safe_metric(_get_metric_for_meta(mp))
        emit(f"(path={mp}, hop={len(mp)}, sort_by={metric_display}) rank={rank} {metric_display}={metric_val:.6f}")

    best_single = ranked[0]
    # best_set = [best_single]

    # # --------------------
    # # Authors' “combine top-2/3” check (by rank_metric)
    # # --------------------
    # best_single_metric = _get_metric_for_meta(best_single)

    # if len(ranked) >= 2:
    #     second = ranked[1]
    #     f1_2, auc_2 = _eval_mpgnn_on_split(data_mpgnn, [best_single, second], hidden_dim, split=eval_split)
    #     combo_metric_2 = _safe_metric(float(auc_2) if metric_key == "auc_roc" else float(f1_2))

    #     if combo_metric_2 > best_single_metric:
    #         best_set = [best_single, second]

    #         if len(ranked) >= 3:
    #             third = ranked[2]
    #             f1_3, auc_3 = _eval_mpgnn_on_split(data_mpgnn, [best_single, second, third], hidden_dim, split=eval_split)
    #             combo_metric_3 = _safe_metric(float(auc_3) if metric_key == "auc_roc" else float(f1_3))

    #             if combo_metric_3 > combo_metric_2:
    #                 best_set = [best_single, second, third]

    # emit(f"[MPS] Selected metapaths: {best_set} (by {metric_display})")
    return best_single
