import importlib
import importlib.util
import os
import struct
import sys
from typing import Any, Callable, cast

import numpy as np

try:
    from tqdm import tqdm as _tqdm_impl
except Exception:
    _tqdm_impl = None


class _TqdmFallback:
    def __init__(self, iterable: Any, **_kwargs: Any) -> None:
        self._iterable = iterable

    def __iter__(self) -> Any:
        return iter(self._iterable)

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _progress(iterable: Any, **kwargs: Any) -> Any:
    if _tqdm_impl is None:
        return _TqdmFallback(iterable, **kwargs)
    tqdm_impl = cast(Any, _tqdm_impl)
    return tqdm_impl(iterable, **kwargs)


def _progress_write(message: str) -> None:
    if _tqdm_impl is None:
        print(message)
        return
    tqdm_impl = cast(Any, _tqdm_impl)
    tqdm_impl.write(message)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _local_store_dir() -> str | None:
    return os.environ.get("PARASITIC_LOCAL_DB")


def _prepare_query_level_data_local(
    dataset_name: str,
    db_dir: str,
    healthy_top_k: int,
    error_buffer: int,
) -> dict[str, Any]:
    from local_store import LocalStore

    alias = _canonical_mteb_dataset_name(dataset_name)
    store = LocalStore(db_dir)
    if not store.dataset_exists(alias):
        raise ValueError(
            f"Local dataset alias {alias!r} not found under {db_dir!r}. "
            f"Run 0_math_approach/direct_feed_local.py --dataset ... first."
        )

    query_embs = store.load_embeddings(alias, "queries")
    corpus_embs = store.load_embeddings(alias, "corpus")
    if not query_embs or not corpus_embs:
        raise ValueError(f"Missing embeddings for dataset {alias}")

    qrels: dict[str, set[str]] = {}
    for query_id, corpus_id, score in store.load_qrels(alias, min_score=1):
        if query_id in query_embs and corpus_id in corpus_embs:
            qrels.setdefault(query_id, set()).add(corpus_id)

    return _query_level_data_from_embeddings(query_embs, corpus_embs, qrels, alias)


def decode_vector(raw: bytes | None) -> np.ndarray | None:
    if not raw:
        return None
    values = struct.unpack(f"{len(raw) // 4}f", raw)
    if len(values) <= 1:
        return None
    return np.array(values[1:], dtype=np.float32)


def classify_query_health(ranked_ids: list[str], relevant_ids: set[str]) -> str:
    if not ranked_ids:
        return "error"
    return "healthy" if ranked_ids[0] in relevant_ids else "error"


def select_healthy_confusers(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> list[str]:
    confusers: list[str] = []
    for doc_id in ranked_ids:
        if doc_id not in relevant_ids:
            confusers.append(doc_id)
            if len(confusers) >= k:
                break
    return confusers


def select_error_confusers(
    ranked_ids: list[str],
    relevant_ids: set[str],
    best_relevant_doc_id: str | None,
    buffer_below: int = 10,
) -> list[str]:
    if best_relevant_doc_id is None:
        return [doc_id for doc_id in ranked_ids if doc_id not in relevant_ids][: max(1, buffer_below)]

    blockers: list[str] = []
    for doc_id in ranked_ids:
        if doc_id == best_relevant_doc_id:
            break
        if doc_id not in relevant_ids:
            blockers.append(doc_id)

    below_buffer: list[str] = []
    seen_best = False
    for doc_id in ranked_ids:
        if doc_id == best_relevant_doc_id:
            seen_best = True
            continue
        if seen_best and doc_id not in relevant_ids:
            below_buffer.append(doc_id)
            if len(below_buffer) >= buffer_below:
                break

    return blockers + below_buffer


def split_queries_by_query_level(
    items: list[dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_query_ids, val_query_ids = split_query_ids_by_health(items, val_fraction=val_fraction, seed=seed)
    train_items = filter_query_data_by_ids(items, train_query_ids)
    val_items = filter_query_data_by_ids(items, val_query_ids)
    return train_items, val_items


def split_query_ids_by_health(
    items: list[dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[set[str], set[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        query_id = str(item["query_id"])
        grouped.setdefault(query_id, []).append(item)

    rng = np.random.RandomState(seed)
    healthy_query_ids: list[str] = []
    error_query_ids: list[str] = []
    for query_id, query_items in grouped.items():
        if bool(query_items[0].get("is_healthy")):
            healthy_query_ids.append(query_id)
        else:
            error_query_ids.append(query_id)

    rng.shuffle(healthy_query_ids)
    rng.shuffle(error_query_ids)

    split_healthy = int(len(healthy_query_ids) * (1.0 - val_fraction))
    split_error = int(len(error_query_ids) * (1.0 - val_fraction))
    train_query_ids = set(healthy_query_ids[:split_healthy] + error_query_ids[:split_error])

    return train_query_ids, set(grouped.keys()) - train_query_ids


def split_query_ids_three_way(
    items: list[dict[str, Any]],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Split query ids into disjoint train/validation/test sets, stratified by health.

    The three-way split isolates a test set that must never be consulted by the
    acceptance gate or by any configuration-selection rule. Train drives the
    surrogate optimization, validation drives the gate and candidate selection,
    and test is reserved for the final one-pass evaluation.

    Args:
        items: Query-level rows carrying ``query_id`` and ``is_healthy``.
        val_fraction: Fraction of queries assigned to validation.
        test_fraction: Fraction of queries assigned to test. ``0.0`` reproduces
            the legacy two-way behaviour with an empty test set.

    Returns:
        ``(train_query_ids, val_query_ids, test_query_ids)`` as disjoint sets
        covering every query id in ``items``.

    Raises:
        ValueError: If fractions are negative or sum to 1.0 or more.
    """
    val_fraction = float(val_fraction)
    test_fraction = float(test_fraction)
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be >= 0")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0 (train split would be empty)")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        query_id = str(item["query_id"])
        grouped.setdefault(query_id, []).append(item)

    rng = np.random.RandomState(seed)
    healthy_query_ids: list[str] = []
    error_query_ids: list[str] = []
    for query_id, query_items in grouped.items():
        if bool(query_items[0].get("is_healthy")):
            healthy_query_ids.append(query_id)
        else:
            error_query_ids.append(query_id)

    rng.shuffle(healthy_query_ids)
    rng.shuffle(error_query_ids)

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()
    for stratum in (healthy_query_ids, error_query_ids):
        n = len(stratum)
        n_train = int(n * (1.0 - val_fraction - test_fraction))
        n_test = int(n * test_fraction)
        train_ids.update(stratum[:n_train])
        test_ids.update(stratum[n_train:n_train + n_test])
        val_ids.update(stratum[n_train + n_test:])

    return train_ids, val_ids, test_ids


def filter_query_data_by_ids(items: list[dict[str, Any]], query_ids: set[str]) -> list[dict[str, Any]]:
    return [item for item in items if str(item.get("query_id")) in query_ids]



def _count_query_health(items: list[dict[str, Any]]) -> dict[str, int]:
    total = len(items)
    healthy = sum(1 for item in items if bool(item.get("is_healthy")))
    return {
        "total": int(total),
        "healthy": int(healthy),
        "error": int(total - healthy),
    }


def _format_query_health_summary(items: list[dict[str, Any]]) -> str:
    counts = _count_query_health(items)
    total = counts["total"]
    if total == 0:
        return "0 queries (0 error, 0 healthy)"
    error_ratio = counts["error"] / total
    healthy_ratio = counts["healthy"] / total
    return (
        f"{total} queries ({counts['error']} error, {counts['healthy']} healthy | "
        f"{error_ratio:.1%} error, {healthy_ratio:.1%} healthy)"
    )


def _print_query_split_summary(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    header: str = "Query-level train/validation split (stratified by health label):",
) -> None:
    _progress_write(header)
    _progress_write(f"  Train      : {_format_query_health_summary(train_items)}")
    _progress_write(f"  Validation : {_format_query_health_summary(val_items)}")


def _should_run_validation(epoch: int, total_epochs: int, eval_every: int) -> bool:
    cadence = max(1, int(eval_every))
    return epoch % cadence == 0 or epoch == total_epochs


def _metric_value(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return float(default)


def _format_exact_metric_summary(metrics: dict[str, Any]) -> str:
    return (
        f"degraded={int(_metric_value(metrics, 'healthy_degraded_count', default=0))}, "
        f"rank_gain={int(_metric_value(metrics, 'error_rank_gain_total', default=0))}, "
        f"improved_errors={int(_metric_value(metrics, 'improved_error_count', default=0))}, "
        f"top1={int(_metric_value(metrics, 'error_top1_fixed_count', default=0))}, "
        f"top10={int(_metric_value(metrics, 'error_top10_gain_count', default=0))}, "
        f"ndcg@10={_metric_value(metrics, 'ndcg@10_after', 'ndcg@10', default=0.0):.4f}, "
        f"MRR={_metric_value(metrics, 'MRR_after', 'MRR', default=0.0):.4f}"
    )


def _explain_exact_acceptance_change(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None = None,
) -> str:
    candidate_degraded = int(candidate_metrics.get("healthy_degraded_count", 0))
    if candidate_degraded > 0:
        return f"healthy_degraded_count={candidate_degraded} > 0"

    if incumbent_metrics is None:
        return "no incumbent yet and candidate preserves healthy queries"

    incumbent_degraded = int(incumbent_metrics.get("healthy_degraded_count", 0))
    if incumbent_degraded > 0:
        return f"candidate restores zero degradation from incumbent healthy_degraded_count={incumbent_degraded}"

    ordered_keys = [
        "error_rank_gain_total",
        "error_top1_fixed_count",
        "error_top10_gain_count",
        "ndcg@10_after",
        "MRR_after",
    ]
    for key in ordered_keys:
        candidate_value = _metric_value(candidate_metrics, key, default=0.0)
        incumbent_value = _metric_value(incumbent_metrics, key, default=0.0)
        if candidate_value != incumbent_value:
            if key.endswith("_after"):
                return f"{key}: {incumbent_value:.4f} -> {candidate_value:.4f}"
            return f"{key}: {int(incumbent_value)} -> {int(candidate_value)}"

    return "exact tie on acceptance metrics"


def accept_candidate_by_top1_vs_degraded(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None = None,
) -> bool:
    """Adaptive gate: accept if error_top1_fixed_count > healthy_degraded_count.

    If incumbent_metrics is provided, compare net gains (top1_fixed - degraded) and
    accept only if candidate net > incumbent net.
    """
    candidate_degraded = int(candidate_metrics.get("healthy_degraded_count", 0))
    candidate_top1_fixed = int(candidate_metrics.get("error_top1_fixed_count", 0))

    if candidate_top1_fixed <= candidate_degraded:
        return False

    if incumbent_metrics is None:
        return True

    incumbent_degraded = int(incumbent_metrics.get("healthy_degraded_count", 0))
    incumbent_top1_fixed = int(incumbent_metrics.get("error_top1_fixed_count", 0))

    candidate_net = candidate_top1_fixed - candidate_degraded
    incumbent_net = incumbent_top1_fixed - incumbent_degraded
    return candidate_net > incumbent_net


def _explain_top1_vs_degraded_change(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None = None,
) -> str:
    candidate_degraded = int(candidate_metrics.get("healthy_degraded_count", 0))
    candidate_top1_fixed = int(candidate_metrics.get("error_top1_fixed_count", 0))
    candidate_net = candidate_top1_fixed - candidate_degraded

    if incumbent_metrics is None:
        if candidate_net > 0:
            return (
                f"no incumbent; top1_fixed={candidate_top1_fixed} > degraded={candidate_degraded} "
                f"(net={candidate_net})"
            )
        return (
            f"no incumbent; top1_fixed={candidate_top1_fixed} <= degraded={candidate_degraded} "
            f"(net={candidate_net})"
        )

    incumbent_degraded = int(incumbent_metrics.get("healthy_degraded_count", 0))
    incumbent_top1_fixed = int(incumbent_metrics.get("error_top1_fixed_count", 0))
    incumbent_net = incumbent_top1_fixed - incumbent_degraded

    if candidate_net > incumbent_net:
        return (
            f"net top1 gain: {incumbent_net} -> {candidate_net} "
            f"(top1_fixed={candidate_top1_fixed}, degraded={candidate_degraded})"
        )
    return (
        f"net top1 gain {candidate_net} <= incumbent {incumbent_net} "
        f"(top1_fixed={candidate_top1_fixed}, degraded={candidate_degraded})"
    )


def _build_candidate_global(metrics_full: dict[str, Any]) -> dict[str, Any]:
    candidate_global = dict(metrics_full.get("global", {}))
    retrieval = metrics_full.get("retrieval", {})
    candidate_global.update(
        {
            "ndcg@10_after": retrieval.get("after", {}).get("ndcg@10", 0.0),
            "MRR_after": retrieval.get("after", {}).get("MRR", 0.0),
        }
    )
    return candidate_global


def _canonical_mteb_dataset_name(dataset_name: str) -> str:
    """Return the exact ``RD_MTEB_DATASETS.DATASET_NAME`` used by MTEB rows."""
    normalized = str(dataset_name or "").strip()
    if normalized.lower().startswith("mteb/"):
        return normalized.split("/", 1)[1]
    return normalized


def _query_level_data_from_embeddings(
    query_embs: dict[str, np.ndarray],
    corpus_embs: dict[str, np.ndarray],
    qrels: dict[str, set[str]],
    dataset_id: Any,
    healthy_top_k: int = 10,
    error_buffer: int = 10,
) -> dict[str, Any]:
    query_ids = list(qrels.keys())
    corpus_ids = list(corpus_embs.keys())

    if not query_ids:
        return {
            "query_data": [],
            "query_embs": query_embs,
            "corpus_embs": corpus_embs,
            "dataset_id": dataset_id,
        }

    query_matrix = np.stack([query_embs[query_id] for query_id in query_ids])
    corpus_matrix = np.stack([corpus_embs[corpus_id] for corpus_id in corpus_ids])
    query_norm = query_matrix / (np.linalg.norm(query_matrix, axis=1, keepdims=True) + 1e-10)
    corpus_norm = corpus_matrix / (np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10)
    similarity_matrix = query_norm @ corpus_norm.T

    query_data: list[dict[str, Any]] = []
    for query_index, query_id in enumerate(query_ids):
        scores = similarity_matrix[query_index]
        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [corpus_ids[index] for index in ranked_indices]

        relevant_set = qrels[query_id]
        best_relevant_doc_id: str | None = None
        best_relevant_rank = float("inf")
        for rank_index, doc_id in enumerate(ranked_doc_ids):
            if doc_id in relevant_set:
                best_relevant_doc_id = doc_id
                best_relevant_rank = rank_index + 1
                break

        health_label = classify_query_health(ranked_doc_ids, relevant_set)
        is_healthy = health_label == "healthy"

        healthy_confuser_ids: list[str] = []
        error_confuser_ids: list[str] = []
        if is_healthy:
            healthy_confuser_ids = select_healthy_confusers(
                ranked_ids=ranked_doc_ids,
                relevant_ids=relevant_set,
                k=healthy_top_k,
            )
            confuser_ids = healthy_confuser_ids
        else:
            error_confuser_ids = select_error_confusers(
                ranked_ids=ranked_doc_ids,
                relevant_ids=relevant_set,
                best_relevant_doc_id=best_relevant_doc_id,
                buffer_below=error_buffer,
            )
            confuser_ids = error_confuser_ids

        query_data.append(
            {
                "query_id": query_id,
                "R_q": sorted(relevant_set),
                "relevant_set_ids": sorted(relevant_set),
                "top1_doc_id": ranked_doc_ids[0] if ranked_doc_ids else None,
                "health_label": health_label,
                "is_healthy": is_healthy,
                "best_relevant_doc_id": best_relevant_doc_id,
                "best_rel_id": best_relevant_doc_id,
                "best_relevant_rank": best_relevant_rank,
                "initial_rank": best_relevant_rank,
                "healthy_confuser_ids": healthy_confuser_ids,
                "error_confuser_ids": error_confuser_ids,
                "confuser_ids": confuser_ids,
            },
        )

    return {
        "query_data": query_data,
        "query_embs": query_embs,
        "corpus_embs": corpus_embs,
        "dataset_id": dataset_id,
    }


def prepare_query_level_data(
    dataset_name: str,
    healthy_top_k: int = 10,
    error_buffer: int = 10,
) -> dict[str, Any]:
    local_db_dir = _local_store_dir()
    if not local_db_dir:
        raise RuntimeError("PARASITIC_LOCAL_DB must be set to run in local mode.")
    return _prepare_query_level_data_local(
        dataset_name, local_db_dir, healthy_top_k, error_buffer
    )


def split_query_data(
    query_data: list[dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return split_queries_by_query_level(query_data, val_fraction=val_fraction, seed=seed)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """Stable cosine similarity between two 1-D numpy arrays."""
    if a is None or b is None:
        return float("-inf")
    num = float(np.dot(a, b))
    denom = float((np.linalg.norm(a) * np.linalg.norm(b)) + eps)
    return num / denom


def _logsumexp(xs: np.ndarray) -> float:
    """Numerically stable log-sum-exp for 1-D array."""
    if len(xs) == 0:
        return float("-inf")
    m = float(np.max(xs))
    return m + float(np.log(np.sum(np.exp(xs - m))))


def s_rel(query_id: str, relevant_ids: list[str], query_embs: dict[str, np.ndarray], corpus_embs: dict[str, np.ndarray]) -> float:
    q_vec = query_embs.get(query_id)
    if q_vec is None or not relevant_ids:
        return float("-inf")
    scores = []
    for r in relevant_ids:
        r_vec = corpus_embs.get(r)
        if r_vec is None:
            continue
        scores.append(_cosine(q_vec, r_vec))
    if not scores:
        return float("-inf")
    return float(np.max(np.array(scores, dtype=np.float64)))


def healthy_loss_for_query(
    query: dict[str, Any],
    query_embs: dict[str, np.ndarray],
    corpus_embs: dict[str, np.ndarray],
    tau_h: float = 0.05,
) -> float:
    qid = str(query.get("query_id"))
    q_vec = query_embs.get(qid)
    if q_vec is None:
        return 0.0

    relevant_ids = query.get("relevant_set_ids", [])
    s_pos = s_rel(qid, relevant_ids, query_embs, corpus_embs)

    confuser_ids = query.get("healthy_confuser_ids") or query.get("confuser_ids") or []
    if not confuser_ids:
        return 0.0

    terms = []
    for cid in confuser_ids:
        cvec = corpus_embs.get(cid)
        if cvec is None:
            continue
        s_n = _cosine(q_vec, cvec)
        terms.append((s_n - s_pos) / float(tau_h))

    if not terms:
        return 0.0

    return float(_logsumexp(np.array(terms, dtype=np.float64)))


def compute_healthy_loss(
    query_data: list[dict[str, Any]],
    query_embs: dict[str, np.ndarray],
    corpus_embs: dict[str, np.ndarray],
    tau_h: float = 0.05,
    weighting: str = "per_query",
    preserve_healthy_weight: float = 1.0,
) -> float:
    healthy_queries = [q for q in query_data if q.get("is_healthy")]
    if not healthy_queries:
        return 0.0

    per_q_losses = []
    confuser_counts = []
    for q in healthy_queries:
        lq = healthy_loss_for_query(q, query_embs, corpus_embs, tau_h=tau_h)
        per_q_losses.append(float(lq))
        confuser_counts.append(len(q.get("healthy_confuser_ids") or q.get("confuser_ids") or []))

    per_q_losses = np.array(per_q_losses, dtype=np.float64)
    confuser_counts = np.array(confuser_counts, dtype=np.float64)

    if weighting == "per_confuser":
        total_confusers = float(np.sum(confuser_counts))
        if total_confusers == 0:
            return float(np.mean(per_q_losses))
        return float(np.sum(per_q_losses * confuser_counts) / total_confusers)
    if weighting == "preserve_healthy":
        base = float(np.mean(per_q_losses))
        # preserve_healthy_mode: multiply mean per-query loss by provided scalar
        return float(preserve_healthy_weight * base)

    return float(np.mean(per_q_losses))


def healthy_loss_for_query_torch(
    query: dict[str, Any],
    query_embs: dict[str, Any],
    corpus_embs: dict[str, Any],
    tau_h: float = 0.05,
) -> Any:
    torch = _torch_module()
    qid = str(query.get("query_id"))
    q_vec_raw = query_embs.get(qid)
    if q_vec_raw is None:
        return torch.tensor(0.0, dtype=torch.float32)

    q_vec = _to_torch_vec(q_vec_raw)
    relevant_ids = query.get("relevant_set_ids", [])
    s_pos = s_rel_torch(qid, relevant_ids, query_embs, corpus_embs)
    if s_pos is None:
        return torch.tensor(0.0, dtype=q_vec.dtype, device=q_vec.device)

    confuser_ids = query.get("healthy_confuser_ids") or query.get("confuser_ids") or []
    if not confuser_ids:
        return torch.tensor(0.0, dtype=q_vec.dtype, device=q_vec.device)

    tau_tensor = torch.as_tensor(float(tau_h), dtype=q_vec.dtype, device=q_vec.device)
    tau_safe = torch.where(tau_tensor == 0, torch.full_like(tau_tensor, 1e-12), tau_tensor)

    terms: list[Any] = []
    for cid in confuser_ids:
        cvec_raw = corpus_embs.get(cid)
        if cvec_raw is None:
            continue
        s_n = _cosine_torch(q_vec, _to_torch_vec(cvec_raw, device=q_vec.device))
        terms.append((s_n - s_pos) / tau_safe)

    if not terms:
        return torch.tensor(0.0, dtype=q_vec.dtype, device=q_vec.device)
    stacked = torch.stack(terms)
    m = torch.max(stacked)
    return m + torch.log(torch.sum(torch.exp(stacked - m)))


def compute_healthy_loss_torch(
    query_data: list[dict[str, Any]],
    query_embs: dict[str, Any],
    corpus_embs: dict[str, Any],
    tau_h: float = 0.05,
    weighting: str = "per_query",
    preserve_healthy_weight: float = 1.0,
) -> Any:
    torch = _torch_module()
    healthy_queries = [q for q in query_data if q.get("is_healthy")]
    if not healthy_queries:
        return torch.tensor(0.0, dtype=torch.float32)

    per_q_losses = [healthy_loss_for_query_torch(q, query_embs, corpus_embs, tau_h=tau_h) for q in healthy_queries]
    # Convert list[Any] to tensor safely
    stacked = torch.stack(per_q_losses)

    s = str(weighting or "").strip().lower()
    if s == "per_confuser":
        # Need per-query confuser counts to weight; compute from query rows
        confuser_counts = [float(len(q.get("healthy_confuser_ids") or q.get("confuser_ids") or [])) for q in healthy_queries]
        total_confusers = float(sum(confuser_counts))
        if total_confusers == 0:
            return stacked.mean()
        weights = torch.as_tensor(confuser_counts, dtype=stacked.dtype, device=stacked.device)
        return float((stacked * weights).sum() / weights.sum())
    if s == "preserve_healthy":
        base = stacked.mean()
        return base * float(preserve_healthy_weight)

    # default: per_query mean
    return stacked.mean()


def _torch_module() -> Any:
    return importlib.import_module("torch")


def _try_torch_module() -> Any | None:
    try:
        return _torch_module()
    except ModuleNotFoundError:
        return None


def _to_torch_vec(vec: Any, *, device: Any = None) -> Any:
    torch = _torch_module()
    if isinstance(vec, torch.Tensor):
        if device is not None and vec.device != device:
            return vec.to(device)
        return vec
    return torch.as_tensor(vec, dtype=torch.float32, device=device)


def _cosine_torch(a: Any, b: Any, eps: float = 1e-10) -> Any:
    torch = _torch_module()
    a_t = _to_torch_vec(a)
    b_t = _to_torch_vec(b, device=a_t.device)
    return torch.dot(a_t, b_t) / ((torch.linalg.norm(a_t) * torch.linalg.norm(b_t)) + float(eps))


def s_rel_torch(
    query_id: str,
    relevant_ids: list[str],
    query_embs: dict[str, Any],
    corpus_embs: dict[str, Any],
) -> Any:
    torch = _torch_module()
    q_vec_raw = query_embs.get(query_id)
    if q_vec_raw is None or not relevant_ids:
        return None
    q_vec = _to_torch_vec(q_vec_raw)

    scores: list[Any] = []
    for rid in relevant_ids:
        r_vec_raw = corpus_embs.get(rid)
        if r_vec_raw is None:
            continue
        scores.append(_cosine_torch(q_vec, _to_torch_vec(r_vec_raw, device=q_vec.device)))
    if not scores:
        return None
    return torch.max(torch.stack(scores))


def error_loss_for_query_torch(
    query: dict[str, Any],
    query_embs: dict[str, Any],
    corpus_embs: dict[str, Any],
    tau_e: float = 0.05,
) -> Any:
    torch = _torch_module()
    qid = str(query.get("query_id"))
    q_vec_raw = query_embs.get(qid)
    if q_vec_raw is None:
        return torch.tensor(0.0, dtype=torch.float32)

    q_vec = _to_torch_vec(q_vec_raw)
    relevant_ids = query.get("relevant_set_ids", [])
    s_pos = s_rel_torch(qid, relevant_ids, query_embs, corpus_embs)
    if s_pos is None:
        return torch.zeros((), dtype=q_vec.dtype, device=q_vec.device)

    confuser_ids = query.get("error_confuser_ids") or query.get("confuser_ids") or []
    if not confuser_ids:
        return torch.zeros((), dtype=q_vec.dtype, device=q_vec.device)

    tau_tensor = torch.as_tensor(float(tau_e), dtype=q_vec.dtype, device=q_vec.device)
    tau_safe = torch.where(tau_tensor == 0, torch.full_like(tau_tensor, 1e-12), tau_tensor)

    terms: list[Any] = []
    for cid in confuser_ids:
        cvec_raw = corpus_embs.get(cid)
        if cvec_raw is None:
            continue
        s_n = _cosine_torch(q_vec, _to_torch_vec(cvec_raw, device=q_vec.device))
        terms.append(torch.sigmoid((s_n - s_pos) / tau_safe))

    if not terms:
        return torch.zeros((), dtype=q_vec.dtype, device=q_vec.device)
    return torch.stack(terms).sum()


def compute_error_loss_torch(
    query_data: list[dict[str, Any]],
    query_embs: dict[str, Any],
    corpus_embs: dict[str, Any],
    tau_e: float = 0.05,
) -> Any:
    torch = _torch_module()
    error_queries = [q for q in query_data if not q.get("is_healthy")]
    if not error_queries:
        return torch.tensor(0.0, dtype=torch.float32)

    per_q_losses = [error_loss_for_query_torch(q, query_embs, corpus_embs, tau_e=tau_e) for q in error_queries]
    return torch.stack(per_q_losses).mean()


def error_loss_for_query(
    query: dict[str, Any],
    query_embs: dict[str, Any],
    corpus_embs: dict[str, Any],
    tau_e: float = 0.05,
) -> float:
    torch = _try_torch_module()
    if torch is not None:
        return float(error_loss_for_query_torch(query, query_embs, corpus_embs, tau_e=tau_e).detach().cpu().item())

    qid = str(query.get("query_id"))
    q_vec = query_embs.get(qid)
    if q_vec is None:
        return 0.0

    relevant_ids = query.get("relevant_set_ids", [])
    s_pos = s_rel(qid, relevant_ids, query_embs, corpus_embs)
    if not np.isfinite(s_pos):
        return 0.0

    confuser_ids = query.get("error_confuser_ids") or query.get("confuser_ids") or []
    if not confuser_ids:
        return 0.0

    denom = float(tau_e) if float(tau_e) != 0.0 else 1e-12
    loss = 0.0
    for cid in confuser_ids:
        cvec = corpus_embs.get(cid)
        if cvec is None:
            continue
        s_n = _cosine(q_vec, cvec)
        loss += float(1.0 / (1.0 + np.exp(-((s_n - s_pos) / denom))))
    return float(loss)


def compute_error_loss(
    query_data: list[dict[str, Any]],
    query_embs: dict[str, np.ndarray],
    corpus_embs: dict[str, np.ndarray],
    tau_e: float = 0.05,
) -> float:
    torch = _try_torch_module()
    if torch is not None:
        return float(compute_error_loss_torch(query_data, query_embs, corpus_embs, tau_e=tau_e).detach().cpu().item())

    error_queries = [q for q in query_data if not q.get("is_healthy")]
    if not error_queries:
        return 0.0
    per_q_losses = [error_loss_for_query(q, query_embs, corpus_embs, tau_e=tau_e) for q in error_queries]
    return float(np.mean(np.array(per_q_losses, dtype=np.float64)))


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)


def _compute_full_rankings(
    query_ids: list[str],
    query_embs: dict[str, np.ndarray],
    corpus_ids: list[str],
    corpus_embs: dict[str, np.ndarray],
) -> dict[str, list[str]]:
    if not query_ids or not corpus_ids:
        return {query_id: [] for query_id in query_ids}

    query_matrix = np.stack([query_embs[query_id] for query_id in query_ids])
    corpus_matrix = np.stack([corpus_embs[doc_id] for doc_id in corpus_ids])
    similarities = _normalize_matrix(query_matrix) @ _normalize_matrix(corpus_matrix).T

    rankings: dict[str, list[str]] = {}
    for query_index, query_id in enumerate(query_ids):
        ranked_indices = np.argsort(similarities[query_index])[::-1]
        rankings[query_id] = [corpus_ids[doc_idx] for doc_idx in ranked_indices]
    return rankings


def _build_full_run(
    query_ids: list[str],
    rankings: dict[str, list[str]],
    query_embs: dict[str, np.ndarray],
    corpus_ids: list[str],
    corpus_embs: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    if not query_ids or not corpus_ids:
        return {query_id: {} for query_id in query_ids}

    q_matrix = np.stack([query_embs[qid] for qid in query_ids])
    c_matrix = np.stack([corpus_embs[cid] for cid in corpus_ids])
    sims = _normalize_matrix(q_matrix) @ _normalize_matrix(c_matrix).T
    index_by_doc = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}

    run: dict[str, dict[str, float]] = {}
    for q_idx, query_id in enumerate(query_ids):
        ranked_doc_ids = rankings.get(query_id, [])
        run[query_id] = {
            doc_id: float(sims[q_idx, index_by_doc[doc_id]])
            for doc_id in ranked_doc_ids
            if doc_id in index_by_doc
        }
    return run


def _best_relevant_rank(
    ranked_doc_ids: list[str],
    relevant_ids: set[str],
) -> int:
    for idx, doc_id in enumerate(ranked_doc_ids):
        if doc_id in relevant_ids:
            return idx + 1
    return len(ranked_doc_ids) + 1


def _blocker_count_before_best_relevant(
    ranked_doc_ids: list[str],
    relevant_ids: set[str],
) -> int:
    blockers = 0
    for doc_id in ranked_doc_ids:
        if doc_id in relevant_ids:
            return blockers
        blockers += 1
    return blockers


def _dcg_at_k(relevances: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = np.array(relevances[:k], dtype=np.float64)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    gains = (2.0**rel) - 1.0
    return float(np.sum(gains * discounts))


def _ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    rel_sorted = sorted((int(v) for v in qrels.values() if int(v) > 0), reverse=True)
    if not rel_sorted:
        return 0.0
    observed = [int(qrels.get(doc_id, 0)) for doc_id in ranked_doc_ids]
    dcg = _dcg_at_k(observed, k)
    idcg = _dcg_at_k(rel_sorted, k)
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _recall_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant_ids = {doc_id for doc_id, rel in qrels.items() if int(rel) > 0}
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_ids)
    return float(hits / len(relevant_ids))


def _average_precision_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant_ids = {doc_id for doc_id, rel in qrels.items() if int(rel) > 0}
    if not relevant_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank_idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank_idx

    denom = min(len(relevant_ids), k)
    if denom == 0:
        return 0.0
    return float(precision_sum / denom)


def _reciprocal_rank(ranked_doc_ids: list[str], qrels: dict[str, int]) -> float:
    relevant_ids = {doc_id for doc_id, rel in qrels.items() if int(rel) > 0}
    for rank_idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_ids:
            return float(1.0 / rank_idx)
    return 0.0


def _evaluate_retrieval_metrics_offline(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    if not qrels:
        return {
            "ndcg@10": 0.0,
            "recall@10": 0.0,
            "map@10": 0.0,
            "MRR": 0.0,
        }

    query_ids = list(qrels.keys())
    ndcgs = []
    recalls = []
    aps = []
    rrs = []
    for query_id in query_ids:
        ranked = rankings.get(query_id, [])
        qrel = qrels.get(query_id, {})
        ndcgs.append(_ndcg_at_k(ranked, qrel, k=k))
        recalls.append(_recall_at_k(ranked, qrel, k=k))
        aps.append(_average_precision_at_k(ranked, qrel, k=k))
        rrs.append(_reciprocal_rank(ranked, qrel))

    return {
        "ndcg@10": float(np.mean(np.array(ndcgs, dtype=np.float64))),
        "recall@10": float(np.mean(np.array(recalls, dtype=np.float64))),
        "map@10": float(np.mean(np.array(aps, dtype=np.float64))),
        "MRR": float(np.mean(np.array(rrs, dtype=np.float64))),
    }


def _evaluate_retrieval_metrics(
    rankings_before: dict[str, list[str]],
    rankings_after: dict[str, list[str]],
    query_ids: list[str],
    query_embs: dict[str, np.ndarray],
    corpus_ids: list[str],
    corpus_before: dict[str, np.ndarray],
    corpus_after: dict[str, np.ndarray],
    qrels: dict[str, dict[str, int]],
    query_embs_after: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    if query_embs_after is None:
        query_embs_after = query_embs
    try:
        evaluation_module = importlib.import_module("mteb_benchmark.evaluation")
        EvaluationModule = getattr(evaluation_module, "EvaluationModule")
        evaluator = EvaluationModule(qrels)

        run_before = _build_full_run(
            query_ids=query_ids,
            rankings=rankings_before,
            query_embs=query_embs,
            corpus_ids=corpus_ids,
            corpus_embs=corpus_before,
        )
        run_after = _build_full_run(
            query_ids=query_ids,
            rankings=rankings_after,
            query_embs=query_embs_after,
            corpus_ids=corpus_ids,
            corpus_embs=corpus_after,
        )
        before = evaluator.evaluate(run_before, metrics=["ndcg@10", "recall@10", "map@10"])
        after = evaluator.evaluate(run_after, metrics=["ndcg@10", "recall@10", "map@10"])

        offline_before = _evaluate_retrieval_metrics_offline(rankings_before, qrels, k=10)
        offline_after = _evaluate_retrieval_metrics_offline(rankings_after, qrels, k=10)
        before["MRR"] = offline_before["MRR"]
        after["MRR"] = offline_after["MRR"]
    except Exception:
        before = _evaluate_retrieval_metrics_offline(rankings_before, qrels, k=10)
        after = _evaluate_retrieval_metrics_offline(rankings_after, qrels, k=10)

    improvement = {metric: float(after.get(metric, 0.0) - before.get(metric, 0.0)) for metric in before}
    return {
        "before": before,
        "after": after,
        "improvement": improvement,
    }


def evaluate_exact_full_corpus_before_after(
    validation_queries: list[dict[str, Any]],
    query_embs: dict[str, np.ndarray],
    corpus_before: dict[str, np.ndarray],
    corpus_after: dict[str, np.ndarray],
    query_embs_after: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if query_embs_after is None:
        query_embs_after = query_embs
    if not validation_queries:
        empty_metrics = {
            "ndcg@10": 0.0,
            "recall@10": 0.0,
            "map@10": 0.0,
            "MRR": 0.0,
        }
        return {
            "query_metrics": {},
            "global": {
                "healthy_preserved_count": 0,
                "healthy_degraded_count": 0,
                "improved_error_count": 0,
                "error_rank_gain_total": 0,
                "error_top1_fixed_count": 0,
                "error_top10_gain_count": 0,
                "top10_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
                "top5_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
                "top3_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
                "top1_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
                "query_count": 0,
            },
            "retrieval": {
                "before": dict(empty_metrics),
                "after": dict(empty_metrics),
                "improvement": {metric: 0.0 for metric in empty_metrics},
            },
        }

    query_ids = [str(query["query_id"]) for query in validation_queries if str(query.get("query_id")) in query_embs]
    corpus_ids = list(corpus_before.keys())
    qrels = {
        str(query["query_id"]): {str(doc_id): 1 for doc_id in (query.get("relevant_set_ids") or query.get("R_q") or [])}
        for query in validation_queries
        if str(query.get("query_id")) in query_embs
    }

    rankings_before = _compute_full_rankings(query_ids, query_embs, corpus_ids, corpus_before)
    rankings_after = _compute_full_rankings(query_ids, query_embs_after, corpus_ids, corpus_after)

    query_metrics: dict[str, dict[str, Any]] = {}
    healthy_preserved_count = 0
    healthy_degraded_count = 0
    improved_error_count = 0
    error_rank_gain_total = 0
    error_top1_fixed_count = 0
    error_top10_gain_count = 0

    transitions = {
        "top10_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
        "top5_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
        "top3_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
        "top1_transitions": {"entered": 0, "exited": 0, "stayed_in": 0, "stayed_out": 0},
    }

    relevant_by_query = {
        str(query["query_id"]): set(str(doc_id) for doc_id in (query.get("relevant_set_ids") or query.get("R_q") or []))
        for query in validation_queries
    }

    def _update_transition(bucket: dict[str, int], before_rank: int, after_rank: int, k: int) -> None:
        before_in = before_rank <= k
        after_in = after_rank <= k
        if (not before_in) and after_in:
            bucket["entered"] += 1
        elif before_in and (not after_in):
            bucket["exited"] += 1
        elif before_in and after_in:
            bucket["stayed_in"] += 1
        else:
            bucket["stayed_out"] += 1

    for query_id in query_ids:
        relevant_ids = relevant_by_query.get(query_id, set())
        ranked_before = rankings_before.get(query_id, [])
        ranked_after = rankings_after.get(query_id, [])

        rank_before = _best_relevant_rank(ranked_before, relevant_ids)
        rank_after = _best_relevant_rank(ranked_after, relevant_ids)
        blockers_before = _blocker_count_before_best_relevant(ranked_before, relevant_ids)
        blockers_after = _blocker_count_before_best_relevant(ranked_after, relevant_ids)
        healthy_before = rank_before == 1
        healthy_after = rank_after == 1
        delta_rank = rank_before - rank_after

        query_metrics[query_id] = {
            "rank_before": int(rank_before),
            "rank_after": int(rank_after),
            "delta_rank": int(delta_rank),
            "healthy_before": bool(healthy_before),
            "healthy_after": bool(healthy_after),
            "blockers_before": int(blockers_before),
            "blockers_after": int(blockers_after),
        }

        if healthy_before and healthy_after:
            healthy_preserved_count += 1
        if healthy_before and (not healthy_after):
            healthy_degraded_count += 1

        if not healthy_before:
            rank_gain = max(0, int(rank_before - rank_after))
            error_rank_gain_total += rank_gain
            if rank_after < rank_before:
                improved_error_count += 1
            if (rank_before > 1) and (rank_after == 1):
                error_top1_fixed_count += 1
            if (rank_before > 10) and (rank_after <= 10):
                error_top10_gain_count += 1

        _update_transition(transitions["top10_transitions"], rank_before, rank_after, 10)
        _update_transition(transitions["top5_transitions"], rank_before, rank_after, 5)
        _update_transition(transitions["top3_transitions"], rank_before, rank_after, 3)
        _update_transition(transitions["top1_transitions"], rank_before, rank_after, 1)

    retrieval = _evaluate_retrieval_metrics(
        rankings_before=rankings_before,
        rankings_after=rankings_after,
        query_ids=query_ids,
        query_embs=query_embs,
        corpus_ids=corpus_ids,
        corpus_before=corpus_before,
        corpus_after=corpus_after,
        qrels=qrels,
        query_embs_after=query_embs_after,
    )

    return {
        "query_metrics": query_metrics,
        "global": {
            "healthy_preserved_count": int(healthy_preserved_count),
            "healthy_degraded_count": int(healthy_degraded_count),
            "improved_error_count": int(improved_error_count),
            "error_rank_gain_total": int(error_rank_gain_total),
            "error_top1_fixed_count": int(error_top1_fixed_count),
            "error_top10_gain_count": int(error_top10_gain_count),
            "query_count": int(len(query_ids)),
            **transitions,
        },
        "retrieval": retrieval,
    }


def exact_acceptance_key(metrics: dict[str, Any], include_mrr: bool = True) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("error_rank_gain_total", 0.0)),
        float(metrics.get("error_top1_fixed_count", 0.0)),
        float(metrics.get("error_top10_gain_count", 0.0)),
        float(metrics.get("ndcg@10_after", 0.0)),
        float(metrics.get("MRR_after", 0.0)) if include_mrr else 0.0,
    )


def accept_candidate_by_exact_metrics(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None = None,
    include_mrr: bool = True,
) -> bool:
    candidate_degraded = int(candidate_metrics.get("healthy_degraded_count", 0))
    if candidate_degraded > 0:
        return False

    if incumbent_metrics is None:
        return True

    incumbent_degraded = int(incumbent_metrics.get("healthy_degraded_count", 0))
    if incumbent_degraded > 0:
        return True

    candidate_key = exact_acceptance_key(candidate_metrics, include_mrr=include_mrr)
    incumbent_key = exact_acceptance_key(incumbent_metrics, include_mrr=include_mrr)
    return candidate_key > incumbent_key


def _apply_orthogonal_projection_to_dict(emb_dict: dict[str, np.ndarray], v: np.ndarray | None) -> dict[str, np.ndarray]:
    if v is None:
        return {k: np.copy(vv) for k, vv in emb_dict.items()}
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return {k: np.copy(vv) for k, vv in emb_dict.items()}
    v_unit = v / norm
    out: dict[str, np.ndarray] = {}
    for k, vec in emb_dict.items():
        vec = np.array(vec, dtype=np.float32)
        proj = np.dot(vec, v_unit) * v_unit
        out[k] = vec - proj
    return out


def _print_bench_triplets(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    phase: str,
) -> None:
    import json

    def _split_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        total = len(items)
        healthy = sum(1 for r in items if bool(r.get("is_healthy")))
        return {"total": total, "healthy": healthy, "error": total - healthy}

    payload = {
        "phase": phase,
        "train": _split_counts(train_items),
        "val": _split_counts(val_items),
    }
    print(f"[BENCH_TRIPLETS] {json.dumps(payload, ensure_ascii=False)}")


def _run_ablation_and_print_triplets(
    accepted_vectors: list[np.ndarray],
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    train_query_ids: set[str],
    val_query_ids: set[str],
    query_embs: dict[str, np.ndarray],
    corpus_original: dict[str, np.ndarray],
    qrel_map: dict[str, set[str]],
    healthy_top_k: int = 10,
    error_buffer: int = 10,
) -> None:
    import json

    for k in range(len(accepted_vectors), -1, -1):
        corpus_current = {kk: np.array(vv, dtype=np.float32) for kk, vv in corpus_original.items()}
        for v in accepted_vectors[:k]:
            corpus_current = _apply_orthogonal_projection_to_dict(corpus_current, v)

        query_data = _recompute_query_data_from_qrels(
            qrel_map, query_embs, corpus_current,
            healthy_top_k=healthy_top_k, error_buffer=error_buffer,
        )
        train_q = filter_query_data_by_ids(query_data, train_query_ids)
        val_q = filter_query_data_by_ids(query_data, val_query_ids)

        total_q = len(query_data)
        if total_q > 0:
            _print_bench_triplets(train_q, val_q, phase=f"ablation_{k}")
        else:
            print(f'[BENCH_TRIPLETS] {{"phase":"ablation_{k}","train":{{"total":0,"healthy":0,"error":0}},"val":{{"total":0,"healthy":0,"error":0}}}}')


def _recompute_query_data_from_qrels(
    qrel_map: dict[str, set[str]],
    query_embs: dict[str, np.ndarray],
    corpus_embs_current: dict[str, np.ndarray],
    healthy_top_k: int = 10,
    error_buffer: int = 10,
) -> list[dict[str, Any]]:
    query_ids = list(qrel_map.keys())
    corpus_ids = list(corpus_embs_current.keys())
    if not query_ids or not corpus_ids:
        return []

    q_matrix = np.stack([query_embs[qid] for qid in query_ids if qid in query_embs])
    c_matrix = np.stack([corpus_embs_current[cid] for cid in corpus_ids])
    q_norm = q_matrix / (np.linalg.norm(q_matrix, axis=1, keepdims=True) + 1e-10)
    c_norm = c_matrix / (np.linalg.norm(c_matrix, axis=1, keepdims=True) + 1e-10)
    sims = q_norm @ c_norm.T

    query_data: list[dict[str, Any]] = []
    for q_idx, qid in enumerate(query_ids):
        if qid not in query_embs:
            continue
        scores = sims[q_idx]
        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [corpus_ids[idx] for idx in ranked_indices]

        relevant_set = qrel_map.get(qid, set())

        best_relevant_doc_id: str | None = None
        best_relevant_rank = float("inf")
        for rank_index, doc_id in enumerate(ranked_doc_ids):
            if doc_id in relevant_set:
                best_relevant_doc_id = doc_id
                best_relevant_rank = rank_index + 1
                break

        health_label = classify_query_health(ranked_doc_ids, relevant_set)
        is_healthy = health_label == "healthy"

        healthy_confuser_ids: list[str] = []
        error_confuser_ids: list[str] = []
        if is_healthy:
            healthy_confuser_ids = select_healthy_confusers(ranked_doc_ids, relevant_set, k=healthy_top_k)
            confuser_ids = healthy_confuser_ids
        else:
            error_confuser_ids = select_error_confusers(ranked_doc_ids, relevant_set, best_relevant_doc_id, buffer_below=error_buffer)
            confuser_ids = error_confuser_ids

        query_data.append(
            {
                "query_id": qid,
                "R_q": sorted(relevant_set),
                "relevant_set_ids": sorted(relevant_set),
                "top1_doc_id": ranked_doc_ids[0] if ranked_doc_ids else None,
                "health_label": health_label,
                "is_healthy": is_healthy,
                "best_relevant_doc_id": best_relevant_doc_id,
                "best_rel_id": best_relevant_doc_id,
                "best_relevant_rank": best_relevant_rank,
                "initial_rank": best_relevant_rank,
                "healthy_confuser_ids": healthy_confuser_ids,
                "error_confuser_ids": error_confuser_ids,
                "confuser_ids": confuser_ids,
            }
        )

    return query_data


def map_search_strictness(
    strictness: str | None,
    error_count: int | None = None,
) -> tuple[str, float]:
    """
    Map high-level search_strictness into (healthy_weight_strategy, healthy_weight_value).

    Returns:
      - healthy_weight_strategy: 'uniform' | 'fixed' | 'error_scaled'
      - healthy_weight_value: numeric fixed value used when healthy_weight_strategy == 'fixed'

    'error_scaled' means the healthy weight w_h grows with the number of remaining
    errors (error_count + 1) rather than staying at a fixed numeric value; this is
    unrelated to the validation gate's 'adaptive' acceptance policy (top1_gt_degraded),
    which is a separate accept/reject decision, not a loss weight.

    If error_count is provided and healthy_weight_strategy == 'error_scaled', the
    returned healthy_weight_value will be float(error_count + 1) to be used by
    callers that expect an explicit numeric weight.

    The search_strictness levels 'strict' and 'quite_strict' both resolve to the
    same (fixed, 10.0) weight; the legacy 'quite_strict' label is kept only for
    backward compatibility with already-recorded artifact paths and is not a
    distinct behaviour.
    """
    if strictness is None:
        # default to the conservative 'quite_strict' behaviour
        strictness = "quite_strict"

    s = str(strictness).strip().lower()
    # accept french misspelling alias
    if s in ("very_strict", "adaptative"):
        healthy_strategy = "error_scaled"
        hv = float((int(error_count) + 1) if error_count is not None else 1.0)
    elif s in ("strict", "quite_strict"):
        healthy_strategy = "fixed"
        hv = 10.0
    elif s == "relaxed":
        healthy_strategy = "fixed"
        hv = 1.0
    else:
        # unknown: fallback to the strict/quite_strict behaviour
        healthy_strategy = "fixed"
        hv = 10.0

    if healthy_strategy == "error_scaled" and error_count is not None:
        hv = float(int(error_count) + 1)

    return healthy_strategy, float(hv)
