import importlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np


def _load_shared_core_for_cluster() -> Any:
    # load the newly created adjacent shared core module
    core_path = Path(__file__).with_name("_debruitage_query_core.py")
    spec = importlib.util.spec_from_file_location("_debruitage_query_core", str(core_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load _debruitage_query_core.py for cluster helper reuse")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load_shared_core_for_cluster()
_TqdmFallback = _core._TqdmFallback
_progress = _core._progress
_progress_write = _core._progress_write
_local_store_dir = _core._local_store_dir
_normalize_matrix = _core._normalize_matrix
_best_relevant_rank = _core._best_relevant_rank
_blocker_count_before_best_relevant = _core._blocker_count_before_best_relevant
prepare_query_level_data = _core.prepare_query_level_data
split_query_ids_by_health = _core.split_query_ids_by_health
split_query_ids_three_way = _core.split_query_ids_three_way
filter_query_data_by_ids = _core.filter_query_data_by_ids
_format_query_health_summary = _core._format_query_health_summary
_print_query_split_summary = _core._print_query_split_summary
_torch_module = _core._torch_module
_to_torch_vec = _core._to_torch_vec
_should_run_validation = _core._should_run_validation
_format_exact_metric_summary = _core._format_exact_metric_summary
_explain_exact_acceptance_change = _core._explain_exact_acceptance_change
_build_candidate_global = _core._build_candidate_global
compute_healthy_loss = _core.compute_healthy_loss
compute_healthy_loss_torch = _core.compute_healthy_loss_torch
compute_error_loss = _core.compute_error_loss
compute_error_loss_torch = _core.compute_error_loss_torch
evaluate_exact_full_corpus_before_after = _core.evaluate_exact_full_corpus_before_after
accept_candidate_by_exact_metrics = _core.accept_candidate_by_exact_metrics
_apply_orthogonal_projection_to_dict = _core._apply_orthogonal_projection_to_dict


def accept_candidate_by_top1_vs_degraded(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None = None,
) -> bool:
    """Accept a candidate only when it fixes more top-1 errors than healthy queries it degrades."""
    candidate_degraded = int(candidate_metrics.get("healthy_degraded_count", 0))
    candidate_top1_fixed = int(candidate_metrics.get("error_top1_fixed_count", 0))
    
    if candidate_top1_fixed <= candidate_degraded:
        return False

    if incumbent_metrics is None:
        return True

    incumbent_degraded = int(incumbent_metrics.get("healthy_degraded_count", 0))
    incumbent_top1_fixed = int(incumbent_metrics.get("error_top1_fixed_count", 0))

    # Net gain is the number of top-1 fixes minus the number of degraded healthy queries.
    candidate_net = candidate_top1_fixed - candidate_degraded
    incumbent_net = incumbent_top1_fixed - incumbent_degraded
    
    return candidate_net > incumbent_net


def _explain_top1_vs_degraded_change(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None = None,
) -> str:
    """Explain top1-vs-degraded acceptance decision, reflecting incumbent comparison."""
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


def _acceptance_strategy_dispatch(strategy: str):
    """Return a single-source (select_best_fn, accept_and_explain_fn) pair."""
    if strategy == "top1_gt_degraded":

        def accept_and_explain_top1(candidate, incumbent):
            accepted = accept_candidate_by_top1_vs_degraded(candidate, incumbent)
            reason = _explain_top1_vs_degraded_change(candidate, incumbent)
            return accepted, reason

        return accept_candidate_by_top1_vs_degraded, accept_and_explain_top1

    def accept_and_explain(candidate, incumbent):
        accepted = accept_candidate_by_exact_metrics(candidate, incumbent)
        reason = _explain_exact_acceptance_change(candidate, incumbent)
        return accepted, reason

    return accept_candidate_by_exact_metrics, accept_and_explain


def get_healthy_weight(
    strategy: str,
    healthy_count: int,
    error_count: int,
    fixed_value: float = 5.0,
) -> float:
    """Return the healthy-query weight for the chosen strategy."""
    s = str(strategy or "").strip().lower()
    # accept legacy aliases predating the error_scaled rename
    if s in ("adaptative", "adaptive"):
        s = "error_scaled"

    if s == "uniform":
        return 1.0
    elif s == "fixed":
        return fixed_value
    elif s == "error_scaled":
        return float(error_count + 1)
    else:
        return 1.0
_recompute_query_data_from_qrels = _core._recompute_query_data_from_qrels
_print_bench_triplets = _core._print_bench_triplets
_run_ablation_and_print_triplets = _core._run_ablation_and_print_triplets


def _build_ranked_snapshot(
    query_ids: list[str],
    query_embs: dict[str, np.ndarray],
    corpus_embs: dict[str, np.ndarray],
) -> dict[str, tuple[list[str], list[float]]]:
    corpus_ids = list(corpus_embs.keys())
    if not query_ids or not corpus_ids:
        return {query_id: ([], []) for query_id in query_ids}

    query_matrix = np.stack([query_embs[query_id] for query_id in query_ids])
    corpus_matrix = np.stack([corpus_embs[corpus_id] for corpus_id in corpus_ids])
    similarity_matrix = _normalize_matrix(query_matrix) @ _normalize_matrix(corpus_matrix).T

    out: dict[str, tuple[list[str], list[float]]] = {}
    for query_idx, query_id in enumerate(query_ids):
        scores = similarity_matrix[query_idx]
        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [str(corpus_ids[idx]) for idx in ranked_indices]
        ranked_scores = [float(scores[idx]) for idx in ranked_indices]
        out[query_id] = (ranked_doc_ids, ranked_scores)
    return out


def enrich_query_rows_with_ranked_scores(
    query_rows: list[dict[str, Any]],
    query_embs: dict[str, np.ndarray],
    corpus_embs: dict[str, np.ndarray],
    qrel_map: dict[str, set[str]],
) -> list[dict[str, Any]]:
    query_ids = [
        str(row.get("query_id"))
        for row in query_rows
        if str(row.get("query_id")) in query_embs
        and (row.get("ranked_doc_ids") is None or row.get("ranked_scores") is None)
    ]
    ranked_snapshot = _build_ranked_snapshot(query_ids, query_embs, corpus_embs)

    enriched_rows: list[dict[str, Any]] = []
    for row in query_rows:
        qid = str(row.get("query_id"))
        enriched = dict(row)
        if row.get("ranked_doc_ids") is not None and row.get("ranked_scores") is not None:
            enriched_rows.append(enriched)
            continue
        ranked_doc_ids, ranked_scores = ranked_snapshot.get(qid, ([], []))
        if ranked_doc_ids and ranked_scores:
            relevant_ids = set(qrel_map.get(qid, set()))
            best_relevant_doc_id = None
            for doc_id in ranked_doc_ids:
                if doc_id in relevant_ids:
                    best_relevant_doc_id = doc_id
                    break

            best_rank = float(_best_relevant_rank(ranked_doc_ids, relevant_ids)) if relevant_ids else float("inf")
            top1_doc_id = ranked_doc_ids[0] if ranked_doc_ids else None
            is_healthy = bool(top1_doc_id is not None and top1_doc_id in relevant_ids)
            enriched["ranked_doc_ids"] = ranked_doc_ids
            enriched["ranked_scores"] = ranked_scores
            enriched["top1_doc_id"] = top1_doc_id
            enriched["best_relevant_doc_id"] = best_relevant_doc_id
            enriched["best_rel_id"] = best_relevant_doc_id
            enriched["best_relevant_rank"] = best_rank
            enriched["initial_rank"] = best_rank
            enriched["is_healthy"] = is_healthy
            enriched["health_label"] = "healthy" if is_healthy else "error"
            enriched["R_q"] = sorted(relevant_ids)
            enriched["relevant_set_ids"] = sorted(relevant_ids)
        enriched_rows.append(enriched)

    return enriched_rows


def _lazy_import_sklearn():
    """Lazily import scikit-learn and provide a clear actionable error if missing.

    Returns the sklearn module if available. If not installed, raises RuntimeError with
    a concise installation hint so users running the clustered entrypoint know how to proceed.
    """
    try:
        import sklearn  # type: ignore

        return sklearn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scikit-learn is required for the clustered entrypoint but is not installed."
            " Install it with: pip install scikit-learn"
        ) from exc


def _extract_relevant_ids(error_query: dict[str, Any]) -> set[str]:
    raw_relevant = error_query.get("relevant_set_ids")
    if raw_relevant is None:
        raw_relevant = error_query.get("R_q")
    if raw_relevant is None:
        raise ValueError("Malformed error query row: missing relevant_set_ids/R_q")

    relevant_ids = {str(doc_id) for doc_id in raw_relevant}
    if not relevant_ids:
        raise ValueError("Malformed error query row: relevant_set_ids/R_q must be non-empty")
    return relevant_ids


def _extract_ranked_docs_and_scores(error_query: dict[str, Any]) -> tuple[list[str], list[float]]:
    ranked_doc_ids_raw = error_query.get("ranked_doc_ids")
    ranked_scores_raw = error_query.get("ranked_scores")
    if ranked_doc_ids_raw is None or ranked_scores_raw is None:
        raise ValueError("Malformed error query row: ranked_doc_ids and ranked_scores are required")

    ranked_doc_ids = [str(doc_id) for doc_id in ranked_doc_ids_raw]
    ranked_scores = [float(score) for score in ranked_scores_raw]
    if not ranked_doc_ids:
        raise ValueError("Malformed error query row: ranked_doc_ids must be non-empty")
    if len(ranked_doc_ids) != len(ranked_scores):
        raise ValueError("Malformed error query row: ranked_doc_ids and ranked_scores length mismatch")
    return ranked_doc_ids, ranked_scores


def extract_hybrid_error_signature(error_query: dict[str, Any]) -> np.ndarray:
    if bool(error_query.get("is_healthy")):
        raise ValueError("Hybrid signature extraction is only defined for error queries")
    relevant_ids_raw = error_query.get("relevant_set_ids") or error_query.get("R_q")
    if relevant_ids_raw is None:
        raise ValueError("Malformed error query row: missing relevant_set_ids/R_q")
    ranked_doc_ids_raw = error_query.get("ranked_doc_ids")
    ranked_scores_raw = error_query.get("ranked_scores")
    if ranked_doc_ids_raw is None or ranked_scores_raw is None:
        raise ValueError("Malformed error query row: ranked_doc_ids and ranked_scores are required")

    ranked_doc_ids = [str(doc_id) for doc_id in ranked_doc_ids_raw]
    ranked_scores = [float(score) for score in ranked_scores_raw]
    relevant_ids = {str(doc_id) for doc_id in relevant_ids_raw}

    best_relevant_rank = len(ranked_doc_ids) + 1
    best_relevant_score = 0.0
    blockers = 0
    for idx, doc_id in enumerate(ranked_doc_ids):
        if doc_id in relevant_ids:
            best_relevant_rank = idx + 1
            best_relevant_score = ranked_scores[idx]
            break
        blockers += 1

    top1_score = ranked_scores[0] if ranked_scores else 0.0
    confuser_count = sum(1 for doc_id in ranked_doc_ids if doc_id not in relevant_ids)

    return np.array(
        [
            float(np.log1p(float(best_relevant_rank))),
            float(np.log1p(float(blockers))),
            float(top1_score - best_relevant_score),
            float(np.log1p(float(confuser_count))),
        ],
        dtype=np.float32,
    )


def build_hybrid_error_signatures(
    error_queries: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    if not error_queries:
        raise ValueError("Cannot build hybrid signatures: no error queries provided")

    signatures: list[np.ndarray] = []
    query_ids: list[str] = []
    seen_query_ids: set[str] = set()

    for row in error_queries:
        query_id = str(row.get("query_id"))
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query_id in error queries: {query_id}")
        seen_query_ids.add(query_id)
        sig = extract_hybrid_error_signature(row)
        signatures.append(sig)
        query_ids.append(query_id)

    return np.stack(signatures).astype(np.float32, copy=False), query_ids


def build_confuser_semantic_signatures(
    error_queries: list[dict[str, Any]],
    corpus_embs: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    if not error_queries:
        raise ValueError("Cannot build signatures: no train-side error queries provided")

    query_ids: list[str] = []
    signatures: list[np.ndarray] = []
    seen_query_ids: set[str] = set()

    for row in error_queries:
        query_id_raw = row.get("query_id")
        if query_id_raw is None:
            continue
        
        query_id = str(query_id_raw)
        if query_id in seen_query_ids:
            continue

        ranked_doc_ids = row.get("ranked_doc_ids") or []
        relevant_ids = set(str(doc_id) for doc_id in (row.get("relevant_set_ids") or row.get("R_q") or []))

        confuser_vector = None
        for doc_id in ranked_doc_ids:
            doc_id_str = str(doc_id)
            if doc_id_str not in relevant_ids:
                confuser_vector = corpus_embs.get(doc_id_str)
                if confuser_vector is not None:
                    break
        
        if confuser_vector is not None:
            seen_query_ids.add(query_id)
            query_ids.append(query_id)
            signatures.append(confuser_vector)

    if not signatures:
        raise ValueError("Cannot build signatures: no confuser latent vectors found for error queries")

    matrix = np.stack(signatures).astype(np.float32, copy=False)
    return matrix, query_ids


def _as_2d_float32_matrix(signatures: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(signatures, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a 2-D matrix, got ndim={matrix.ndim}")
    if matrix.shape[0] <= 0:
        raise ValueError(f"{label} must include at least one row")
    if matrix.shape[1] <= 0:
        raise ValueError(f"{label} must include at least one feature column")
    return matrix


def handle_invalid_k(
    n_clusters: int,
    n_samples: int,
    n_distinct_signatures: int | None = None,
) -> None:
    if n_clusters <= 0:
        raise ValueError(f"Invalid K={n_clusters}: n_clusters must be >= 1")
    if n_samples <= 0:
        raise ValueError("Cannot cluster: no train-side error signatures available")
    if n_clusters > n_samples:
        raise ValueError(
            f"Invalid K={n_clusters}: requested clusters exceed train-side error signatures ({n_samples})"
        )

    if n_distinct_signatures is not None and n_distinct_signatures < n_clusters:
        raise ValueError(
            f"Invalid K={n_clusters}: only {n_distinct_signatures} distinct train-side error signatures available"
        )


def fit_train_only_scaler(train_error_signatures: np.ndarray) -> Any:
    _lazy_import_sklearn()
    matrix = _as_2d_float32_matrix(train_error_signatures, label="train_error_signatures")

    preprocessing = importlib.import_module("sklearn.preprocessing")
    scaler = preprocessing.Normalizer(norm="l2")
    scaler.fit(matrix)
    return scaler


def build_clusterer(n_clusters: int, random_state: int | None) -> Any:
    _lazy_import_sklearn()
    if n_clusters <= 0:
        raise ValueError(f"Invalid K={n_clusters}: n_clusters must be >= 1")

    cluster_module = importlib.import_module("sklearn.cluster")
    return cluster_module.KMeans(
        n_clusters=int(n_clusters),
        random_state=int(random_state) if random_state is not None else None,
        n_init=10,
    )


def fit_clusterer(clusterer: Any, train_signatures: np.ndarray) -> Any:
    matrix = _as_2d_float32_matrix(train_signatures, label="train_signatures")
    n_clusters = int(getattr(clusterer, "n_clusters"))
    distinct_rows = int(np.unique(matrix, axis=0).shape[0])
    handle_invalid_k(
        n_clusters=n_clusters,
        n_samples=int(matrix.shape[0]),
        n_distinct_signatures=distinct_rows,
    )

    exceptions = importlib.import_module("sklearn.exceptions")
    convergence_warning = exceptions.ConvergenceWarning

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        clusterer.fit(matrix)

    for warning_item in captured:
        if issubclass(warning_item.category, convergence_warning):
            warning_text = str(warning_item.message)
            if "distinct clusters" in warning_text.lower():
                raise RuntimeError(
                    "Degenerate clustering: fewer distinct clusters formed than requested K."
                    " Reduce K or provide more diverse train-side error signatures."
                )

    labels = np.asarray(getattr(clusterer, "labels_", []), dtype=np.int64)
    if labels.shape[0] != matrix.shape[0]:
        raise RuntimeError(
            "Degenerate clustering: fitted labels are inconsistent with train-side signature rows"
        )

    unique_clusters = np.unique(labels)
    if int(unique_clusters.shape[0]) != n_clusters:
        raise RuntimeError(
            "Degenerate clustering: fewer clusters produced than requested K."
            " Reduce K or improve train-side error-signature diversity."
        )

    return clusterer


def predict_cluster_assignments(clusterer: Any, signatures: np.ndarray) -> np.ndarray:
    matrix = _as_2d_float32_matrix(signatures, label="signatures")
    assignments = np.asarray(clusterer.predict(matrix), dtype=np.int64)
    if assignments.shape[0] != matrix.shape[0]:
        raise RuntimeError("Predict assignment mismatch: output length differs from input rows")
    return assignments


def build_query_cluster_mapping(query_ids: list[str], assignments: np.ndarray) -> dict[str, int]:
    if len(query_ids) != int(assignments.shape[0]):
        raise ValueError("Cannot build cluster mapping: query_ids and assignments length mismatch")

    mapping: dict[str, int] = {}
    for query_id, cluster_id in zip(query_ids, assignments):
        mapping[str(query_id)] = int(cluster_id)
    return mapping


def fit_train_error_clusters(
    query_ids: list[str],
    train_error_signatures: np.ndarray,
    *,
    n_clusters: int,
    seed: int | None = None,
) -> tuple[Any, Any, np.ndarray, dict[str, int]]:
    matrix = _as_2d_float32_matrix(train_error_signatures, label="train_error_signatures")
    if len(query_ids) != int(matrix.shape[0]):
        raise ValueError(
            "Cannot fit train error clusters: query_ids and train_error_signatures length mismatch"
        )

    scaler = fit_train_only_scaler(matrix)
    train_scaled = scaler.transform(matrix)
    clusterer = build_clusterer(n_clusters=n_clusters, random_state=seed)
    fit_clusterer(clusterer, train_scaled)
    assignments = predict_cluster_assignments(clusterer, train_scaled)
    mapping = build_query_cluster_mapping(query_ids, assignments)
    return scaler, clusterer, assignments, mapping


def assign_validation_clusters_for_diagnostics(
    query_ids: list[str],
    validation_error_signatures: np.ndarray,
    *,
    scaler: Any,
    clusterer: Any,
) -> dict[str, int]:
    matrix = _as_2d_float32_matrix(validation_error_signatures, label="validation_error_signatures")
    transformed = scaler.transform(matrix)
    assignments = predict_cluster_assignments(clusterer, transformed)
    return build_query_cluster_mapping(query_ids, assignments)


def _normalize_query_id_set(query_ids: set[str] | list[str] | tuple[str, ...]) -> set[str]:
    return {str(query_id) for query_id in query_ids}


def derive_active_error_query_ids(train_items: list[dict[str, Any]]) -> set[str]:
    active_error_query_ids: set[str] = set()
    for row in train_items:
        query_id = str(row.get("query_id"))
        if not bool(row.get("is_healthy")):
            active_error_query_ids.add(query_id)
    return active_error_query_ids


def derive_clustered_active_error_query_ids(
    train_items: list[dict[str, Any]],
    query_cluster_mapping: dict[str, int],
) -> set[str]:
    active_error_query_ids: set[str] = set()
    excluded_query_ids: list[str] = []
    for row in train_items:
        query_id = str(row.get("query_id"))
        if bool(row.get("is_healthy")):
            continue
        if query_id in query_cluster_mapping:
            active_error_query_ids.add(query_id)
        else:
            excluded_query_ids.append(query_id)

    if excluded_query_ids:
        _progress_write(
            "Clustering snapshot excluded "
            f"{len(excluded_query_ids)} train-side error queries without usable cluster signatures."
        )
    return active_error_query_ids


def refresh_active_error_query_ids(
    active_error_query_ids: set[str] | list[str] | tuple[str, ...],
    refreshed_train_items: list[dict[str, Any]],
) -> set[str]:
    active_ids = _normalize_query_id_set(active_error_query_ids)
    if not active_ids:
        return set()

    refreshed_health_by_query: dict[str, bool] = {}
    for row in refreshed_train_items:
        query_id = str(row.get("query_id"))
        refreshed_health_by_query[query_id] = bool(row.get("is_healthy"))

    refreshed_active_ids: set[str] = set()
    for query_id in active_ids:
        if query_id not in refreshed_health_by_query:
            continue
        if not refreshed_health_by_query[query_id]:
            refreshed_active_ids.add(query_id)
    return refreshed_active_ids


def refresh_active_error_sets(
    active_errors_by_query: dict[str, Any],
    refreshed_health_by_query: dict[str, bool],
) -> dict[str, Any]:
    refreshed_active: dict[str, Any] = {}
    for query_id, payload in active_errors_by_query.items():
        normalized_query_id = str(query_id)
        if normalized_query_id not in refreshed_health_by_query:
            continue
        if bool(refreshed_health_by_query[normalized_query_id]):
            continue
        refreshed_active[normalized_query_id] = payload
    return refreshed_active


def assemble_cluster_local_train_set(
    train_items: list[dict[str, Any]],
    *,
    active_cluster_id: int,
    query_cluster_mapping: dict[str, int],
    active_error_query_ids: set[str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    active_ids = _normalize_query_id_set(active_error_query_ids)

    selected_rows: list[dict[str, Any]] = []
    for row in train_items:
        query_id = str(row.get("query_id"))
        is_healthy = bool(row.get("is_healthy"))
        if is_healthy:
            selected_rows.append(row)
            continue

        if query_id not in active_ids:
            continue

        if query_id not in query_cluster_mapping:
            raise ValueError(
                f"Missing cluster assignment for active error query_id={query_id}"
            )

        if int(query_cluster_mapping[query_id]) != int(active_cluster_id):
            continue
        selected_rows.append(row)

    return selected_rows


def assemble_cluster_local_train_sets(
    assignments: list[int] | np.ndarray,
    dataset: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    if len(assignments) != len(dataset):
        raise ValueError("assignments and dataset length mismatch")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for cluster_id_raw, row in zip(assignments, dataset):
        cluster_id = int(cluster_id_raw)
        grouped.setdefault(cluster_id, []).append(row)
    return grouped


def _train_error_snapshot_by_cluster(
    train_items: list[dict[str, Any]],
    query_cluster_mapping: dict[str, int],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in train_items:
        if bool(row.get("is_healthy")):
            continue
        query_id = str(row.get("query_id"))
        cluster_id = query_cluster_mapping.get(query_id)
        if cluster_id is None:
            continue
        grouped.setdefault(int(cluster_id), []).append(row)
    return grouped


def order_clusters_by_error_volume(
    train_items: list[dict[str, Any]],
    query_cluster_mapping: dict[str, int],
) -> list[int]:
    grouped = _train_error_snapshot_by_cluster(train_items, query_cluster_mapping)
    stats: list[tuple[int, float, float, int]] = []
    for cluster_id, rows in grouped.items():
        if not rows:
            continue
        mean_rank = float(np.mean([float(row.get("best_relevant_rank", 0.0) or 0.0) for row in rows]))
        mean_blockers = float(
            np.mean(
                [
                    float(_blocker_count_before_best_relevant(list(row.get("ranked_doc_ids") or []), _extract_relevant_ids(row)))
                    for row in rows
                ]
            )
        )
        stats.append((int(cluster_id), mean_rank, mean_blockers, len(rows)))

    # Sort by descending volume, then by descending hardness for ties.
    stats.sort(key=lambda x: (-x[3], -x[1], -x[2], x[0]))
    return [cluster_id for cluster_id, _rank, _blockers, _size in stats]


def _refresh_active_errors_with_fixed_train_ids(
    active_error_query_ids: set[str],
    refreshed_train_items: list[dict[str, Any]],
    fixed_train_query_ids: set[str],
) -> set[str]:
    refreshed_active = refresh_active_error_query_ids(active_error_query_ids, refreshed_train_items)
    return {query_id for query_id in refreshed_active if query_id in fixed_train_query_ids}


def _refresh_active_errors_bidirectional(
    train_q: list[dict[str, Any]],
    fixed_train_query_ids: set[str],
    query_cluster_mapping: dict[str, int],
    corpus_embs_current: dict[str, np.ndarray],
    *,
    scaler: Any,
    clusterer: Any,
) -> set[str]:
    """Recompute active_error_query_ids bidirectionally after a projection update.

    - Errors initially-in-cluster that became healthy: removed from active set.
    - Previously-healthy queries that became errors: reintroduced as active and
      assigned to the nearest cluster via the frozen scaler + clusterer
      (signature = current top-1 non-relevant doc embedding).

    Queries whose signature cannot be computed (no exploitable confuser in the
    current ranking) are silently skipped and stay out of the active set until
    a subsequent refresh produces a usable signature.
    """
    active: set[str] = set()
    new_error_rows: list[dict[str, Any]] = []

    for row in train_q:
        query_id = str(row.get("query_id"))
        if query_id not in fixed_train_query_ids:
            continue
        if bool(row.get("is_healthy")):
            continue
        active.add(query_id)
        if query_id not in query_cluster_mapping:
            new_error_rows.append(row)

    if new_error_rows:
        try:
            signatures, new_query_ids = build_confuser_semantic_signatures(
                new_error_rows, corpus_embs_current
            )
        except ValueError:
            # No exploitable confusers found for any of the new errors: skip reassignment.
            return active

        scaled = scaler.transform(signatures)
        assignments = predict_cluster_assignments(clusterer, scaled)
        for query_id, cluster_id in zip(new_query_ids, assignments):
            query_cluster_mapping[str(query_id)] = int(cluster_id)

        # Queries that had no usable signature remain outside the mapping and
        # therefore outside any cluster-local training set for now.
        assigned_ids = {str(qid) for qid in new_query_ids}
        unassigned = {
            str(row.get("query_id")) for row in new_error_rows
        } - assigned_ids
        if unassigned:
            _progress_write(
                f"  Refresh: {len(unassigned)} newly-appeared errors had no usable confuser signature and were not re-clustered."
            )

    return active


def _validate_cluster_args(parser: Any, args: Any) -> None:
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.eval_every < 1:
        parser.error("--eval-every must be >= 1")
    if getattr(args, "n_clusters", None) is None:
        warnings.warn(
            "Initializing cluster count from --iterations is deprecated; "
            "please use explicit --n-clusters instead.",
            DeprecationWarning,
            stacklevel=2,
        )

def _fit_fixed_clusters_for_train_errors(
    train_q: list[dict[str, Any]],
    corpus_embs: dict[str, np.ndarray],
    *,
    iterations: int,
    seed: int | None = None,
) -> tuple[list[int], dict[str, int]]:
    cluster_order, query_cluster_mapping, _scaler, _clusterer = _fit_fixed_clusters_for_train_errors_with_models(
        train_q,
        corpus_embs,
        iterations=iterations,
        seed=seed,
    )
    return cluster_order, query_cluster_mapping


def _fit_fixed_clusters_for_train_errors_with_models(
    train_q: list[dict[str, Any]],
    corpus_embs: dict[str, np.ndarray],
    *,
    iterations: int,
    seed: int | None = None,
) -> tuple[list[int], dict[str, int], Any, Any]:
    train_error_rows = [row for row in train_q if not bool(row.get("is_healthy"))]
    train_signatures, train_error_query_ids = build_confuser_semantic_signatures(train_error_rows, corpus_embs)
    scaler, clusterer, assignments, query_cluster_mapping = fit_train_error_clusters(
        train_error_query_ids,
        train_signatures,
        n_clusters=int(iterations),
        seed=seed,
    )
    cluster_sizes: dict[int, int] = {}
    for cluster_id_raw in assignments:
        cluster_id = int(cluster_id_raw)
        cluster_sizes[cluster_id] = int(cluster_sizes.get(cluster_id, 0) + 1)

    singleton_cluster_ids = sorted(cluster_id for cluster_id, size in cluster_sizes.items() if int(size) <= 1)
    if singleton_cluster_ids:
        raise ValueError(
            "Invalid clustered setup: singleton train-side cluster(s) detected "
            f"{singleton_cluster_ids}. Reduce --iterations (K) or increase train-side error queries."
        )

    cluster_order = order_clusters_by_error_volume(train_q, query_cluster_mapping)
    if not cluster_order:
        raise RuntimeError("No non-empty train-side error clusters available for the clustered loop")
    return cluster_order, query_cluster_mapping, scaler, clusterer


def _query_relevance_map(query_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        str(row.get("query_id")): set(str(doc_id) for doc_id in (row.get("relevant_set_ids") or row.get("R_q") or []))
        for row in query_rows
    }


def _log_cluster_snapshot_diagnostics(
    train_q: list[dict[str, Any]],
    val_q: list[dict[str, Any]],
    query_cluster_mapping: dict[str, int],
    *,
    cluster_order: list[int],
    scaler: Any,
    clusterer: Any,
    corpus_embs_current: dict[str, np.ndarray],
    order_message: str,
) -> None:
    _progress_write(f"{order_message}: {cluster_order}")

    try:
        cluster_snapshot = _train_error_snapshot_by_cluster(train_q, query_cluster_mapping)
        cluster_sizes = {int(cid): len(rows) for cid, rows in cluster_snapshot.items()}
        _progress_write(f"Cluster sizes (train-side errors by cluster): {cluster_sizes}")
    except Exception:
        _progress_write("Cluster sizes: unavailable (diagnostics failure)")

    try:
        val_error_rows = [row for row in val_q if not bool(row.get("is_healthy"))]
        if not val_error_rows:
            _progress_write("Validation cluster diagnostics: no validation-side error queries")
            return

        val_signatures, val_error_query_ids = build_confuser_semantic_signatures(
            val_error_rows,
            corpus_embs_current,
        )
        val_cluster_mapping = assign_validation_clusters_for_diagnostics(
            val_error_query_ids,
            val_signatures,
            scaler=scaler,
            clusterer=clusterer,
        )
        val_cluster_sizes: dict[int, int] = {}
        for cluster_id in val_cluster_mapping.values():
            normalized = int(cluster_id)
            val_cluster_sizes[normalized] = int(val_cluster_sizes.get(normalized, 0) + 1)
        _progress_write(
            "Validation cluster diagnostics (error queries by current train clustering): "
            f"{val_cluster_sizes}"
        )
    except Exception as e:
        _progress_write(f"Validation cluster diagnostics unavailable: {e}")


def _compute_validation_cluster_sizes(
    val_q: list[dict[str, Any]],
    *,
    scaler: Any,
    clusterer: Any,
    corpus_embs_current: dict[str, np.ndarray],
) -> dict[int, int]:
    """Assign current validation-side error queries to train-side clusters for diagnostics."""
    val_error_rows = [row for row in val_q if not bool(row.get("is_healthy"))]
    if not val_error_rows:
        return {}

    val_signatures, val_error_query_ids = build_confuser_semantic_signatures(
        val_error_rows,
        corpus_embs_current,
    )
    val_cluster_mapping = assign_validation_clusters_for_diagnostics(
        val_error_query_ids,
        val_signatures,
        scaler=scaler,
        clusterer=clusterer,
    )
    val_cluster_sizes: dict[int, int] = {}
    for cluster_id in val_cluster_mapping.values():
        normalized = int(cluster_id)
        val_cluster_sizes[normalized] = int(val_cluster_sizes.get(normalized, 0) + 1)
    return val_cluster_sizes


def _make_cluster_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Debruitage cluster - clustered query-level loop with global exact acceptance")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mteb/scifact",
        help="MTEB dataset name (ex: mteb/scifact)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per clustered candidate")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum number of clustered outer iterations (max accepted correction vectors).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Explicit initial number of train-side clusters (K) to use for the clustered path. "
            "If not provided, legacy behaviour derives initial K from --iterations. "
            "This flag separates the clustering cardinality from the outer iteration budget."
        ),
    )
    parser.add_argument(
        "--healthy-top-k",
        type=int,
        default=10,
        help="Top-K non-relevant docs to consider as confusers for healthy queries. "
             "Controls the protection footprint of L_h. Too low on large/shared corpora "
             "leaves the loss blind to non-selected docs that can overtake after projection "
             "(consider 50-100 for corpora with hundreds of shared docs).",
    )
    parser.add_argument(
        "--error-buffer",
        type=int,
        default=10,
        help="Buffer size below best relevant to include as confusers for error queries. "
             "Increases L_e coverage beyond immediate blockers.",
    )
    parser.add_argument("--tau-h", type=float, default=0.05, help="Tau for healthy loss")
    parser.add_argument("--tau-e", type=float, default=0.05, help="Tau for error loss")
    parser.add_argument("--eval-every", type=int, default=1, help="Exact validation cadence in epochs")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction at query-level")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Held-out test fraction at query-level. Test queries are never seen by the acceptance gate; 0.0 keeps the legacy train/val behaviour.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the train/val/test split, error clustering, and gradient search restarts. "
             "Omit for a fresh random split/clustering on every run; pass a fixed value to reproduce one exact run.",
    )
    parser.add_argument(
        "--acceptance-strategy",
        type=str,
        default="exact",
        choices=["exact", "top1_gt_degraded", "strict", "adaptive", "adaptative"],
        help="Acceptance strategy: exact (zero degradation) or top1_gt_degraded (accept if top1 fixes > degraded)",
    )
    parser.add_argument(
        "--gate-acceptance-strategy",
        type=str,
        choices=["strict", "adaptive"],
        default="strict",
        help="Explicit gate acceptance strategy (strict|adaptive). If provided, overrides legacy --acceptance-strategy.",
    )
    parser.add_argument(
        "--search-strictness",
        type=str,
        choices=["very_strict", "strict", "relaxed"],
        default="strict",
        help="Search-time strictness mapping to search_acceptance + healthy-weight behaviour",
    )
    parser.add_argument(
        "--gate-scope",
        type=str,
        default="val",
        choices=["val", "full"],
        help="Query set used to compute degradation/gain metrics for the acceptance gate: "
             "'val' (query-level default, validation only) or 'full' (train+val, better represents "
             "true global impact of the projection)",
    )
    parser.add_argument(
        "--healthy-weight-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "fixed", "error_scaled"],
        help="Weight strategy for healthy queries: uniform (1.0), fixed (use --healthy-weight-value), error_scaled (error_count + 1). Distinct from --gate-acceptance-strategy's 'adaptive' gate.",
    )
    parser.add_argument(
        "--healthy-weight-value",
        type=float,
        default=5.0,
        help="Fixed weight value for healthy queries when using --healthy-weight-strategy=fixed",
    )
    return parser


def main() -> None:
    parser = _make_cluster_parser()
    args = parser.parse_args()
    _validate_cluster_args(parser, args)
    if args.seed is not None:
        np.random.seed(int(args.seed))

    _progress_write(
        f"Preparing query-level data for dataset {args.dataset} (healthy_top_k={args.healthy_top_k}, error_buffer={args.error_buffer})"
    )
    try:
        data = prepare_query_level_data(args.dataset, healthy_top_k=args.healthy_top_k, error_buffer=args.error_buffer)
    except Exception as e:
        _progress_write(f"Error: failed to prepare query-level data: {e}")
        raise SystemExit(1)

    query_data: list[dict[str, Any]] = cast(list, data.get("query_data") or [])
    query_embs: dict[str, np.ndarray] = cast(dict, data.get("query_embs") or {})
    corpus_embs: dict[str, np.ndarray] = cast(dict, data.get("corpus_embs") or {})

    if not query_data:
        _progress_write("No query-level data found for this dataset. Exiting.")
        return

    qrel_map = _query_relevance_map(query_data)
    query_data = enrich_query_rows_with_ranked_scores(query_data, query_embs, corpus_embs, qrel_map)

    if float(getattr(args, "test_fraction", 0.0)) > 0.0:
        train_query_ids, val_query_ids, test_query_ids = split_query_ids_three_way(
            query_data,
            val_fraction=float(args.val_fraction),
            test_fraction=float(args.test_fraction),
            seed=args.seed,
        )
    else:
        train_query_ids, val_query_ids = split_query_ids_by_health(
            query_data,
            val_fraction=float(args.val_fraction),
            seed=args.seed,
        )
        test_query_ids = set()
    train_q = filter_query_data_by_ids(query_data, train_query_ids)
    val_q = filter_query_data_by_ids(query_data, val_query_ids)
    test_q = filter_query_data_by_ids(query_data, test_query_ids)
    _progress_write(f"Total query-level items: {_format_query_health_summary(query_data)}")
    _print_query_split_summary(train_q, val_q)
    if test_q:
        _progress_write(f"  Test       : {_format_query_health_summary(test_q)}")
    _print_bench_triplets(train_q, val_q, "initial")

    try:
        _lazy_import_sklearn()
    except RuntimeError as e:
        _progress_write(str(e))
        raise SystemExit(1)

    # Determine initial clustering K: explicit --n-clusters takes precedence; otherwise fallback to legacy iterations value.
    initial_k = int(args.n_clusters) if getattr(args, "n_clusters", None) is not None else int(args.iterations)
    try:
        cluster_order, query_cluster_mapping, train_scaler, train_clusterer = _fit_fixed_clusters_for_train_errors_with_models(
            train_q,
            corpus_embs,
            iterations=initial_k,
            seed=args.seed,
        )
    except ValueError as e:
        _progress_write(f"Error: invalid clustered setup: {e}")
        raise SystemExit(1)
    except RuntimeError as e:
        _progress_write(f"Error: clustered setup failed: {e}")
        raise SystemExit(1)

    initial_order_message = "Initial cluster order (largest volume first)"
    _log_cluster_snapshot_diagnostics(
        train_q,
        val_q,
        query_cluster_mapping,
        cluster_order=cluster_order,
        scaler=train_scaler,
        clusterer=train_clusterer,
        corpus_embs_current=corpus_embs,
        order_message=initial_order_message,
    )

    using_local_store = bool(_local_store_dir())
    if not using_local_store:
        raise RuntimeError("PARASITIC_LOCAL_DB must be set to run in local mode.")

    try:
        corpus_current: dict[str, np.ndarray] = {k: np.array(v, dtype=np.float32) for k, v in corpus_embs.items()}
        corpus_original: dict[str, np.ndarray] = {k: np.array(v, dtype=np.float32) for k, v in corpus_embs.items()}

        # Note: cluster path uses candidate-only final acceptance; do not
        # maintain a cross-iteration incumbent_metrics state here.
        accepted_vectors: list[np.ndarray] = []
        active_error_query_ids = derive_clustered_active_error_query_ids(
            train_q,
            query_cluster_mapping,
        )
        torch_mod = _torch_module()

        # Normalize gate acceptance strategy: explicit new flag overrides legacy flag
        gate_choice = getattr(args, "gate_acceptance_strategy", None)
        if gate_choice is None:
            gate_choice = getattr(args, "acceptance_strategy", "exact")
        gate_choice = str(gate_choice).strip().lower()
        if gate_choice in ("strict", "exact"):
            acceptance_strategy = "exact"
        else:
            acceptance_strategy = "top1_gt_degraded"
        _accept_candidate_for_best, _accept_and_explain = _acceptance_strategy_dispatch(acceptance_strategy)

        # Map search_strictness into the healthy-weight behaviour for cluster-local training.
        # Use the already-loaded shared core module to avoid relative import issues when run as a script
        map_search_strictness = getattr(_core, "map_search_strictness")
        cluster_error_count = sum(1 for row in train_q if not bool(row.get("is_healthy")))
        healthy_weight_strategy, healthy_weight_value = map_search_strictness(
            getattr(args, "search_strictness", None), error_count=cluster_error_count
        )

        # Build a cluster-scoped accept-and-explain wrapper that always runs the
        # acceptance decision in a candidate-only mode (incumbent=None) and
        # produces an explanation text that does NOT mention any incumbent.
        def _cluster_accept_and_explain(candidate_metrics: dict[str, Any]) -> tuple[bool, str]:
            # Run the underlying acceptance predicate with incumbent=None
            try:
                accepted = _accept_candidate_for_best(candidate_metrics, None)
            except Exception:
                # Fallback to conservative reject on unexpected errors
                accepted = False

            # Produce a cleaned explanation string without 'no incumbent' wording.
            try:
                # Use the same explanation generator but pass incumbent=None
                reason = _accept_and_explain(candidate_metrics, None)[1]
            except Exception:
                reason = "candidate-only gate decision"

            # Normalize phrases that refer to an incumbent to a cluster-local wording
            if isinstance(reason, str) and "no incumbent" in reason:
                # Replace verbose wording with a concise candidate-only explanation
                reason = reason.replace("no incumbent; ", "").replace("no incumbent yet and ", "")
                reason = f"candidate-only gate: {reason}"

            return accepted, reason

        # Adaptive outer loop: after each accepted vector we recompute train-side
        # clusters in the CURRENT projected space with
        # k_remaining = args.iterations - len(accepted_vectors).
        # Initial clustering uses the startup order computed above.
        # Every adaptive reclustering afterward restarts from the largest current cluster.
        outer_it = 0
        need_recluster = False
        current_cluster_index = 0
        while len(accepted_vectors) < int(args.iterations):
            if need_recluster:
                # Adaptive reclustering must respect explicit n_clusters when provided,
                # otherwise preserve legacy behaviour of deriving K from iterations.
                if getattr(args, "n_clusters", None) is not None:
                    k_remaining = max(1, min(int(args.n_clusters), len(active_error_query_ids)))
                else:
                    k_remaining = int(args.iterations) - len(accepted_vectors)
                if k_remaining <= 0:
                    _progress_write("No remaining cluster slots (k_remaining <= 0). Stopping.")
                    break
                try:
                    cluster_order, query_cluster_mapping, train_scaler, train_clusterer = _fit_fixed_clusters_for_train_errors_with_models(
                        train_q,
                        corpus_current,
                        iterations=int(k_remaining),
                        seed=args.seed,
                    )
                except Exception as e:
                    _progress_write(
                        f"Adaptive reclustering unavailable for k_remaining={k_remaining}: {e}. Stopping outer loop."
                    )
                    break
                _log_cluster_snapshot_diagnostics(
                    train_q,
                    val_q,
                    query_cluster_mapping,
                    cluster_order=cluster_order,
                    scaler=train_scaler,
                    clusterer=train_clusterer,
                    corpus_embs_current=corpus_current,
                    order_message=(
                        "Adaptive cluster order (largest volume first, "
                        f"k_remaining={k_remaining})"
                    ),
                )
                active_error_query_ids = derive_clustered_active_error_query_ids(
                    train_q,
                    query_cluster_mapping,
                )
                current_cluster_index = 0
                need_recluster = False

            if current_cluster_index >= len(cluster_order):
                _progress_write("No more cluster slots in current clustering snapshot. Stopping.")
                break

            outer_it += 1
            active_cluster_id = cluster_order[current_cluster_index]
            current_cluster_index += 1
            _progress_write("\n" + "=" * 50)
            _progress_write(f"Outer iteration {outer_it}: active cluster={active_cluster_id}, train epochs={args.epochs}")
            _progress_write(f"Active cluster sequence position: {outer_it} / {int(args.iterations)}")
            _print_query_split_summary(train_q, val_q, header="Current query split snapshot:")

            # Build the query set used by the acceptance gate.
            # --gate-scope=val  -> validation only (query-level default behaviour)
            # --gate-scope=full -> train + val (gate sees the full global impact of the projection)
            gate_scope = getattr(args, "gate_scope", "val")
            if gate_scope == "full":
                gate_queries = train_q + val_q
                _progress_write(
                    f"Gate scope = full (train+val): {_format_query_health_summary(gate_queries)}"
                )
            else:
                gate_queries = val_q
                _progress_write(
                    f"Gate scope = val: {_format_query_health_summary(gate_queries)}"
                )

            cluster_train_q = assemble_cluster_local_train_set(
                train_q,
                active_cluster_id=int(active_cluster_id),
                query_cluster_mapping=query_cluster_mapping,
                active_error_query_ids=active_error_query_ids,
            )

            cluster_error_count = sum(1 for row in cluster_train_q if not bool(row.get("is_healthy")))
            _progress_write(
                f"Cluster-local training set: {_format_query_health_summary(cluster_train_q)} (active cluster errors={cluster_error_count})"
            )
            # Also print the active cluster id and current error ids for runtime tracing.
            try:
                active_ids_sorted = sorted({str(r.get("query_id")) for r in cluster_train_q if not bool(r.get("is_healthy"))})
                _progress_write(f"Active cluster id={active_cluster_id} contains {len(active_ids_sorted)} active queries: {active_ids_sorted}")
            except Exception:
                pass
            if cluster_error_count == 0:
                _progress_write(
                    "Active cluster has no remaining train-side errors in the current clustering snapshot. "
                    "Continuing to the next available cluster slot."
                )
                continue

            first_key = next(iter(corpus_current))
            dim = int(corpus_current[first_key].shape[0])
            v0 = np.random.normal(size=(dim,)).astype(np.float32)
            if np.linalg.norm(v0) == 0:
                _progress_write("Generated zero initialization vector; skipping cluster slot.")
                continue
            v0 = v0 / (np.linalg.norm(v0) + 1e-12)

            candidate_best_metrics = None
            candidate_best_v = None
            candidate_best_after = None
            patience = max(1, args.epochs // 4)
            no_improve = 0

            device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
            v_param = torch_mod.nn.Parameter(torch_mod.tensor(v0, dtype=torch_mod.float32, device=device))
            optim = torch_mod.optim.Adam([v_param], lr=1e-2)
            pbar = _progress(range(1, args.epochs + 1), desc=f"Iter {outer_it} cluster {active_cluster_id}", unit="epoch", leave=False)

            # Compute healthy weight once per cluster (cluster size and error count are fixed for this slot)
            # Prefer healthy weight from mapped search_strictness, but fallback to explicit cluster args
            hw_strategy = healthy_weight_strategy if 'healthy_weight_strategy' in locals() else None
            hw_value = healthy_weight_value if 'healthy_weight_value' in locals() else None
            # If mapping provided a strategy, use it
            if 'healthy_weight_strategy' in locals():
                hw_strategy = healthy_weight_strategy
                hw_value = healthy_weight_value
            else:
                hw_strategy = getattr(args, 'healthy_weight_strategy', 'uniform')
                hw_value = getattr(args, 'healthy_weight_value', 5.0)
            cluster_healthy_count = len(cluster_train_q) - cluster_error_count
            healthy_weight = get_healthy_weight(
                hw_strategy,
                cluster_healthy_count,
                cluster_error_count,
                hw_value,
            )

            for ep in pbar:
                v_unit = v_param / (torch_mod.linalg.norm(v_param) + 1e-12)
                corpus_after_torch: dict[str, Any] = {}
                for k, vec in corpus_current.items():
                    x = _to_torch_vec(vec, device=device)
                    proj = torch_mod.dot(x, v_unit) * v_unit
                    corpus_after_torch[k] = x - proj

                try:
                    healthy_loss_t = compute_healthy_loss_torch(
                        cluster_train_q,
                        query_embs,
                        corpus_after_torch,
                        tau_h=float(args.tau_h),
                    )
                    error_loss_t = compute_error_loss_torch(
                        cluster_train_q,
                        query_embs,
                        corpus_after_torch,
                        tau_e=float(args.tau_e),
                    )
                except Exception as e:
                    _progress_write(f"Torch loss exception: {e}")
                    # Keep v_param in the computation graph so backward() always has a path.
                    healthy_loss_t = torch_mod.sum(v_param) * 0.0
                    error_loss_t = torch_mod.sum(v_param) * 0.0

                # Apply healthy-weight strategy (symmetric with NumPy branch)
                loss = error_loss_t + healthy_loss_t * float(healthy_weight) + torch_mod.sum(v_param) * 0.0

                optim.zero_grad()
                loss.backward()
                optim.step()

                v_cur = v_param.detach().cpu().numpy()
                postfix = {
                    "h_loss": f"{float(healthy_loss_t.detach().cpu().item()):.4f}",
                    "e_loss": f"{float(error_loss_t.detach().cpu().item()):.4f}",
                }

                if _should_run_validation(ep, args.epochs, int(args.eval_every)):
                    corpus_after = _apply_orthogonal_projection_to_dict(corpus_current, v_cur)
                    metrics_full = evaluate_exact_full_corpus_before_after(gate_queries, query_embs, corpus_current, corpus_after)
                    candidate_global = _build_candidate_global(metrics_full)

                    if candidate_best_metrics is None or _accept_candidate_for_best(candidate_global, candidate_best_metrics):
                        candidate_best_metrics = candidate_global
                        candidate_best_v = v_cur.copy()
                        candidate_best_after = corpus_after
                        no_improve = 0
                    else:
                        no_improve += 1

                    postfix.update(
                        {
                            "rank_gain": str(int(candidate_global.get("error_rank_gain_total", 0))),
                            "degraded": str(int(candidate_global.get("healthy_degraded_count", 0))),
                            "top1": str(int(candidate_global.get("error_top1_fixed_count", 0))),
                            "top10": str(int(candidate_global.get("error_top10_gain_count", 0))),
                        }
                    )
                    _progress_write(f"  Validation epoch {ep}/{args.epochs}: {_format_exact_metric_summary(candidate_global)}")

                    if no_improve >= patience:
                        _progress_write(
                            f"  Early stopping cluster {active_cluster_id} at epoch {ep} (no improvement {no_improve}/{patience})"
                        )
                        break

                pbar.set_postfix(postfix)

            pbar.close()
            if candidate_best_metrics is None:
                _progress_write("No valid cluster candidate produced a validation checkpoint. Stopping.")
                break

            # Final acceptance MUST depend only on the candidate metrics in the
            # current projected state. Do not compare against any incumbent
            # (global previously-accepted) metrics here. Use the cluster-scoped
            # wrapper which enforces candidate-only mode and sanitizes the
            # explanation string.
            accepted, reason = _cluster_accept_and_explain(candidate_best_metrics)

            if not accepted:
                _progress_write(
                    f"Outer iteration {outer_it} REJECTED: {reason}"
                )
                _progress_write(f"  Candidate metrics: {_format_exact_metric_summary(candidate_best_metrics)}")
                break

            _progress_write(f"Outer iteration {outer_it} ACCEPTED: {reason}")
            _progress_write(f"  Accepted metrics: {_format_exact_metric_summary(candidate_best_metrics)}")
            # Do NOT record an incumbent_metrics summary for cross-iteration
            # comparison in the clustered outer loop. Acceptance is local-only.
            if candidate_best_v is not None:
                # Diagnostic: norm of the accepted vector + cosine similarity against all
                # previously accepted vectors. Helps detect (a) drift/collapse of v_k norms
                # and (b) colinearity between successive v_k (redundant directions).
                v_norm = float(np.linalg.norm(candidate_best_v))
                if v_norm > 0 and accepted_vectors:
                    v_unit = candidate_best_v / (v_norm + 1e-12)
                    cos_with_previous = []
                    for prev_idx, v_prev in enumerate(accepted_vectors, start=1):
                        prev_norm = float(np.linalg.norm(v_prev))
                        if prev_norm == 0:
                            cos_with_previous.append((prev_idx, 0.0))
                            continue
                        prev_unit = v_prev / (prev_norm + 1e-12)
                        cos = float(np.dot(v_unit, prev_unit))
                        cos_with_previous.append((prev_idx, cos))
                    cos_summary = ", ".join(f"v{idx}={c:+.4f}" for idx, c in cos_with_previous)
                    max_abs_cos = max((abs(c) for _, c in cos_with_previous), default=0.0)
                    _progress_write(
                        f"  Vector diagnostics: ||v_{len(accepted_vectors) + 1}||={v_norm:.4f}, "
                        f"max|cos(v_new, v_prev)|={max_abs_cos:.4f}, cos details: [{cos_summary}]"
                    )
                else:
                    _progress_write(
                        f"  Vector diagnostics: ||v_{len(accepted_vectors) + 1}||={v_norm:.4f} (first accepted vector)"
                    )
            # Only append concrete numpy vectors; candidate_best_v may be None
            # in marginal cases that confuse static type checkers or unexpected paths.
            if candidate_best_v is not None:
                accepted_vectors.append(candidate_best_v)
            else:
                _progress_write("Accepted candidate had no concrete vector; skipping append.")
            need_recluster = True

            if candidate_best_after is not None:
                corpus_current = candidate_best_after

            query_data = _recompute_query_data_from_qrels(
                qrel_map,
                query_embs,
                corpus_current,
                healthy_top_k=int(args.healthy_top_k),
                error_buffer=int(args.error_buffer),
            )
            query_data = enrich_query_rows_with_ranked_scores(query_data, query_embs, corpus_current, qrel_map)
            train_q = filter_query_data_by_ids(query_data, train_query_ids)
            val_q = filter_query_data_by_ids(query_data, val_query_ids)
            active_error_query_ids = derive_active_error_query_ids(train_q)
            _progress_write(
                "After acceptance: refreshed query-level supervision within fixed split buckets; "
                "next iteration will recluster residual train-side errors adaptively."
            )
            _print_query_split_summary(train_q, val_q, header="Refreshed fixed query split snapshot:")
            _print_bench_triplets(train_q, val_q, f"iteration_{outer_it}")
            if not active_error_query_ids:
                _progress_write(
                    "After acceptance: no remaining train-side error queries. Stopping outer loop after current accepted vector."
                )
                break

        _run_ablation_and_print_triplets(
            accepted_vectors,
            train_q, val_q,
            train_query_ids, val_query_ids,
            query_embs, corpus_original, qrel_map,
            healthy_top_k=int(args.healthy_top_k),
            error_buffer=int(args.error_buffer),
        )

        _progress_write("\nValidation evaluation...")
        val_eval = evaluate_exact_full_corpus_before_after(val_q, query_embs, corpus_original, corpus_current)
        val_retrieval = val_eval.get("retrieval", {})
        val_retrieval_before = val_retrieval.get("before", {}) or {}
        val_retrieval_after = val_retrieval.get("after", {}) or {}
        val_global = val_eval.get("global", {})
        _progress_write("Validation retrieval metrics (before): " + ", ".join(f"{k}={float(v):.4f}" for k, v in val_retrieval_before.items()))
        _progress_write("Validation retrieval metrics (after) : " + ", ".join(f"{k}={float(v):.4f}" for k, v in val_retrieval_after.items()))
        _progress_write(
            f"Validation summary | Healthy degraded: {int(val_global.get('healthy_degraded_count', 0))} "
            f"| Error rank gain: {int(val_global.get('error_rank_gain_total', 0))} "
            f"| Error top1 fixes: {int(val_global.get('error_top1_fixed_count', 0))} "
            f"| Error top10 gains: {int(val_global.get('error_top10_gain_count', 0))}"
        )

        if test_query_ids:
            _progress_write("\nTest evaluation (held-out, never seen by the gate)...")
            test_q = filter_query_data_by_ids(query_data, test_query_ids)
            test_eval = evaluate_exact_full_corpus_before_after(test_q, query_embs, corpus_original, corpus_current)
            test_retrieval = test_eval.get("retrieval", {})
            test_retrieval_before = test_retrieval.get("before", {}) or {}
            test_retrieval_after = test_retrieval.get("after", {}) or {}
            test_global = test_eval.get("global", {})
            _progress_write("Test retrieval metrics (before): " + ", ".join(f"{k}={float(v):.4f}" for k, v in test_retrieval_before.items()))
            _progress_write("Test retrieval metrics (after) : " + ", ".join(f"{k}={float(v):.4f}" for k, v in test_retrieval_after.items()))
            _progress_write(
                f"Test summary | Healthy degraded: {int(test_global.get('healthy_degraded_count', 0))} "
                f"| Error rank gain: {int(test_global.get('error_rank_gain_total', 0))} "
                f"| Error top1 fixes: {int(test_global.get('error_top1_fixed_count', 0))} "
                f"| Error top10 gains: {int(test_global.get('error_top10_gain_count', 0))}"
            )

        _progress_write("\nFinal evaluation on full query set...")
        final_eval = evaluate_exact_full_corpus_before_after(query_data, query_embs, corpus_original, corpus_current)
        final_query_metrics = final_eval.get("query_metrics", {})
        final_global = final_eval.get("global", {})
        final_healthy = sum(1 for q in final_query_metrics.values() if q.get("healthy_after"))
        final_errors = int(final_global.get("query_count", 0)) - final_healthy

        healthy_degraded = int(final_global.get("healthy_degraded_count", 0))
        error_rank_gain_total = int(final_global.get("error_rank_gain_total", 0))
        improved_error_count = int(final_global.get("improved_error_count", 0))
        error_top1_fixed_count = int(final_global.get("error_top1_fixed_count", 0))
        error_top10_gain_count = int(final_global.get("error_top10_gain_count", 0))

        _progress_write(
            f"Iterations accepted: {len(accepted_vectors)} | Final healthy: {final_healthy} | Final errors: {final_errors}\n"
            f"Healthy degraded count: {healthy_degraded} | Error rank gain total: {error_rank_gain_total} | Improved error count: {improved_error_count}\n"
            f"Error top1 fixes: {error_top1_fixed_count} | Error top10 gains: {error_top10_gain_count}"
        )

        retrieval = final_eval.get("retrieval", {})
        retrieval_before = retrieval.get("before", {}) or {}
        retrieval_after = retrieval.get("after", {}) or {}
        _progress_write("Retrieval metrics (before): " + ", ".join(f"{k}={float(v):.4f}" for k, v in retrieval_before.items()))
        _progress_write("Retrieval metrics (after) : " + ", ".join(f"{k}={float(v):.4f}" for k, v in retrieval_after.items()))

    finally:
        pass


if __name__ == "__main__":
    main()
