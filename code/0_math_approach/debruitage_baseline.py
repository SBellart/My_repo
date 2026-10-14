"""Unsupervised post-hoc baselines (whitening, all-but-the-top) for the benchmark matrix.

Fits a global affine transform on corpus embeddings, applies it to both corpus and
query embeddings (as prescribed by Su et al. 2021 and Mu & Viswanath 2018), then
reports the same validation/test/full metrics and stdout markers as the denoising
entrypoints so ``parse_bench_output`` and the matrix artifacts stay compatible.
"""
import importlib.util as _importlib_util
import os
import random
import sys
from pathlib import Path as _Path
from typing import Any, cast

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _load_shared_core() -> Any:
    core_path = _Path(__file__).with_name("_debruitage_query_core.py")
    spec = _importlib_util.spec_from_file_location("_debruitage_query_core", str(core_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load _debruitage_query_core.py")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load_shared_core()

prepare_query_level_data = _core.prepare_query_level_data
split_query_ids_three_way = _core.split_query_ids_three_way
filter_query_data_by_ids = _core.filter_query_data_by_ids
evaluate_exact_full_corpus_before_after = _core.evaluate_exact_full_corpus_before_after
_format_query_health_summary = _core._format_query_health_summary


def _corpus_matrix(corpus_embs: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    ids = list(corpus_embs.keys())
    return ids, np.stack([corpus_embs[doc_id] for doc_id in ids]).astype(np.float64)


def _apply_affine(
    emb_dict: dict[str, np.ndarray],
    mu: np.ndarray,
    w: np.ndarray | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, vec in emb_dict.items():
        centered = np.asarray(vec, dtype=np.float64) - mu.ravel()
        transformed = centered @ w if w is not None else centered
        out[key] = transformed.astype(np.float32)
    return out


def fit_mean_centering(corpus_embs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    _, matrix = _corpus_matrix(corpus_embs)
    return matrix.mean(axis=0, keepdims=True), None


def fit_whitening(corpus_embs: dict[str, np.ndarray], eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray | None]:
    """Whitening of Su et al. (2021): center then map corpus covariance to identity."""
    _, matrix = _corpus_matrix(corpus_embs)
    mu = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mu
    cov = (centered.T @ centered) / max(1, centered.shape[0])
    u, s, _ = np.linalg.svd(cov)
    w = u @ np.diag(1.0 / np.sqrt(s + eps))
    return mu, w


def fit_all_but_the_top(
    corpus_embs: dict[str, np.ndarray],
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """All-but-the-top of Mu & Viswanath (2018): remove mean and top-D principal directions.

    Default ``D = max(1, dim // 100)`` follows the paper's recommendation.
    """
    _, matrix = _corpus_matrix(corpus_embs)
    dim = matrix.shape[1]
    d_components = int(n_components) if n_components is not None else max(1, dim // 100)

    mu = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[:d_components]
    w = np.eye(dim) - top.T @ top
    return mu, w


BASELINE_FITTERS = {
    "mean_centering": fit_mean_centering,
    "whitening": fit_whitening,
    "abtt": fit_all_but_the_top,
}


def apply_mean_centering(corpus_embs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mu, w = fit_mean_centering(corpus_embs)
    return _apply_affine(corpus_embs, mu, w)


def apply_whitening(corpus_embs: dict[str, np.ndarray], eps: float = 1e-8) -> dict[str, np.ndarray]:
    mu, w = fit_whitening(corpus_embs, eps=eps)
    return _apply_affine(corpus_embs, mu, w)


def apply_all_but_the_top(
    corpus_embs: dict[str, np.ndarray],
    n_components: int | None = None,
) -> dict[str, np.ndarray]:
    mu, w = fit_all_but_the_top(corpus_embs, n_components=n_components)
    return _apply_affine(corpus_embs, mu, w)


BASELINE_METHODS = {
    "mean_centering": apply_mean_centering,
    "whitening": apply_whitening,
    "abtt": apply_all_but_the_top,
}


def _print_eval_block(label_before: str, label_after: str, summary_label: str, evaluation: dict[str, Any]) -> None:
    retrieval = evaluation.get("retrieval", {})
    before = retrieval.get("before", {}) or {}
    after = retrieval.get("after", {}) or {}
    global_metrics = evaluation.get("global", {})
    print(label_before + ", ".join(f"{k}={float(v):.4f}" for k, v in before.items()))
    print(label_after + ", ".join(f"{k}={float(v):.4f}" for k, v in after.items()))
    print(
        f"{summary_label} | Healthy degraded: {int(global_metrics.get('healthy_degraded_count', 0))} "
        f"| Error rank gain: {int(global_metrics.get('error_rank_gain_total', 0))} "
        f"| Error top1 fixes: {int(global_metrics.get('error_top1_fixed_count', 0))} "
        f"| Error top10 gains: {int(global_metrics.get('error_top10_gain_count', 0))}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Unsupervised post-hoc baseline (corpus-side global transform)")
    parser.add_argument("--dataset", type=str, required=True, help="MTEB dataset name (ex: mteb/scifact)")
    parser.add_argument(
        "--baseline-method",
        type=str,
        required=True,
        choices=sorted(BASELINE_METHODS.keys()),
        help="Global unsupervised transform applied to corpus embeddings",
    )
    parser.add_argument(
        "--abtt-components",
        type=int,
        default=None,
        help="Number of dominant components removed by abtt (default: dim // 100)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction at query-level")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Held-out test fraction at query-level (same split as the denoising runners)",
    )
    parser.add_argument("--healthy-top-k", type=int, default=10)
    parser.add_argument("--error-buffer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for train/val/test split")
    args = parser.parse_args()

    _set_seed(args.seed)


    print(
        f"Preparing query-level data for dataset {args.dataset} "
        f"(baseline={args.baseline_method}, healthy_top_k={args.healthy_top_k}, error_buffer={args.error_buffer})"
    )
    try:
        data = prepare_query_level_data(args.dataset, healthy_top_k=args.healthy_top_k, error_buffer=args.error_buffer)
    except Exception as e:
        print(f"Error: failed to prepare query-level data: {e}")
        raise SystemExit(1)

    query_data: list[dict[str, Any]] = cast(list, data.get("query_data") or [])
    query_embs: dict[str, np.ndarray] = cast(dict, data.get("query_embs") or {})
    corpus_embs: dict[str, np.ndarray] = cast(dict, data.get("corpus_embs") or {})

    if not query_data:
        print("No query-level data found for this dataset. Exiting.")
        return

    train_query_ids, val_query_ids, test_query_ids = split_query_ids_three_way(
        query_data,
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        seed=args.seed,
    )
    val_q = filter_query_data_by_ids(query_data, val_query_ids)
    test_q = filter_query_data_by_ids(query_data, test_query_ids)
    print(f"Total query-level items: {_format_query_health_summary(query_data)}")
    print(f"  Train      : {len(train_query_ids)} queries | Validation : {len(val_query_ids)} queries | Test : {len(test_query_ids)} queries")

    corpus_original: dict[str, np.ndarray] = {k: np.array(v, dtype=np.float32) for k, v in corpus_embs.items()}

    if args.baseline_method == "abtt":
        mu, w = fit_all_but_the_top(corpus_original, n_components=args.abtt_components)
    else:
        mu, w = BASELINE_FITTERS[args.baseline_method](corpus_original)
    corpus_after = _apply_affine(corpus_original, mu, w)
    query_embs_after = _apply_affine(query_embs, mu, w)

    print("\nValidation evaluation...")
    val_eval = evaluate_exact_full_corpus_before_after(
        val_q, query_embs, corpus_original, corpus_after, query_embs_after=query_embs_after
    )
    _print_eval_block(
        "Validation retrieval metrics (before): ",
        "Validation retrieval metrics (after) : ",
        "Validation summary",
        val_eval,
    )

    if test_query_ids:
        print("\nTest evaluation (held-out, never seen by the gate)...")
        test_eval = evaluate_exact_full_corpus_before_after(
            test_q, query_embs, corpus_original, corpus_after, query_embs_after=query_embs_after
        )
        _print_eval_block(
            "Test retrieval metrics (before): ",
            "Test retrieval metrics (after) : ",
            "Test summary",
            test_eval,
        )

    print("\nFinal evaluation on full query set...")
    final_eval = evaluate_exact_full_corpus_before_after(
        query_data, query_embs, corpus_original, corpus_after, query_embs_after=query_embs_after
    )
    final_query_metrics = final_eval.get("query_metrics", {})
    final_global = final_eval.get("global", {})
    final_healthy = sum(1 for q in final_query_metrics.values() if q.get("healthy_after"))
    final_errors = int(final_global.get("query_count", 0)) - final_healthy

    print(
        f"Iterations accepted: 0 | Final healthy: {final_healthy} | Final errors: {final_errors}\n"
        f"Healthy degraded count: {int(final_global.get('healthy_degraded_count', 0))} "
        f"| Error rank gain total: {int(final_global.get('error_rank_gain_total', 0))} "
        f"| Improved error count: {int(final_global.get('improved_error_count', 0))}\n"
        f"Error top1 fixes: {int(final_global.get('error_top1_fixed_count', 0))} "
        f"| Error top10 gains: {int(final_global.get('error_top10_gain_count', 0))}"
    )

    retrieval = final_eval.get("retrieval", {})
    retrieval_before = retrieval.get("before", {}) or {}
    retrieval_after = retrieval.get("after", {}) or {}
    print("Retrieval metrics (before): " + ", ".join(f"{k}={float(v):.4f}" for k, v in retrieval_before.items()))
    print("Retrieval metrics (after) : " + ", ".join(f"{k}={float(v):.4f}" for k, v in retrieval_after.items()))


if __name__ == "__main__":
    main()
