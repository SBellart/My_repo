import importlib
import os
import sys
from typing import Any, Callable, cast

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import importlib.util as _importlib_util
from pathlib import Path as _Path

def _load_shared_core() -> Any:
    core_path = _Path(__file__).with_name("_debruitage_query_core.py")
    spec = _importlib_util.spec_from_file_location("_debruitage_query_core", str(core_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load _debruitage_query_core.py")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load_shared_core()

# Re-export shared helpers expected by tests and by v4
_TqdmFallback = _core._TqdmFallback
_progress = _core._progress
_progress_write = _core._progress_write
_local_store_dir = _core._local_store_dir
decode_vector = _core.decode_vector
classify_query_health = _core.classify_query_health
select_healthy_confusers = _core.select_healthy_confusers
select_error_confusers = _core.select_error_confusers
split_queries_by_query_level = _core.split_queries_by_query_level
split_query_ids_by_health = _core.split_query_ids_by_health
split_query_ids_three_way = _core.split_query_ids_three_way
filter_query_data_by_ids = _core.filter_query_data_by_ids
_count_query_health = _core._count_query_health
_format_query_health_summary = _core._format_query_health_summary
_print_query_split_summary = _core._print_query_split_summary
_should_run_validation = _core._should_run_validation
_metric_value = _core._metric_value
_format_exact_metric_summary = _core._format_exact_metric_summary
_explain_exact_acceptance_change = _core._explain_exact_acceptance_change
_build_candidate_global = _core._build_candidate_global
prepare_query_level_data = _core.prepare_query_level_data
split_query_data = _core.split_query_data
_cosine = _core._cosine
_logsumexp = _core._logsumexp
s_rel = _core.s_rel
healthy_loss_for_query = _core.healthy_loss_for_query
compute_healthy_loss = _core.compute_healthy_loss
healthy_loss_for_query_torch = _core.healthy_loss_for_query_torch
compute_healthy_loss_torch = _core.compute_healthy_loss_torch
_torch_module = _core._torch_module
_try_torch_module = _core._try_torch_module
_to_torch_vec = _core._to_torch_vec
_cosine_torch = _core._cosine_torch
s_rel_torch = _core.s_rel_torch
error_loss_for_query_torch = _core.error_loss_for_query_torch
compute_error_loss_torch = _core.compute_error_loss_torch
error_loss_for_query = _core.error_loss_for_query
compute_error_loss = _core.compute_error_loss
_normalize_matrix = _core._normalize_matrix
_compute_full_rankings = _core._compute_full_rankings
_build_full_run = _core._build_full_run
_best_relevant_rank = _core._best_relevant_rank
_blocker_count_before_best_relevant = _core._blocker_count_before_best_relevant
_dcg_at_k = _core._dcg_at_k
_ndcg_at_k = _core._ndcg_at_k
_recall_at_k = _core._recall_at_k
_average_precision_at_k = _core._average_precision_at_k
_reciprocal_rank = _core._reciprocal_rank
_evaluate_retrieval_metrics_offline = _core._evaluate_retrieval_metrics_offline
_evaluate_retrieval_metrics = _core._evaluate_retrieval_metrics
evaluate_exact_full_corpus_before_after = _core.evaluate_exact_full_corpus_before_after
exact_acceptance_key = _core.exact_acceptance_key
accept_candidate_by_exact_metrics = _core.accept_candidate_by_exact_metrics
_apply_orthogonal_projection_to_dict = _core._apply_orthogonal_projection_to_dict
_recompute_query_data_from_qrels = _core._recompute_query_data_from_qrels
_print_bench_triplets = _core._print_bench_triplets
map_search_strictness = _core.map_search_strictness
_run_ablation_and_print_triplets = _core._run_ablation_and_print_triplets


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parasitic projection learning - global (query-level) variant")
    parser.add_argument("--dataset", type=str, default="mteb/scifact", help="MTEB dataset name (ex: mteb/scifact)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per restart candidate")
    parser.add_argument("--iterations", type=int, default=5, help="Number of outer correction iterations (apply accepted vector and repeat)")
    parser.add_argument("--n-restarts", type=int, default=8, help="Random restarts per iteration to explore candidate vectors")
    parser.add_argument("--healthy-top-k", type=int, default=10, help="Top-K non-relevant docs to consider for healthy queries")
    parser.add_argument("--error-buffer", type=int, default=10, help="Buffer size below best relevant to include for error queries")
    parser.add_argument(
        "--acceptance-strategy",
        choices=["exact", "top1_gt_degraded", "strict", "adaptive", "adaptative"],
        default="exact",
        help=(
            "Acceptance strategy. Legacy callers may use 'exact' or 'top1_gt_degraded'.\n"
            "Also accept paper-facing normalizations: 'strict'->exact, 'adaptive'->top1_gt_degraded (normalizes)."
        ),
    )
    parser.add_argument(
        "--search-strictness",
        choices=["very_strict", "strict", "relaxed"],
        default="strict",
        help="Search-time strictness mapping to search_acceptance + healthy-weight behaviour",
    )
    parser.add_argument(
        "--gate-acceptance-strategy",
        choices=["strict", "adaptive"],
        default=None,
        help="Explicit gate acceptance strategy (strict|adaptive). If provided, overrides legacy --acceptance-strategy.",
    )
    parser.add_argument("--tau-h", type=float, default=0.05, help="Tau for healthy loss (used during per-restart training)")
    parser.add_argument("--tau-e", type=float, default=0.05, help="Tau for error loss (used during per-restart training)")
    parser.add_argument("--eval-every", type=int, default=1, help="Exact validation cadence in epochs per restart")
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
        help="Random seed for the train/val/test split and gradient search restarts. "
             "Omit for a fresh random split on every run; pass a fixed value to reproduce one exact run.",
    )

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(int(args.seed))

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.n_restarts < 1:
        parser.error("--n-restarts must be >= 1")
    if args.eval_every < 1:
        parser.error("--eval-every must be >= 1")


    print(f"Preparing query-level data for dataset {args.dataset} (healthy_top_k={args.healthy_top_k}, error_buffer={args.error_buffer})")
    try:
        data = prepare_query_level_data(args.dataset, healthy_top_k=args.healthy_top_k, error_buffer=args.error_buffer)
    except Exception as e:
        # Fail fast with a clear message instead of a raw traceback for common user errors
        print(f"Error: failed to prepare query-level data: {e}")
        raise SystemExit(1)
    # Explicitly type these variables to satisfy static checkers and ensure
    # downstream code treats them as concrete mappings (not Optional).
    query_data: list[dict[str, Any]] = cast(list, data.get("query_data") or [])
    query_embs: dict[str, np.ndarray] = cast(dict, data.get("query_embs") or {})
    corpus_embs: dict[str, np.ndarray] = cast(dict, data.get("corpus_embs") or {})

    if not query_data:
        print("No query-level data found for this dataset. Exiting.")
        return

    # Build qrel map from initial query_data so we can recompute snapshots after projections
    qrel_map: dict[str, set[str]] = {str(q["query_id"]): set(q.get("relevant_set_ids") or q.get("R_q") or []) for q in query_data}

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

    using_local_store = bool(_local_store_dir())
    if not using_local_store:
        raise RuntimeError("PARASITIC_LOCAL_DB must be set to run in local mode.")

    try:
        # Begin iterative search for correction directions
        # Ensure corpus_current has a precise type for static analysis.
        corpus_current: dict[str, np.ndarray] = {k: np.array(v, dtype=np.float32) for k, v in corpus_embs.items()}
        corpus_original: dict[str, np.ndarray] = {k: np.array(v, dtype=np.float32) for k, v in corpus_embs.items()}

        incumbent_metrics: dict[str, Any] | None = None
        accepted_vectors: list[np.ndarray] = []

        # Outer iterations now correspond to applying accepted correction vectors
        # sequentially. Each outer iteration performs several random-restart
        # training runs; within each restart we optimize a candidate vector for
        # `args.epochs` epochs and retain the best checkpoint for that restart.
        for outer_it in range(1, args.iterations + 1):
            _progress_write("\n" + "=" * 50)
            _progress_write(f"Outer iteration {outer_it}: running {args.n_restarts} restarts, train epochs={args.epochs}")
            _print_query_split_summary(train_q, val_q, header="Current fixed query split snapshot:")

            # Track the best overall candidate across restarts for this outer iteration
            best_local = None
            best_local_v = None
            best_local_after = None

            # Precompute some small helpers
            first_key = next(iter(corpus_current))
            dim = int(corpus_current[first_key].shape[0])

            torch_mod = _try_torch_module()
            # Map search strictness into the healthy-weight behaviour used by the loss below.
            error_count = sum(1 for q in train_q if not bool(q.get("is_healthy")))
            healthy_weight_strategy, healthy_weight_value = map_search_strictness(
                getattr(args, "search_strictness", None), error_count=error_count
            )

            # Normalize gate acceptance strategy once for this outer iteration
            gate_choice = getattr(args, "gate_acceptance_strategy", None)
            if gate_choice is None:
                gate_choice = getattr(args, "acceptance_strategy", "exact")
            gate_choice = str(gate_choice).strip().lower()
            if gate_choice in ("strict", "exact"):
                accept_gate_fn = accept_candidate_by_exact_metrics
            else:
                # use cluster's top1_vs_degraded implementation from shared core
                accept_gate_fn = getattr(_core, "accept_candidate_by_top1_vs_degraded")

            for restart in range(args.n_restarts):
                _progress_write(f" Restart {restart+1}/{args.n_restarts}")

                # Create an initial candidate vector
                v0 = np.random.normal(size=(dim,)).astype(np.float32)
                if np.linalg.norm(v0) == 0:
                    continue
                v0 = v0 / (np.linalg.norm(v0) + 1e-12)

                # Per-restart best checkpoint
                restart_best_metrics = None
                restart_best_v = None
                restart_best_after = None

                patience = max(1, args.epochs // 4)
                no_improve = 0

                if torch_mod is not None:
                    # Gradient-based optimization using torch
                    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
                    v_param = torch_mod.nn.Parameter(torch_mod.tensor(v0, dtype=torch_mod.float32, device=device))
                    optim = torch_mod.optim.Adam([v_param], lr=1e-2)
                    pbar = _progress(range(1, args.epochs + 1), desc=f"Iter {outer_it} restart {restart+1}", unit="epoch", leave=False)

                    for ep in pbar:
                        # project corpus in torch so gradients flow to v_param
                        v_unit = v_param / (torch_mod.linalg.norm(v_param) + 1e-12)
                        # build corpus_after as dict[str, torch.Tensor]
                        corpus_after_torch: dict[str, Any] = {}
                        for k, vec in corpus_current.items():
                            x = _to_torch_vec(vec, device=device)
                            proj = torch_mod.dot(x, v_unit) * v_unit
                            corpus_after_torch[k] = x - proj

                        # compute training losses on train_q
                        try:
                            # Compute losses using core helpers. Pass healthy-weight strategy/value
                            _hw_strategy = str(healthy_weight_strategy or "").strip().lower()
                            _hw_value = float(healthy_weight_value or 1.0)
                            # Map weighting param for torch helper
                            if _hw_strategy in ("fixed", "error_scaled"):
                                hw_weighting = "preserve_healthy"
                                hw_value = float(_hw_value)
                            else:
                                hw_weighting = "per_query"
                                hw_value = 1.0

                            healthy_loss_t = compute_healthy_loss_torch(
                                train_q,
                                query_embs,
                                corpus_after_torch,
                                tau_h=float(args.tau_h),
                                weighting=hw_weighting,
                                preserve_healthy_weight=hw_value,
                            )
                            error_loss_t = compute_error_loss_torch(train_q, query_embs, corpus_after_torch, tau_e=float(args.tau_e))
                        except Exception:
                            # Fallback: compute losses in numpy and convert to scalar
                            healthy_loss_t = torch_mod.tensor(float(0.0), device=device)
                            error_loss_t = torch_mod.tensor(float(0.0), device=device)

                        loss = error_loss_t + healthy_loss_t + torch_mod.sum(v_param) * 0.0

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
                            metrics_full = evaluate_exact_full_corpus_before_after(val_q, query_embs, corpus_current, corpus_after)
                            candidate_global = _build_candidate_global(metrics_full)
                            if restart_best_metrics is None or accept_gate_fn(candidate_global, restart_best_metrics):
                                restart_best_metrics = candidate_global
                                restart_best_v = v_cur.copy()
                                restart_best_after = corpus_after
                                no_improve = 0

                        pbar.set_postfix(postfix)

                    pbar.close()
                else:
                    # No torch: use simple random-hillclimb over epochs (gradient-free)
                    v_base = v0.copy()
                    restart_best_loss = None
                    pbar = _progress(range(1, args.epochs + 1), desc=f"Iter {outer_it} restart {restart+1}", unit="epoch", leave=False)
                    for ep in pbar:
                        sigma = 0.5 * (1.0 - (ep / max(1, args.epochs)))
                        v_cand = v_base + np.random.normal(scale=sigma, size=v_base.shape).astype(np.float32)
                        if np.linalg.norm(v_cand) == 0:
                            continue
                        v_cand = v_cand / (np.linalg.norm(v_cand) + 1e-12)
                        corpus_after = _apply_orthogonal_projection_to_dict(corpus_current, v_cand)

                        # training losses on train_q
                        # Map healthy-weight strategy to compute_healthy_loss parameters
                        _hw_strategy = str(healthy_weight_strategy or "").strip().lower()
                        _hw_value = float(healthy_weight_value or 1.0)
                        if _hw_strategy in ("fixed", "error_scaled"):
                            healthy_val = compute_healthy_loss(
                                train_q, query_embs, corpus_after, tau_h=float(args.tau_h), weighting="preserve_healthy", preserve_healthy_weight=_hw_value
                            )
                        else:
                            # uniform or unknown -> default per-query weighting
                            healthy_val = compute_healthy_loss(train_q, query_embs, corpus_after, tau_h=float(args.tau_h))
                        error_val = compute_error_loss(train_q, query_embs, corpus_after, tau_e=float(args.tau_e))
                        loss_val = error_val + healthy_val

                        postfix = {
                            "h_loss": f"{healthy_val:.4f}",
                            "e_loss": f"{error_val:.4f}",
                        }

                        if restart_best_loss is None or loss_val < restart_best_loss:
                            restart_best_loss = loss_val
                            v_base = v_cand

                            if _should_run_validation(ep, args.epochs, int(args.eval_every)):
                                metrics_full = evaluate_exact_full_corpus_before_after(val_q, query_embs, corpus_current, corpus_after)
                                candidate_global = _build_candidate_global(metrics_full)
                                if restart_best_metrics is None or accept_gate_fn(candidate_global, restart_best_metrics):
                                    restart_best_metrics = candidate_global
                                    restart_best_v = v_base.copy()
                                    restart_best_after = corpus_after
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
                                _progress_write(
                                    f"  Validation epoch {ep}/{args.epochs}: {_format_exact_metric_summary(candidate_global)}"
                                )
                        else:
                            no_improve += 1

                        pbar.set_postfix(postfix)

                        if no_improve >= patience:
                            _progress_write(f"  Early stopping (no improvement) at epoch {ep}")
                            break

                    pbar.close()

                # End of restart: consider restart_best_metrics
                if restart_best_metrics is None:
                    _progress_write(f"  Restart {restart+1} produced no validation checkpoint.")
                    continue

                _progress_write(
                    f"  Restart {restart+1} best candidate: {_format_exact_metric_summary(restart_best_metrics)}"
                )

                # Select best across restarts using the configured acceptance gate
                if best_local is None or accept_gate_fn(restart_best_metrics, best_local):
                    best_local = restart_best_metrics
                    best_local_v = restart_best_v
                    best_local_after = restart_best_after

            # After all restarts in this outer iteration, decide acceptance vs incumbent
            # Refresh query-level supervision snapshot from qrels using current corpus
            # only after determining the best_local candidate for this outer iteration
            if best_local is None:
                _progress_write("No valid candidate found in this outer iteration. Stopping.")
                break

            # Final acceptance decision uses the normalized gate function
            accepted = accept_gate_fn(best_local, incumbent_metrics)
            if accepted:
                _progress_write(
                    f"Outer iteration {outer_it} ACCEPTED: {_explain_exact_acceptance_change(best_local, incumbent_metrics)}"
                )
                _progress_write(f"  Accepted metrics: {_format_exact_metric_summary(best_local)}")
                incumbent_metrics = best_local
                if best_local_v is not None:
                    accepted_vectors.append(best_local_v)

                # apply best_local_after to corpus_current and recompute query-level supervision
                if best_local_after is not None:
                    corpus_current = best_local_after
                query_data = _recompute_query_data_from_qrels(qrel_map, query_embs, corpus_current, healthy_top_k=int(args.healthy_top_k), error_buffer=int(args.error_buffer))
                train_q = filter_query_data_by_ids(query_data, train_query_ids)
                val_q = filter_query_data_by_ids(query_data, val_query_ids)
                _progress_write("After acceptance: refreshed query-level supervision within fixed split buckets.")
                _print_query_split_summary(train_q, val_q, header="Refreshed fixed query split snapshot:")
                _print_bench_triplets(train_q, val_q, f"iteration_{outer_it}")
                continue
            else:
                _progress_write(
                    f"Outer iteration {outer_it} REJECTED: {_explain_exact_acceptance_change(best_local, incumbent_metrics)}"
                )
                _progress_write(f"  Candidate metrics: {_format_exact_metric_summary(best_local)}")
                break

        _run_ablation_and_print_triplets(
            accepted_vectors,
            train_q, val_q,
            train_query_ids, val_query_ids,
            query_embs, corpus_original, qrel_map,
            healthy_top_k=int(args.healthy_top_k),
            error_buffer=int(args.error_buffer),
        )

        # Evaluation on the held-out validation split (used by the gate)
        print("\nValidation evaluation...")
        val_eval = evaluate_exact_full_corpus_before_after(val_q, query_embs, corpus_original, corpus_current)
        val_retrieval = val_eval.get("retrieval", {})
        val_retrieval_before = val_retrieval.get("before", {}) or {}
        val_retrieval_after = val_retrieval.get("after", {}) or {}
        val_global = val_eval.get("global", {})
        print("Validation retrieval metrics (before): " + ", ".join(f"{k}={float(v):.4f}" for k, v in val_retrieval_before.items()))
        print("Validation retrieval metrics (after) : " + ", ".join(f"{k}={float(v):.4f}" for k, v in val_retrieval_after.items()))
        print(
            f"Validation summary | Healthy degraded: {int(val_global.get('healthy_degraded_count', 0))} "
            f"| Error rank gain: {int(val_global.get('error_rank_gain_total', 0))} "
            f"| Error top1 fixes: {int(val_global.get('error_top1_fixed_count', 0))} "
            f"| Error top10 gains: {int(val_global.get('error_top10_gain_count', 0))}"
        )

        # Held-out test evaluation: one pass, never consulted by the gate
        if test_query_ids:
            print("\nTest evaluation (held-out, never seen by the gate)...")
            test_q = filter_query_data_by_ids(query_data, test_query_ids)
            test_eval = evaluate_exact_full_corpus_before_after(test_q, query_embs, corpus_original, corpus_current)
            test_retrieval = test_eval.get("retrieval", {})
            test_retrieval_before = test_retrieval.get("before", {}) or {}
            test_retrieval_after = test_retrieval.get("after", {}) or {}
            test_global = test_eval.get("global", {})
            print("Test retrieval metrics (before): " + ", ".join(f"{k}={float(v):.4f}" for k, v in test_retrieval_before.items()))
            print("Test retrieval metrics (after) : " + ", ".join(f"{k}={float(v):.4f}" for k, v in test_retrieval_after.items()))
            print(
                f"Test summary | Healthy degraded: {int(test_global.get('healthy_degraded_count', 0))} "
                f"| Error rank gain: {int(test_global.get('error_rank_gain_total', 0))} "
                f"| Error top1 fixes: {int(test_global.get('error_top1_fixed_count', 0))} "
                f"| Error top10 gains: {int(test_global.get('error_top10_gain_count', 0))}"
            )

        # Final evaluation across full query set
        print("\nFinal evaluation on full query set...")
        final_eval = evaluate_exact_full_corpus_before_after(query_data, query_embs, corpus_original, corpus_current)
        final_query_metrics = final_eval.get("query_metrics", {})
        final_global = final_eval.get("global", {})
        # compute final healthy count
        final_healthy = sum(1 for q in final_query_metrics.values() if q.get("healthy_after"))
        final_errors = int(final_global.get("query_count", 0)) - final_healthy

        # Rich final summary with exact rank movement statistics
        healthy_degraded = int(final_global.get("healthy_degraded_count", 0))
        error_rank_gain_total = int(final_global.get("error_rank_gain_total", 0))
        improved_error_count = int(final_global.get("improved_error_count", 0))
        error_top1_fixed_count = int(final_global.get("error_top1_fixed_count", 0))
        error_top10_gain_count = int(final_global.get("error_top10_gain_count", 0))

        print(
            f"Iterations accepted: {len(accepted_vectors)} | Final healthy: {final_healthy} | Final errors: {final_errors}\n"
            f"Healthy degraded count: {healthy_degraded} | Error rank gain total: {error_rank_gain_total} | Improved error count: {improved_error_count}\n"
            f"Error top1 fixes: {error_top1_fixed_count} | Error top10 gains: {error_top10_gain_count}"
        )

        # Print retrieval-level before/after summary if available
        retrieval = final_eval.get("retrieval", {})
        retrieval_before = retrieval.get("before", {}) or {}
        retrieval_after = retrieval.get("after", {}) or {}
        print("Retrieval metrics (before): " + ", ".join(f"{k}={float(v):.4f}" for k, v in retrieval_before.items()))
        print("Retrieval metrics (after) : " + ", ".join(f"{k}={float(v):.4f}" for k, v in retrieval_after.items()))

    finally:
        pass

if __name__ == "__main__":
    main()
