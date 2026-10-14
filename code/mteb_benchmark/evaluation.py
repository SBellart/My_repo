from __future__ import annotations

import logging

import numpy as np
import pytrec_eval

from mteb_benchmark.config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

_FRIENDLY_TO_TREC = {
    "ndcg@10": "ndcg_cut_10",
    "recall@10": "recall_10",
    "map@10": "map_cut_10",
}
_TREC_TO_FRIENDLY = {value: key for key, value in _FRIENDLY_TO_TREC.items()}


class EvaluationModule:
    """Compute retrieval rankings and benchmark metrics.

    Args:
        qrels: Relevance judgments in ``pytrec_eval`` qrels format.
    """

    def __init__(self, qrels: dict[str, dict[str, int]]) -> None:
        """Initialize the evaluation module.

        Args:
            qrels: Mapping of query IDs to document relevance labels.
        """
        self.qrels = qrels

    def compute_ranking(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        query_ids: list[str],
        corpus_ids: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, dict[str, float]]:
        """Compute a top-k cosine-similarity ranking run.

        Args:
            query_embeddings: Query embedding matrix of shape ``(Q, D)``.
            corpus_embeddings: Corpus embedding matrix of shape ``(C, D)``.
            query_ids: Query IDs aligned with ``query_embeddings``.
            corpus_ids: Corpus IDs aligned with ``corpus_embeddings``.
            top_k: Number of documents to keep per query.

        Returns:
            Ranking run in ``pytrec_eval`` format.

        Raises:
            ValueError: If shapes or IDs do not align.
        """
        self._validate_inputs(
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            top_k=top_k,
        )

        q_norm = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10
        )
        c_norm = corpus_embeddings / (
            np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        )
        similarity_matrix = q_norm @ c_norm.T

        top_k_limited = min(top_k, len(corpus_ids))
        run: dict[str, dict[str, float]] = {}

        for query_index, query_id in enumerate(query_ids):
            scores = similarity_matrix[query_index]
            if top_k_limited >= len(corpus_ids):
                ranked_indices = np.argsort(scores)[::-1]
            else:
                candidate_indices = np.argpartition(scores, -top_k_limited)[
                    -top_k_limited:
                ]
                ranked_indices = candidate_indices[
                    np.argsort(scores[candidate_indices])[::-1]
                ]
            run[query_id] = {
                corpus_ids[doc_index]: float(scores[doc_index])
                for doc_index in ranked_indices
            }

        logger.debug(
            "Computed ranking run for %s queries against %s documents",
            len(query_ids),
            len(corpus_ids),
        )
        return run

    def evaluate(
        self,
        run: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate a run with ``pytrec_eval`` and average per-query scores.

        Args:
            run: Ranking run in ``pytrec_eval`` format.
            metrics: Optional friendly metric names or raw ``pytrec_eval`` names.

        Returns:
            Mean metric values with friendly keys.
        """
        metric_names = self._resolve_metrics(metrics)
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, set(metric_names))
        normalized_run = {query_id: run.get(query_id, {}) for query_id in self.qrels}
        per_query_results = evaluator.evaluate(normalized_run)

        aggregated_results: dict[str, float] = {}
        query_count = max(len(per_query_results), 1)

        for metric_name in metric_names:
            metric_total = sum(
                query_metrics.get(metric_name, 0.0)
                for query_metrics in per_query_results.values()
            )
            aggregated_results[_TREC_TO_FRIENDLY[metric_name]] = metric_total / query_count

        logger.debug("Evaluated run with metrics: %s", metric_names)
        return aggregated_results

    def evaluate_before_after(
        self,
        query_embeddings: np.ndarray,
        corpus_before: np.ndarray,
        corpus_after: np.ndarray,
        query_ids: list[str],
        corpus_ids: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, dict[str, float]]:
        """Compare retrieval metrics before and after a corpus transformation.

        Args:
            query_embeddings: Query embedding matrix.
            corpus_before: Baseline corpus embeddings.
            corpus_after: Updated corpus embeddings.
            query_ids: Query IDs aligned with query embeddings.
            corpus_ids: Corpus IDs aligned with corpus embeddings.
            top_k: Number of documents to keep per query.

        Returns:
            Nested mapping with ``before``, ``after``, and ``improvement`` scores.
        """
        before_run = self.compute_ranking(
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_before,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            top_k=top_k,
        )
        after_run = self.compute_ranking(
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_after,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            top_k=top_k,
        )
        before_results = self.evaluate(before_run)
        after_results = self.evaluate(after_run)
        improvement = {
            metric_name: after_results[metric_name] - before_results[metric_name]
            for metric_name in before_results
        }
        return {
            "before": before_results,
            "after": after_results,
            "improvement": improvement,
        }

    def format_results(self, results: dict[str, float]) -> str:
        """Format metric results as an aligned plain-text table.

        Args:
            results: Mapping of metric names to scores.

        Returns:
            Multi-line string table.
        """
        metric_header = "Metric"
        value_header = "Value"
        metric_width = max(len(metric_header), *(len(metric) for metric in results))
        value_strings = {metric: f"{value:.4f}" for metric, value in results.items()}
        value_width = max(len(value_header), *(len(value) for value in value_strings.values()))

        lines = [
            f"{metric_header:<{metric_width}}  {value_header:>{value_width}}",
            f"{'-' * metric_width}  {'-' * value_width}",
        ]
        for metric_name, value in value_strings.items():
            lines.append(f"{metric_name:<{metric_width}}  {value:>{value_width}}")
        return "\n".join(lines)

    def compute_metrics_after_projection(
        self,
        corpus_embeddings: np.ndarray,
        concept_directions: list[np.ndarray],
        query_embeddings: np.ndarray,
        query_ids: list[str],
        corpus_ids: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, float]:
        """Apply orthogonal projection to corpus embeddings then evaluate.

        For each concept direction, remove the concept subspace from all corpus
        embeddings using ``orthogonal_projection_batch`` from ``projection_utils``,
        then compute retrieval metrics on the projected corpus.

        Args:
            corpus_embeddings: Baseline corpus embedding matrix of shape ``(C, D)``.
            concept_directions: List of concept direction vectors each of shape ``(D,)``.
            query_embeddings: Query embedding matrix of shape ``(Q, D)``.
            query_ids: Query IDs aligned with ``query_embeddings``.
            corpus_ids: Corpus IDs aligned with ``corpus_embeddings``.
            top_k: Number of documents to keep per query.

        Returns:
            Mean metric values after projection with friendly keys.
        """
        from projection_utils import orthogonal_projection_batch

        projected = corpus_embeddings.copy().astype(np.float32)
        for direction in concept_directions:
            projected = orthogonal_projection_batch(projected, direction)

        run = self.compute_ranking(
            query_embeddings=query_embeddings,
            corpus_embeddings=projected,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            top_k=top_k,
        )
        return self.evaluate(run)

    def _resolve_metrics(self, metrics: list[str] | None) -> list[str]:
        if metrics is None:
            return list(_FRIENDLY_TO_TREC.values())

        resolved_metrics: list[str] = []
        for metric_name in metrics:
            if metric_name in _FRIENDLY_TO_TREC:
                resolved_metrics.append(_FRIENDLY_TO_TREC[metric_name])
                continue
            if metric_name in _TREC_TO_FRIENDLY:
                resolved_metrics.append(metric_name)
                continue
            raise ValueError(f"Unsupported metric requested: {metric_name}")
        return resolved_metrics

    def _validate_inputs(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        query_ids: list[str],
        corpus_ids: list[str],
        top_k: int,
    ) -> None:
        if query_embeddings.ndim != 2 or corpus_embeddings.ndim != 2:
            raise ValueError("query_embeddings and corpus_embeddings must be 2D arrays")
        if query_embeddings.shape[1] != corpus_embeddings.shape[1]:
            raise ValueError("query and corpus embeddings must share the same dimension")
        if query_embeddings.shape[0] != len(query_ids):
            raise ValueError("query_ids length must match query_embeddings rows")
        if corpus_embeddings.shape[0] != len(corpus_ids):
            raise ValueError("corpus_ids length must match corpus_embeddings rows")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if len(corpus_ids) == 0:
            raise ValueError("corpus_ids must not be empty")
