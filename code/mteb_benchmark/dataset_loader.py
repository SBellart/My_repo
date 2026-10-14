"""Dataset loading for MTEB public datasets.

This module provides a small library-oriented loader that fetches MTEB datasets
from HuggingFace and normalizes them into pandas DataFrames used by the
benchmark pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

from mteb_benchmark.config import DATASET_REGISTRY

LOGGER = logging.getLogger(__name__)
INSERT_BATCH_SIZE = 100


class DatasetLoader:
    """Load MTEB datasets from HuggingFace.

    Args:
        dataset_name: Registry key from ``DATASET_REGISTRY``.

    Raises:
        ValueError: If the dataset is not registered.
    """

    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.dataset_config: dict[str, Any] = DATASET_REGISTRY[dataset_name]
        self.hf_path = str(self.dataset_config["hf_path"])
        self.ignore_identical_ids = bool(
            self.dataset_config.get("ignore_identical_ids", False),
        )
        self.relevance_type = str(self.dataset_config.get("relevance_type", "binary"))

    def load_from_huggingface(self) -> dict[str, pd.DataFrame]:
        """Load corpus, queries, and qrels from HuggingFace.

        Returns:
            Normalized dataset payload with ``corpus``, ``queries``, and ``qrels``
            pandas DataFrames.
        """

        from datasets import load_dataset

        corpus_source = load_dataset(self.hf_path, self.dataset_config["corpus_subset"])
        queries_source = load_dataset(
            self.hf_path,
            self.dataset_config["queries_subset"],
        )
        qrels_source = load_dataset(self.hf_path, self.dataset_config["qrels_subset"])

        corpus_split = self._select_split(corpus_source, ["train"])
        queries_split = self._select_split(queries_source, ["queries", "train", "test"])
        qrels_split = self._select_split(qrels_source, ["train", "test", "validation"])

        corpus = pd.DataFrame(corpus_split).rename(
            columns={"_id": "external_id", "text": "doc_text", "title": "title"},
        )
        corpus = corpus[["external_id", "title", "doc_text"]].copy()
        corpus["external_id"] = corpus["external_id"].astype(str)
        corpus["title"] = corpus["title"].fillna("")
        corpus["doc_text"] = corpus["doc_text"].fillna("")

        queries = pd.DataFrame(queries_split).rename(
            columns={"_id": "external_id", "text": "query_text"},
        )
        queries = queries[["external_id", "query_text"]].copy()
        queries["external_id"] = queries["external_id"].astype(str)
        queries["query_text"] = queries["query_text"].fillna("")

        qrels = pd.DataFrame(qrels_split).rename(
            columns={"query-id": "query_id", "corpus-id": "corpus_id", "score": "score"},
        )
        qrels = qrels[["query_id", "corpus_id", "score"]].copy()
        qrels["query_id"] = qrels["query_id"].astype(str)
        qrels["corpus_id"] = qrels["corpus_id"].astype(str)
        qrels["score"] = qrels["score"].astype(int)
        qrels["is_relevant"] = qrels["score"] >= 1
        qrels = self._filter_identical_ids(qrels)

        LOGGER.info(
            "Loaded dataset %s from HuggingFace: corpus=%s queries=%s qrels=%s",
            self.dataset_name,
            len(corpus),
            len(queries),
            len(qrels),
        )
        return {
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
        }

    def _filter_identical_ids(self, qrels: pd.DataFrame) -> pd.DataFrame:
        if not self.ignore_identical_ids:
            return qrels
        return qrels[qrels["query_id"] != qrels["corpus_id"]].reset_index(drop=True)

    def _select_split(self, dataset_dict: Any, preferred_splits: list[str]) -> Any:
        available_splits = list(dataset_dict.keys())
        for split_name in preferred_splits:
            if split_name in dataset_dict:
                return dataset_dict[split_name]
        if not available_splits:
            raise ValueError(f"Dataset {self.dataset_name} has no available split")
        return dataset_dict[available_splits[0]]

