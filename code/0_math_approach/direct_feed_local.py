#!/usr/bin/env python3
"""Reviewer-side local dataset ingestion: HuggingFace -> embed -> TinyDB.

This fetches a registered MTEB dataset from HuggingFace, embeds corpus and queries
with a locally-runnable embedder, and writes the result into a local TinyDB document
store (``local_store.py``).

If the target dataset alias already exists in the local store, ingestion is
skipped by default; pass ``--force`` to re-embed and overwrite it.

Usage:
    python 0_math_approach/direct_feed_local.py --dataset scifact
    python 0_math_approach/direct_feed_local.py --dataset scifact \\
        --embedder-backend openai --embedder-model text-embedding-3-small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from local_embedders import get_local_embedder
from local_store import LocalStore, encode_vector
from mteb_benchmark.config import DATASET_REGISTRY
from mteb_benchmark.dataset_loader import DatasetLoader


def build_alias(dataset: str, backend: str, model: str) -> str:
    safe_model = "".join(ch if ch.isalnum() else "-" for ch in model.lower()).strip("-")
    safe_backend = "".join(ch if ch.isalnum() else "-" for ch in backend.lower()).strip("-")
    return f"{dataset}__emb__{safe_backend}__{safe_model}"


def default_model(backend: str) -> str:
    return {
        "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
        "openai": "text-embedding-3-small",
    }[backend]


def feed_local(
    dataset: str,
    backend: str,
    model: str,
    store: LocalStore,
    force: bool = False,
    batch_size: int = 64,
    api_key: str | None = None,
    api_base_url: str | None = None,
) -> str:
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {dataset!r}; see mteb_benchmark/config.py")

    alias = build_alias(dataset, backend, model)
    if store.dataset_exists(alias) and not force:
        print(f"Alias {alias!r} already present in local store, skipping (use --force to overwrite).")
        return alias

    print(f"Loading {dataset!r} from HuggingFace ({DATASET_REGISTRY[dataset]['hf_path']}) ...")
    loader = DatasetLoader(dataset)
    frames = loader.load_from_huggingface()
    corpus_df, queries_df, qrels_df = frames["corpus"], frames["queries"], frames["qrels"]
    print(f"  corpus={len(corpus_df)} queries={len(queries_df)} qrels={len(qrels_df)}")

    embedder = get_local_embedder(backend, model, api_key=api_key, api_base_url=api_base_url)
    print(f"Embedding with backend={backend!r} model={model!r} ...")

    corpus_texts = (corpus_df["title"].fillna("") + " " + corpus_df["doc_text"].fillna("")).str.strip().tolist()
    corpus_vectors = embedder.embed(corpus_texts, batch_size=batch_size)
    corpus_rows = [
        {
            "external_id": str(ext_id),
            "title": str(title),
            "doc_text": str(text),
            "embedding": encode_vector(vec),
        }
        for ext_id, title, text, vec in zip(
            corpus_df["external_id"], corpus_df["title"], corpus_df["doc_text"], corpus_vectors
        )
    ]

    query_texts = queries_df["query_text"].fillna("").tolist()
    query_vectors = embedder.embed(query_texts, batch_size=batch_size)
    query_rows = [
        {"external_id": str(ext_id), "query_text": str(text), "embedding": encode_vector(vec)}
        for ext_id, text, vec in zip(queries_df["external_id"], queries_df["query_text"], query_vectors)
    ]

    qrels_rows = [
        {"query_id": str(qid), "corpus_id": str(cid), "score": int(score)}
        for qid, cid, score in zip(
            qrels_df["query_id"], qrels_df["corpus_id"], qrels_df["score"]
        )
    ]

    store.insert_documents(alias, "corpus", corpus_rows, replace=True)
    store.insert_documents(alias, "queries", query_rows, replace=True)
    store.insert_documents(alias, "qrels", qrels_rows, replace=True)
    store.write_meta(
        alias,
        {
            "dataset": dataset,
            "backend": backend,
            "model": model,
            "corpus_count": len(corpus_rows),
            "query_count": len(query_rows),
            "qrels_count": len(qrels_rows),
            "embedding_dimension": int(corpus_vectors.shape[1]) if len(corpus_rows) else 0,
        },
    )
    print(f"Wrote alias {alias!r}: corpus={len(corpus_rows)} queries={len(query_rows)} qrels={len(qrels_rows)}")
    return alias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument(
        "--embedder-backend",
        default="sentence_transformers",
        choices=["sentence_transformers", "openai"],
    )
    parser.add_argument("--embedder-model", default=None)
    parser.add_argument("--db-dir", default=str(REPO_ROOT / "local_db"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-key", default=None, help="Optional API key override for public endpoints")
    parser.add_argument("--api-base-url", default=None, help="Optional base URL override (e.g. for OpenAI-compatible endpoints)")
    args = parser.parse_args()

    model = args.embedder_model or default_model(args.embedder_backend)
    store = LocalStore(args.db_dir)
    feed_local(
        args.dataset,
        args.embedder_backend,
        model,
        store,
        force=args.force,
        batch_size=args.batch_size,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
