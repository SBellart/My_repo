"""Embedding backends for local reproducibility runs.

Provides two provider paths: ``sentence_transformers`` (fully local, no key)
and ``openai`` (OpenAI-compatible API, ``OPENAI_API_KEY`` or ``--api-key``).
``--api-base-url`` lets you point the OpenAI backend at any OpenAI-compatible
endpoint.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _retry(call: Callable[[], list[list[float]]], attempts: int = 5) -> list[list[float]]:
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


class SentenceTransformersEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, batch_size=batch_size, show_progress_bar=False),
            dtype=np.float32,
        )


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None, api_base_url: str | None = None) -> None:
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or --api-key is required for the openai backend")
        self.model_name = model
        self._client = OpenAI(api_key=api_key, base_url=api_base_url)

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        vectors: list[list[float]] = []
        for batch in _batched(texts, batch_size):
            safe_batch = [t if t.strip() else " " for t in batch]

            def _call() -> list[list[float]]:
                response = self._client.embeddings.create(
                    model=self.model_name, input=safe_batch
                )
                return [item.embedding for item in response.data]

            vectors.extend(_retry(_call))
        return np.asarray(vectors, dtype=np.float32)


BACKENDS = {
    "sentence_transformers": SentenceTransformersEmbedder,
    "openai": OpenAIEmbedder,
}


def get_local_embedder(backend: str, model: str | None = None, api_key: str | None = None, api_base_url: str | None = None):
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    cls = BACKENDS[backend]
    if backend == "openai":
        return cls(model, api_key=api_key, api_base_url=api_base_url) if model else cls(api_key=api_key, api_base_url=api_base_url)
    return cls(model) if model else cls()
