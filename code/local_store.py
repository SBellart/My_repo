"""Local NoSQL (TinyDB) document store for reviewer-side reproducibility.

One TinyDB JSON file per dataset alias holds four tables: ``meta``, 
``corpus``, ``queries``, ``qrels``.
Embeddings are stored as base64-encoded little-endian float32 buffers to keep
files compact and loading exact.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from tinydb import TinyDB
    from tinydb.middlewares import CachingMiddleware
    from tinydb.storages import JSONStorage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tinydb is required for the local store: pip install tinydb"
    ) from exc


def encode_vector(vector: Iterable[float]) -> str:
    array = np.asarray(list(vector), dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("ascii")


def decode_vector(payload: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload), dtype=np.float32).copy()


class LocalStore:
    """Directory of TinyDB files, one per dataset alias."""

    def __init__(self, db_dir: str | Path) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def _db_path(self, alias: str) -> Path:
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in alias)
        return self.db_dir / f"{safe}.json"

    def open(self, alias: str) -> TinyDB:
        return TinyDB(
            self._db_path(alias),
            storage=CachingMiddleware(JSONStorage),
        )

    def dataset_exists(self, alias: str) -> bool:
        return self._db_path(alias).exists()

    def table_count(self, alias: str, table: str) -> int:
        if not self.dataset_exists(alias):
            return 0
        with self.open(alias) as db:
            return len(db.table(table))

    def write_meta(self, alias: str, meta: dict[str, Any]) -> None:
        with self.open(alias) as db:
            table = db.table("meta")
            table.truncate()
            table.insert(dict(meta))

    def read_meta(self, alias: str) -> dict[str, Any] | None:
        if not self.dataset_exists(alias):
            return None
        with self.open(alias) as db:
            rows = db.table("meta").all()
        return rows[0] if rows else None

    def insert_documents(
        self,
        alias: str,
        table: str,
        rows: list[dict[str, Any]],
        replace: bool = False,
    ) -> int:
        with self.open(alias) as db:
            target = db.table(table)
            if replace:
                target.truncate()
            target.insert_multiple(rows)
            return len(target)

    def load_embeddings(self, alias: str, table: str) -> dict[str, np.ndarray]:
        with self.open(alias) as db:
            rows = db.table(table).all()
        embeddings: dict[str, np.ndarray] = {}
        for row in rows:
            payload = row.get("embedding")
            if payload:
                embeddings[str(row["external_id"])] = decode_vector(payload)
        return embeddings

    def load_qrels(self, alias: str, min_score: int = 1) -> list[tuple[str, str, int]]:
        with self.open(alias) as db:
            rows = db.table("qrels").all()
        return [
            (str(r["query_id"]), str(r["corpus_id"]), int(r["score"]))
            for r in rows
            if int(r.get("score", 0)) >= min_score
        ]
