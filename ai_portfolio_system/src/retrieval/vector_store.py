"""Lightweight SQLite-backed vector store for local demos.

The default implementation uses SQLite + numpy for cosine similarity. To use Chroma
instead, install chromadb and set VECTOR_STORE_BACKEND=chroma.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import Config, get_config
from src.core.embeddings import get_embeddings
from src.core.logging import get_logger

logger = get_logger(__name__)

BACKEND = os.getenv("VECTOR_STORE_BACKEND", "sqlite").lower()

if BACKEND == "chroma":
    try:
        import chromadb
    except Exception:  # pragma: no cover
        chromadb = None  # type: ignore
else:
    chromadb = None


class VectorStore:
    """Persistent vector store with SQLite fallback or Chroma backend."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        persist_dir = Path(self.config.get("retrieval.vector.persist_dir", "./data/chroma"))
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = self.config.get("retrieval.vector.collection_name", "portfolio")

        if BACKEND == "chroma" and chromadb is not None:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chroma"
        else:
            self._db_path = persist_dir / "vectors.db"
            self._backend = "sqlite"
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                document TEXT,
                metadata TEXT,
                embedding BLOB
            )
            """
        )
        conn.commit()
        conn.close()

    def add_documents(self, documents: list[str], metadatas: list[dict[str, Any]] | None = None, ids: list[str] | None = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in documents]
        if ids is None:
            ids = [self._hash(doc) for doc in documents]
        embeddings = get_embeddings(self.config).embed(documents)

        if self._backend == "chroma":
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            logger.info("Added %d documents to Chroma store", len(documents))
            return

        conn = sqlite3.connect(self._db_path)
        for id_, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT OR REPLACE INTO vectors(id, document, metadata, embedding) VALUES (?, ?, ?, ?)",
                (id_, doc, json.dumps(meta, ensure_ascii=False), emb_blob),
            )
        conn.commit()
        conn.close()
        logger.info("Added %d documents to SQLite vector store", len(documents))

    def query(self, text: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.config.get("retrieval.vector.top_k", 5)
        query_embedding = get_embeddings(self.config).embed_query(text)
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        if self._backend == "chroma":
            return self._query_chroma(q, k)
        return self._query_sqlite(q, k)

    def _query_chroma(self, q: np.ndarray, k: int) -> list[dict[str, Any]]:
        results = self._collection.query(
            query_embeddings=[q.tolist()], n_results=k, include=["documents", "metadatas", "distances"]
        )
        items = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            items.append({
                "source": "vector",
                "content": doc,
                "metadata": results["metadatas"][0][i] or {},
                "score": round(similarity, 4),
            })
        return items

    def _query_sqlite(self, q: np.ndarray, k: int) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT id, document, metadata, embedding FROM vectors").fetchall()
        conn.close()

        scored = []
        for id_, doc, meta_json, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            # Resize if dimensions differ (e.g. TF-IDF vs sentence-transformers).
            if len(emb) < len(q):
                emb = np.pad(emb, (0, len(q) - len(emb)))
            elif len(emb) > len(q):
                emb = emb[: len(q)]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            similarity = float(np.dot(q, emb))
            scored.append((similarity, doc, meta_json))

        scored.sort(key=lambda x: x[0], reverse=True)
        items = []
        for score, doc, meta_json in scored[:k]:
            items.append({
                "source": "vector",
                "content": doc,
                "metadata": json.loads(meta_json) if meta_json else {},
                "score": round(score, 4),
            })
        return items

    def count(self) -> int:
        if self._backend == "chroma":
            return self._collection.count()
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        conn.close()
        return row[0] if row else 0

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
