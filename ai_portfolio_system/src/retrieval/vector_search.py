"""Vector search with optional reranking and thresholding."""
from __future__ import annotations

from typing import Any

from src.core.config import Config, get_config
from src.retrieval.vector_store import VectorStore


class VectorRetriever:
    """Semantic search over the vector store."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.store = VectorStore(config)
        self.threshold = self.config.get("retrieval.vector.similarity_threshold", 0.5)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        items = self.store.query(query, top_k=top_k)
        filtered = [item for item in items if item["score"] >= self.threshold]
        return filtered

    def add(self, documents: list[str], metadatas: list[dict[str, Any]] | None = None) -> None:
        self.store.add_documents(documents, metadatas=metadatas)
