"""Constraint engine that merges and filters multi-source retrieval results."""
from __future__ import annotations

from typing import Any

from src.core.config import Config, get_config
from src.retrieval.vector_search import VectorRetriever
from src.retrieval.web_search import WebRetriever


class MultiRetriever:
    """Combine vector and web retrieval, then apply global constraints."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.vector = VectorRetriever(config)
        self.web = WebRetriever(config)
        self.max_total = self.config.get("retrieval.web.max_results", 5) + self.config.get("retrieval.vector.top_k", 5)
        self.similarity_threshold = self.config.get("retrieval.vector.similarity_threshold", 0.5)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        vector_items = self.vector.search(query, top_k=top_k)
        web_items = self.web.search(query)
        merged = vector_items + web_items
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[: self.max_total]
