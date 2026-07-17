"""Local embedding wrapper with sentence-transformers primary and TF-IDF fallback."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import Config, get_config
from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None


class LocalEmbeddings:
    """ sentence-transformers based embeddings with TF-IDF fallback."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        model_name = self.config.get("embeddings.model", "sentence-transformers/all-MiniLM-L6-v2")
        device = self.config.get("embeddings.device", "cpu")
        self._model_name = model_name
        self._device = device
        self._st_model: Any | None = None
        self._tfidf: Any | None = None
        self._tfidf_fitted = False
        self._embedding_dim = 384  # all-MiniLM-L6-v2 output dimension

        if SentenceTransformer is not None:
            try:
                cache_dir = Path(__file__).resolve().parents[2] / "data" / "models"
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir))
                self._st_model = SentenceTransformer(model_name, device=device)
                logger.info("Loaded sentence-transformers model: %s", model_name)
                return
            except Exception as e:
                logger.warning("Failed to load sentence-transformers (%s). Falling back to TF-IDF.", e)
        else:
            logger.warning("sentence-transformers not installed. Using TF-IDF fallback.")

        if TfidfVectorizer is not None:
            # char_wb works better for mixed Chinese/English text without external tokenizers.
            self._tfidf = TfidfVectorizer(max_features=self._embedding_dim, analyzer="char_wb", stop_words=None)
        else:
            raise RuntimeError("Neither sentence-transformers nor scikit-learn is available.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._st_model is not None:
            return self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
        return self._tfidf_embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _tfidf_embed(self, texts: list[str]) -> list[list[float]]:
        if not self._tfidf_fitted:
            self._tfidf.fit(texts)
            self._tfidf_fitted = True
        matrix = self._tfidf.transform(texts).toarray()
        # Pad or truncate to a fixed dimension for Chroma compatibility.
        result = []
        for row in matrix:
            padded = np.zeros(self._embedding_dim, dtype=np.float32)
            length = min(len(row), self._embedding_dim)
            padded[:length] = row[:length]
            # Normalize to unit length for cosine similarity.
            norm = np.linalg.norm(padded)
            if norm > 0:
                padded = padded / norm
            result.append(padded.tolist())
        return result


_embedding_singleton: LocalEmbeddings | None = None


def get_embeddings(config: Config | None = None) -> LocalEmbeddings:
    global _embedding_singleton
    if _embedding_singleton is None or config is not None:
        _embedding_singleton = LocalEmbeddings(config)
    return _embedding_singleton
