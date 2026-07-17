"""Web search with DuckDuckGo and constraint filtering."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from src.core.config import Config, get_config
from src.core.logging import get_logger

logger = get_logger(__name__)


try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover
    DDGS = None


class WebRetriever:
    """Search the web, then apply domain/time/recency constraints."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.enabled = self.config.get("retrieval.web.enabled", True)
        self.max_results = self.config.get("retrieval.web.max_results", 5)
        self.timeout = self.config.get("retrieval.web.timeout", 10)
        self.recency_days = self.config.get("retrieval.web.recency_days", 730)
        self.whitelist = [self._normalize_domain(d) for d in self.config.get("retrieval.web.whitelist", [])]
        self.blacklist = [self._normalize_domain(d) for d in self.config.get("retrieval.web.blacklist", [])]

    def search(self, query: str) -> list[dict[str, Any]]:
        if not self.enabled or DDGS is None:
            logger.warning("Web search disabled or duckduckgo_search not installed.")
            return []
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw = ddgs.text(query, max_results=self.max_results, backend="api")
        except Exception as e:
            logger.warning("Web search failed: %s", e)
            return []

        items = []
        for r in raw:
            item = {
                "source": "web",
                "title": r.get("title", ""),
                "content": r.get("body", ""),
                "url": r.get("href", ""),
                "metadata": {"title": r.get("title", ""), "url": r.get("href", ""), "date": ""},
                "score": 0.6,  # default relevance before filtering
            }
            if self._passes_constraints(item):
                items.append(item)
        return items

    def _passes_constraints(self, item: dict[str, Any]) -> bool:
        url = item.get("url", "")
        domain = self._normalize_domain(url)

        if self.blacklist and any(b in domain for b in self.blacklist):
            return False

        if self.whitelist and not any(domain == w or domain.endswith("." + w) for w in self.whitelist):
            logger.debug("Filtered by whitelist: %s", url)
            return False

        # DuckDuckGo doesn't always return dates; if we had one, we would check recency.
        return True

    def _normalize_domain(self, url_or_domain: str) -> str:
        if "//" in url_or_domain:
            parsed = urlparse(url_or_domain)
            domain = parsed.netloc
        else:
            domain = url_or_domain
        return domain.lower().replace("www.", "")
