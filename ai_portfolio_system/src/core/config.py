"""Configuration loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Simple YAML config with environment overrides."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config.yaml"
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as f:
            self._raw: dict[str, Any] = yaml.safe_load(f) or {}
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Allow LLM_PROVIDER, OPENAI_API_KEY, OLLAMA_MODEL to override config."""
        if os.getenv("LLM_PROVIDER"):
            self._raw["llm"]["provider"] = os.getenv("LLM_PROVIDER")
        if os.getenv("OPENAI_API_KEY"):
            self._raw["llm"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("OLLAMA_MODEL"):
            self._raw["llm"]["ollama"]["model"] = os.getenv("OLLAMA_MODEL")
        if os.getenv("EMBEDDINGS_MODEL"):
            self._raw["embeddings"]["model"] = os.getenv("EMBEDDINGS_MODEL")

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access, e.g. config.get('llm.provider')."""
        parts = key.split(".")
        value = self._raw
        for part in parts:
            if not isinstance(value, dict):
                return default
            value = value.get(part, default)
            if value is None:
                return default
        return value

    def section(self, key: str) -> dict[str, Any]:
        return self.get(key, {}) or {}


# Global singleton for convenience, but tests can create new instances.
_config: Config | None = None


def get_config(path: str | Path | None = None) -> Config:
    global _config
    if _config is None or path is not None:
        _config = Config(path)
    return _config
