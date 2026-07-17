"""Base agent with shared utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config import Config, get_config
from src.core.llm import LLM, get_llm
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResult:
    agent_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    iterations: int = 0

    def to_markdown(self) -> str:
        header = f"### [{self.agent_type}]"
        meta = ""
        if self.score is not None:
            meta += f"\n**评分**: {self.score:.2f}"
        if self.iterations:
            meta += f"\n**迭代轮次**: {self.iterations}"
        if self.metadata:
            for k, v in self.metadata.items():
                if k in ("sources", "citations"):
                    continue
                meta += f"\n- {k}: {v}"
        return f"{header}{meta}\n\n{self.content}\n"


class BaseAgent:
    """Common agent base."""

    agent_type: str = "base"

    def __init__(self, llm: LLM | None = None, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.llm = llm or get_llm(self.config)

    def _chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return str(self.llm.chat(messages))
