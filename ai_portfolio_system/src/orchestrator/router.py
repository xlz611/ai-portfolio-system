"""Orchestrator routes user queries to agents and combines results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.base import AgentResult
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.match_agent import MatchAgent
from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent
from src.core.config import Config, get_config
from src.core.llm import LLM, get_llm
from src.core.logging import get_logger
from src.tuning.self_attention_loop import SelfAttentionTuner

logger = get_logger(__name__)


@dataclass
class ConversationState:
    """Tracks conversation history and last results."""

    history: list[dict[str, str]] = field(default_factory=list)
    last_eval: AgentResult | None = None


class Orchestrator:
    """Routes user intent to the right agent(s)."""

    def __init__(self, llm: LLM | None = None, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.llm = llm or get_llm(self.config)
        self.rag = RAGAgent(self.llm, self.config)
        self.sql = SQLAgent(self.llm, self.config)
        self.match = MatchAgent(self.llm, self.config)
        self.evaluator = EvaluatorAgent(self.llm, self.config)
        self.tuner = SelfAttentionTuner(self.llm, self.config)
        self.state = ConversationState()

    def process(self, user_input: str, enable_tuning: bool = True) -> AgentResult:
        """Main entry: classify intent, dispatch, optionally self-improve."""
        intent = self._classify(user_input)
        logger.info("Intent classified as: %s", intent)

        if intent == "sql":
            result = self.sql.run(user_input)
        elif intent == "match":
            resume, jd = self._extract_resume_jd(user_input)
            result = self.match.run(resume, jd)
        elif intent == "evaluate":
            if self.state.last_eval is None:
                result = AgentResult(agent_type="orchestrator", content="还没有可评估的上一条结果。请先发一个问题。")
            else:
                result = self.evaluator.evaluate(self.state.last_eval.metadata.get("query", ""), self.state.last_eval)
        else:
            # RAG with optional self-attention tuning
            if enable_tuning:
                result = self.tuner.run(self.rag, user_input)
            else:
                result = self.rag.run(user_input)

        self.state.history.append({"role": "user", "content": user_input})
        self.state.history.append({"role": "assistant", "content": result.content})
        self.state.last_eval = result

        # Always attach a quick evaluation to non-evaluator outputs when tuning is on.
        if enable_tuning and intent not in ("evaluate",):
            eval_result = self.evaluator.evaluate(user_input, result)
            result.score = eval_result.score
            result.metadata["evaluation"] = eval_result.metadata

        return result

    def _classify(self, text: str) -> str:
        """Keyword-based intent classification, robust to missing API keys."""
        lower = text.lower()
        keywords = {
            "sql": ["sql", "select", "查询", "数据库", "表", "多少", "统计", "top", "平均"],
            "match": ["匹配", "简历", "jd", "岗位", "求职", "缺什么", "差距"],
            "evaluate": ["评估", "打分", "评分", "检查"],
        }
        for intent, words in keywords.items():
            if any(w in lower for w in words):
                return intent
        return "rag"

    def _extract_resume_jd(self, text: str) -> tuple[str, str]:
        """Naively split a message that contains both resume and JD."""
        markers = ["---jd---", "---JD---", "【JD】", "岗位描述：", "jd:"]
        lower = text.lower()
        for marker in markers:
            if marker in lower:
                parts = text.split(marker, 1)
                return parts[0].strip(), parts[1].strip()
        # If no marker, treat first half as resume and second half as JD.
        mid = len(text) // 2
        return text[:mid].strip(), text[mid:].strip()

    def list_agents(self) -> dict[str, str]:
        return {
            "rag": "RAG 求职知识库助手",
            "sql": "SQL 数据分析 Agent",
            "match": "简历/JD 匹配助手",
            "evaluate": "AI 评估器",
        }
