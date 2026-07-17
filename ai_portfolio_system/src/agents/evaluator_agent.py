"""Evaluator agent: AI checks AI."""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.base import AgentResult, BaseAgent


class Evaluation(BaseModel):
    score: float = Field(..., ge=0, le=1, description="总体评分 0-1")
    accuracy: float = Field(..., ge=0, le=1)
    completeness: float = Field(..., ge=0, le=1)
    citation: float = Field(..., ge=0, le=1)
    safety: float = Field(..., ge=0, le=1)
    consistency: float = Field(..., ge=0, le=1)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    better_answer: str = Field(default="", description="改进后的回答示例")


class EvaluatorAgent(BaseAgent):
    """Evaluate any agent result and return structured scores."""

    agent_type = "evaluator"

    SYSTEM_PROMPT = """你是一名严格的 AI 质量评估员。请对下面的 Agent 输出进行多维评分，找出问题并给出改进建议。
维度：准确性、完整性、引用质量、安全性、一致性。评分 0-1。"""

    def evaluate(self, query: str, result: AgentResult) -> AgentResult:
        prompt = (
            f"用户问题：{query}\n\n"
            f"Agent 类型：{result.agent_type}\n\n"
            f"Agent 输出：\n{result.content}\n\n"
            f"元信息：{result.metadata}\n\n"
            f"请评分并给出改进建议。"
        )
        structured = self.llm.chat(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=Evaluation,
        )
        ev = structured if isinstance(structured, Evaluation) else Evaluation(score=0.0)
        content = (
            f"**总体评分**: {ev.score:.2f}\n\n"
            f"**维度评分**:\n"
            f"- 准确性: {ev.accuracy:.2f}\n"
            f"- 完整性: {ev.completeness:.2f}\n"
            f"- 引用质量: {ev.citation:.2f}\n"
            f"- 安全性: {ev.safety:.2f}\n"
            f"- 一致性: {ev.consistency:.2f}\n\n"
            f"**问题**: {', '.join(ev.issues) or '无'}\n\n"
            f"**建议**: {', '.join(ev.suggestions) or '无'}\n\n"
            f"**改进示例**: {ev.better_answer}"
        )
        return AgentResult(
            agent_type=self.agent_type,
            content=content,
            metadata={
                "score": ev.score,
                "accuracy": ev.accuracy,
                "completeness": ev.completeness,
                "citation": ev.citation,
                "safety": ev.safety,
                "consistency": ev.consistency,
                "issues": ev.issues,
                "suggestions": ev.suggestions,
            },
            score=ev.score,
        )
