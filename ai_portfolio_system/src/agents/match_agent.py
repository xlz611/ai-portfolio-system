"""Resume / JD matching agent."""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.base import AgentResult, BaseAgent


class MatchAnalysis(BaseModel):
    match_score: float = Field(..., ge=0, le=1, description="匹配度 0-1")
    matched_skills: list[str] = Field(default_factory=list, description="已匹配技能")
    missing_skills: list[str] = Field(default_factory=list, description="缺失技能")
    learning_priority: list[str] = Field(default_factory=list, description="按优先级排序的补课建议")
    project_suggestions: list[str] = Field(default_factory=list, description="项目改写/补充建议")
    summary: str = Field(..., description="一段中文总结")


class MatchAgent(BaseAgent):
    """Analyze resume against JD and produce structured advice."""

    agent_type = "match"

    SYSTEM_PROMPT = """你是一名资深 HR 与 AI 面试官。请根据提供的简历和岗位描述（JD），给出结构化的匹配分析。
要求：真实、不夸大；只基于输入文本中的事实；输出 JSON 结构。"""

    def run(self, resume: str, jd: str) -> AgentResult:
        prompt = f"简历：\n{resume}\n\n岗位描述（JD）：\n{jd}\n\n请分析匹配度并给出建议。"
        structured = self.llm.chat(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=MatchAnalysis,
        )
        analysis = structured if isinstance(structured, MatchAnalysis) else MatchAnalysis()
        content = (
            f"匹配度：{analysis.match_score:.0%}\n\n"
            f"已匹配技能：{', '.join(analysis.matched_skills) or '无'}\n\n"
            f"缺失技能：{', '.join(analysis.missing_skills) or '无'}\n\n"
            f"补课优先级：\n" + "\n".join(f"- {p}" for p in analysis.learning_priority) + "\n\n"
            f"项目改写建议：\n" + "\n".join(f"- {s}" for s in analysis.project_suggestions) + "\n\n"
            f"总结：{analysis.summary}"
        )
        return AgentResult(
            agent_type=self.agent_type,
            content=content,
            metadata={
                "match_score": analysis.match_score,
                "matched_skills": analysis.matched_skills,
                "missing_skills": analysis.missing_skills,
            },
        )
