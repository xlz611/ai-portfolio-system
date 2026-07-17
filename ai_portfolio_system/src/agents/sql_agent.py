"""SQL Agent: natural language to safe SQL, execute, explain."""
from __future__ import annotations

from src.agents.base import AgentResult, BaseAgent
from src.core.tools import SQLTool


class SQLAgent(BaseAgent):
    """Generate and execute read-only SQL with safety checks."""

    agent_type = "sql"

    SYSTEM_PROMPT = """你是一个 SQL 数据分析助手。只生成安全的 SELECT 查询，不修改数据。
可用表如下：jobs(岗位), candidates(候选人), interviews(面试记录), skills(技能)。
如果用户的问题可能产生危险 SQL，请拒绝并说明原因。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sql_tool = SQLTool()

    def run(self, query: str) -> AgentResult:
        schema = self._get_schema()
        prompt = f"数据库表结构：\n{schema}\n\n用户问题：{query}\n\n请直接生成一条 SQL（只读 SELECT），不要做解释。"
        sql = self._chat(self.SYSTEM_PROMPT, prompt).strip()
        # Extract code block if present
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].strip()

        result = self.sql_tool.run(sql)
        explanation_prompt = (
            f"SQL: {sql}\n执行结果：\n{result.output}\n\n请用中文解释这条 SQL 解决了什么问题，以及结果的含义。"
        )
        explanation = self._chat(self.SYSTEM_PROMPT, explanation_prompt) if result.success else "SQL 执行被拦截，无需解释。"
        content = f"**生成的 SQL**:\n```sql\n{sql}\n```\n\n**执行结果**:\n{result.output}\n\n**解释**:\n{explanation}"
        return AgentResult(
            agent_type=self.agent_type,
            content=content,
            metadata={"sql": sql, "success": result.success, "rows": result.metadata.get("rows", 0) if result.metadata else 0},
        )

    def _get_schema(self) -> str:
        return """jobs(id, title, company, location, category, salary_min, salary_max, description)
candidates(id, name, major, city, skills, expected_salary)
interviews(id, candidate_id, job_id, round, score, feedback)
skills(id, name, category, popularity)"""
