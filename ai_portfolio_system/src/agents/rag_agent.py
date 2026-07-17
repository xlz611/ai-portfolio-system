"""RAG Agent: answers questions with retrieved knowledge and citations."""
from __future__ import annotations

from src.agents.base import AgentResult, BaseAgent
from src.retrieval.constraints import MultiRetriever


class RAGAgent(BaseAgent):
    """Retrieve from knowledge base + web, then answer with citations."""

    agent_type = "rag"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retriever = MultiRetriever(self.config)
        self.system_prompt = self.SYSTEM_PROMPT

    SYSTEM_PROMPT = """你是一个求职知识库助手。请根据提供的参考资料回答问题，并在答案中标注引用来源。
如果参考资料不足，请明确说明。回答要求：准确、简洁、中文。"""

    def run(self, query: str, top_k: int | None = None, extra_instructions: str = "") -> AgentResult:
        items = self.retriever.search(query, top_k=top_k)
        context = self._build_context(items)
        system = self.system_prompt
        if extra_instructions:
            system += f"\n\n本轮优化要求：{extra_instructions}"
        user_prompt = f"问题：{query}\n\n参考资料：\n{context}\n\n请回答上面的问题，并标注引用来源。"
        answer = self._chat(system, user_prompt)
        sources = [self._source_label(item) for item in items]
        return AgentResult(
            agent_type=self.agent_type,
            content=answer,
            metadata={"query": query, "sources": sources, "retrieved_count": len(items)},
        )

    def _build_context(self, items: list) -> str:
        parts = []
        for i, item in enumerate(items, 1):
            meta = item.get("metadata", {})
            if item["source"] == "vector":
                label = f"[知识库 {i}]"
            else:
                label = f"[Web {i}] {meta.get('url', '')}"
            parts.append(f"{label}\n{item['content']}\n")
        return "\n".join(parts) if parts else "（未找到相关参考资料）"

    def _source_label(self, item: dict) -> str:
        if item["source"] == "vector":
            return f"知识库 (score={item['score']})"
        return f"Web: {item.get('metadata', {}).get('url', 'unknown')}"
