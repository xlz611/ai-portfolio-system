"""LLM abstraction with mock, OpenAI, and Ollama backends."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None

from src.core.config import Config, get_config

T = TypeVar("T", bound=BaseModel)


class LLM(ABC):
    """Unified LLM interface."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], response_format: type[T] | None = None) -> str | T:
        """Call the model. Return string or Pydantic object if response_format provided."""
        ...

    def _format_messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


class MockLLM(LLM):
    """Deterministic mock LLM for offline demos."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config)
        self._calls = 0

    def chat(self, messages: list[dict[str, str]], response_format: type[T] | None = None) -> str | T:
        self._calls += 1
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")

        # If a structured format is requested, attempt to return a sensible default.
        if response_format:
            return self._structured_response(system, user, response_format)

        return self._text_response(system, user)

    def _text_response(self, system: str, user: str) -> str:
        lower = user.lower()
        if "sql" in lower or "查询" in lower or "select" in lower:
            return "我已根据你的问题生成安全 SQL：\n```sql\nSELECT title, company, location FROM jobs WHERE category = 'AI' LIMIT 5;\n```\n该查询只读取 allowed_tables 中的 jobs 表，未涉及任何写操作。"
        if "匹配" in lower or "简历" in lower or "jd" in lower:
            return "简历与 JD 匹配分析：\n- 匹配度：78%\n- 已具备：Python、LangChain、RAG 基础\n- 缺失项：SQL 调优、生产部署经验、A/B 测试\n- 建议：补充 SQL 索引与向量数据库性能优化项目。"
        if "你好" in lower or "hello" in lower:
            return "你好！我是 AI 求职项目组合助手。你可以问我求职知识、查询岗位数据、或上传简历/JD 进行匹配。"
        return (
            "这是 Mock LLM 的默认回答。为了获得更真实的效果，请设置 LLM_PROVIDER=openai 或 ollama，"
            "并配置对应的 API key / 本地模型。当前问题摘要：\n" + user[:200]
        )

    def _structured_response(self, system: str, user: str, response_format: type[T]) -> T:
        """Build a default instance of the Pydantic model."""
        schema = response_format.model_json_schema()
        defaults = self._schema_defaults(schema)
        return response_format(**defaults)

    def _schema_defaults(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively build a default JSON object matching the schema."""
        result: dict[str, Any] = {}
        props = schema.get("properties", {})
        for key, spec in props.items():
            if spec.get("type") == "string":
                result[key] = "（Mock LLM 未提供具体内容）"
            elif spec.get("type") == "number":
                result[key] = 0.0
            elif spec.get("type") == "integer":
                result[key] = 0
            elif spec.get("type") == "boolean":
                result[key] = False
            elif spec.get("type") == "array":
                result[key] = []
            elif spec.get("type") == "object":
                result[key] = self._schema_defaults(spec)
        return result


class OpenAILLM(LLM):
    """OpenAI-compatible LLM."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config)
        cfg = self.config.section("llm")["openai"]
        api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI provider selected but OPENAI_API_KEY not set.")
        model = cfg.get("model", "gpt-4o-mini")
        base_url = cfg.get("base_url")
        kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "temperature": 0.3}
        if base_url:
            kwargs["base_url"] = base_url
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed. Run: pip install langchain-openai")
        self._client = ChatOpenAI(**kwargs)

    def chat(self, messages: list[dict[str, str]], response_format: type[T] | None = None) -> str | T:
        if response_format:
            structured = self._client.with_structured_output(response_format)
            return structured.invoke(messages)
        return str(self._client.invoke(messages).content)


class OllamaLLM(LLM):
    """Ollama local LLM."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config)
        cfg = self.config.section("llm")["ollama"]
        model = cfg.get("model", "qwen2.5")
        base_url = cfg.get("base_url", "http://localhost:11434")
        if ChatOllama is None:
            raise RuntimeError("langchain-community not installed. Run: pip install langchain-community")
        self._client = ChatOllama(model=model, base_url=base_url, temperature=0.3)

    def chat(self, messages: list[dict[str, str]], response_format: type[T] | None = None) -> str | T:
        if response_format:
            structured = self._client.with_structured_output(response_format)
            return structured.invoke(messages)
        return str(self._client.invoke(messages).content)


def get_llm(config: Config | None = None) -> LLM:
    """Factory: returns configured LLM implementation."""
    cfg = config or get_config()
    provider = cfg.get("llm.provider", "mock")
    if provider == "openai":
        return OpenAILLM(cfg)
    if provider == "ollama":
        return OllamaLLM(cfg)
    return MockLLM(cfg)
