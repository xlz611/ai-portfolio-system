"""Shared tools: SQL, web search, file reader."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import get_config


@dataclass
class ToolResult:
    name: str
    output: str
    success: bool
    metadata: dict[str, Any] | None = None


class SQLTool:
    """Read-only SQL executor with safety checks."""

    def __init__(self, db_path: str | None = None) -> None:
        cfg = get_config()
        self.db_path = db_path or cfg.get("sql.db_path", "./data/job.db")
        self.allowed_tables = set(cfg.get("sql.allowed_tables", []))
        self.dangerous_keywords = [kw.upper() for kw in cfg.get("sql.dangerous_keywords", [])]

    def _is_safe(self, sql: str) -> tuple[bool, str]:
        upper = sql.upper().strip()
        for kw in self.dangerous_keywords:
            if kw in upper:
                return False, f"检测到危险关键词: {kw}"
        # Only allow querying allowed tables
        for table in self.allowed_tables:
            if table.upper() in upper:
                return True, ""
        if not self.allowed_tables:
            return True, ""
        return False, "SQL 未涉及允许的表"

    def run(self, sql: str) -> ToolResult:
        safe, reason = self._is_safe(sql)
        if not safe:
            return ToolResult("sql", f"[拦截] {reason}", False, {"sql": sql})
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            rows = conn.execute(sql).fetchall()
            columns = [desc[0] for desc in conn.execute(sql).description] if rows else []
            conn.close()
            return ToolResult(
                "sql",
                self._format(rows, columns),
                True,
                {"sql": sql, "rows": len(rows)},
            )
        except Exception as e:
            return ToolResult("sql", f"[错误] {e}", False, {"sql": sql})

    def _format(self, rows: list, columns: list[str]) -> str:
        if not rows:
            return "查询成功，结果为空。"
        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))
        for row in rows[:20]:
            lines.append(" | ".join(str(c) for c in row))
        if len(rows) > 20:
            lines.append(f"... 共 {len(rows)} 行，仅展示前 20 行")
        return "\n".join(lines)


class FileReaderTool:
    """Read allowed text files under data/."""

    def __init__(self, base_dir: str | None = None) -> None:
        base = base_dir or "./data"
        self.base_dir = Path(base).resolve()

    def run(self, path: str) -> ToolResult:
        target = Path(path).resolve()
        try:
            if not str(target).startswith(str(self.base_dir)):
                return ToolResult("file_reader", "[拦截] 只能读取 data/ 目录下的文件", False)
            content = target.read_text(encoding="utf-8", errors="ignore")
            return ToolResult("file_reader", content, True, {"path": str(target)})
        except Exception as e:
            return ToolResult("file_reader", f"[错误] {e}", False, {"path": path})
