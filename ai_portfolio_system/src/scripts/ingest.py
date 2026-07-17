"""Seed demo data into vector DB and SQLite DB."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.core.config import get_config
from src.retrieval.vector_store import VectorStore


KNOWLEDGE_DOCS = [
    """AI 产品经理岗位要求：熟悉大模型应用、RAG、Agent 设计；需要具备需求分析、PRD 撰写、数据分析能力；
加分项：有 LLM 微调、Prompt Engineering、A/B 测试经验。""",
    """RAG 检索增强生成：结合向量数据库与外部知识库，解决大模型幻觉问题。关键技术包括文档分块、Embedding、
相似度检索、重排序、引用溯源。""",
    """SQL 数据分析工程师面试要点：窗口函数、索引优化、子查询、JOIN 性能、数据清洗。常用数据库：MySQL、PostgreSQL、SQLite。""",
    """Python 后端开发技能栈：FastAPI、Flask、SQLAlchemy、Redis、Docker、Git、CI/CD。""",
    """求职简历项目建议：项目要「少而深」，有量化指标；技术栈明确；展示可复现的 GitHub 链接。""",
]


JOBS_DATA = [
    ("AI 产品经理实习生", "某大厂", "北京", "AI", 300, 500, "负责大模型应用产品需求分析与落地"),
    ("数据分析实习生", "某银行", "上海", "数据", 250, 400, "SQL 数据分析与报表开发"),
    ("Python 后端开发实习生", "某电商", "深圳", "后端", 200, 350, "FastAPI 后端接口开发与维护"),
    ("大模型应用工程师", "某 AI 公司", "杭州", "AI", 400, 700, "RAG 与 Agent 系统开发"),
]

CANDIDATES_DATA = [
    ("张三", "人工智能", "北京", "Python,LangChain,RAG", 300),
    ("李四", "数据科学", "上海", "SQL,Python,Pandas", 250),
]

SKILLS_DATA = [
    ("Python", "编程", 0.9),
    ("RAG", "AI", 0.8),
    ("LangChain", "AI", 0.7),
    ("SQL", "数据", 0.85),
    ("FastAPI", "后端", 0.6),
    ("Docker", "工程", 0.5),
]


def seed_vector_store() -> None:
    print("Seeding vector store...")
    store = VectorStore()
    store.add_documents(
        KNOWLEDGE_DOCS,
        metadatas=[{"topic": "求职", "source": "demo"} for _ in KNOWLEDGE_DOCS],
    )
    print(f"Vector store count: {store.count()}")


def seed_sqlite_db() -> None:
    db_path = Path(get_config().get("sql.db_path", "./data/job.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            company TEXT,
            location TEXT,
            category TEXT,
            salary_min INTEGER,
            salary_max INTEGER,
            description TEXT
        );
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            major TEXT,
            city TEXT,
            skills TEXT,
            expected_salary INTEGER
        );
        CREATE TABLE interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            job_id INTEGER,
            round INTEGER,
            score REAL,
            feedback TEXT
        );
        CREATE TABLE skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            popularity REAL
        );
        """
    )
    conn.executemany(
        "INSERT INTO jobs(title, company, location, category, salary_min, salary_max, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
        JOBS_DATA,
    )
    conn.executemany(
        "INSERT INTO candidates(name, major, city, skills, expected_salary) VALUES (?, ?, ?, ?, ?)",
        CANDIDATES_DATA,
    )
    conn.executemany(
        "INSERT INTO skills(name, category, popularity) VALUES (?, ?, ?)",
        SKILLS_DATA,
    )
    conn.commit()
    conn.close()
    print(f"Seeded SQLite DB: {db_path}")


def main() -> None:
    seed_vector_store()
    seed_sqlite_db()
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
