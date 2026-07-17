"""Quick verification script: imports + basic agent run."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent
from src.agents.match_agent import MatchAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.orchestrator.router import Orchestrator
from src.scripts.ingest import seed_vector_store, seed_sqlite_db


def main() -> None:
    print("=== Ingesting demo data ===")
    seed_vector_store()
    seed_sqlite_db()

    print("\n=== Testing RAG Agent ===")
    rag = RAGAgent()
    rag_result = rag.run("AI 产品经理需要哪些技能？")
    print(rag_result.content)
    print(f"Sources: {rag_result.metadata.get('sources', [])}")

    print("\n=== Testing SQL Agent ===")
    sql = SQLAgent()
    sql_result = sql.run("查询所有 AI 岗位")
    print(sql_result.content)

    print("\n=== Testing Match Agent ===")
    resume = (project_root / "data" / "resumes" / "sample_resume.txt").read_text(encoding="utf-8")
    jd = (project_root / "data" / "jds" / "sample_jd.txt").read_text(encoding="utf-8")
    match = MatchAgent()
    match_result = match.run(resume, jd)
    print(match_result.content)

    print("\n=== Testing Evaluator ===")
    evaluator = EvaluatorAgent()
    eval_result = evaluator.evaluate("AI 产品经理需要哪些技能？", rag_result)
    print(eval_result.content)

    print("\n=== Testing Orchestrator ===")
    orch = Orchestrator()
    final = orch.process("你好")
    print(final.content)
    print(f"Score: {final.score}, Iterations: {final.iterations}")

    print("\n=== Verification complete ===")


if __name__ == "__main__":
    main()
