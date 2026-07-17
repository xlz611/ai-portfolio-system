"""Entry point for the AI portfolio system."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is importable when run as `python -m src.main` from project root.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config import get_config
from src.core.logging import get_logger
from src.orchestrator.router import Orchestrator
from src.ui.codex_tui import run_tui

logger = get_logger(__name__)


def run_cli_chat() -> None:
    """Simple REPL for quick testing."""
    orchestrator = Orchestrator()
    print("AI Portfolio System REPL")
    print("命令：/agent /model /quit")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            break
        if user_input.lower() == "/agent":
            for k, v in orchestrator.list_agents().items():
                print(f"- {k}: {v}")
            continue
        if user_input.lower() == "/model":
            cfg = get_config()
            print(f"当前模型: {cfg.get('llm.provider')}")
            continue
        result = orchestrator.process(user_input)
        print(f"\n[{result.agent_type}] 评分: {result.score or 'N/A'}  迭代: {result.iterations}")
        print(result.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Portfolio System")
    parser.add_argument("mode", choices=["tui", "chat", "info"], default="tui", nargs="?", help="运行模式")
    args = parser.parse_args()

    if args.mode == "tui":
        run_tui()
    elif args.mode == "chat":
        run_cli_chat()
    elif args.mode == "info":
        cfg = get_config()
        print(f"LLM provider: {cfg.get('llm.provider')}")
        print(f"Embeddings: {cfg.get('embeddings.model')}")
        print(f"Vector DB: {cfg.get('retrieval.vector.persist_dir')}")


if __name__ == "__main__":
    main()
