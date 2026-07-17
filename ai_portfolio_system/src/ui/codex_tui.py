"""Codex-style terminal UI using Textual."""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

from src.core.config import get_config
from src.orchestrator.router import Orchestrator


class CodexTUI(App):
    """A Codex-like terminal popup for the AI portfolio system."""

    TITLE = "AI Portfolio System"
    CSS = """
    Screen {
        background: #0d1117;
        color: #c9d1d9;
    }
    #title {
        color: #58a6ff;
        text-style: bold;
    }
    #model_info {
        color: #8b949e;
    }
    #chat_log {
        border: none;
        background: #0d1117;
        color: #c9d1d9;
        padding: 1 2;
    }
    #input {
        border: none;
        background: #161b22;
        color: #c9d1d9;
        padding: 1 2;
    }
    .user {
        color: #7ee787;
    }
    .assistant {
        color: #c9d1d9;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, orchestrator: Orchestrator | None = None) -> None:
        super().__init__()
        self.orchestrator = orchestrator or Orchestrator()
        self.config = get_config()
        self.model_name = self.config.get("llm.provider", "mock") + ":" + self.config.get(f"llm.{self.config.get('llm.provider')}.model", "default")

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("AI Portfolio System  —  多智能体求职助手", id="title")
            yield Static(f"model: {self.model_name}  directory: {self.config.path.parent}", id="model_info")
            yield RichLog(id="chat_log", highlight=True, markup=True)
            yield Input(placeholder="输入问题，或 /agent /model /quit", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat_log", RichLog).write("Tip: 输入问题进行对话，/agent 查看可用 Agent，/quit 退出。")
        self.query_one("#input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        input_widget = self.query_one("#input", Input)
        text = event.value.strip()
        input_widget.value = ""
        if not text:
            return

        log = self.query_one("#chat_log", RichLog)
        log.write(f"[user] > {text}")

        if text.startswith("/"):
            self._handle_command(text, log)
            return

        try:
            result = self.orchestrator.process(text)
            log.write(f"[assistant] {result.content}")
            if result.score is not None:
                log.write(f"[meta] 评分: {result.score:.2f}  迭代: {result.iterations}")
        except Exception as e:
            log.write(f"[error] {e}")

    def _handle_command(self, cmd: str, log: RichLog) -> None:
        parts = cmd.lower().split()
        if not parts:
            return
        if parts[0] in ("/quit", "/exit"):
            self.exit()
        elif parts[0] == "/agent":
            agents = self.orchestrator.list_agents()
            for k, v in agents.items():
                log.write(f"- {k}: {v}")
        elif parts[0] == "/model":
            log.write(f"当前模型: {self.model_name}")
        else:
            log.write(f"未知命令: {cmd}")


def run_tui() -> None:
    app = CodexTUI()
    app.run()
