"""Generate a TUI screenshot for documentation."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.codex_tui import CodexTUI


async def main() -> None:
    app = CodexTUI()
    async with app.run_test() as pilot:
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        pilot.app.save_screenshot(str(project_root / "data" / "tui_screenshot.svg"))
    print("Screenshot saved to data/tui_screenshot.svg")


if __name__ == "__main__":
    asyncio.run(main())
