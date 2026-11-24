"""Where documents may go under docs/ (see docs/README.md). Standard library only.

Two users:
- Claude Code runs this file as a PreToolUse hook (.claude/settings.json) before every
  Write/Edit/NotebookEdit. A file that breaks the layout is refused (exit code 2) and the
  reason goes back to Claude.
- tests/test_docs_layout.py imports check() and runs it on every file tracked by git, so
  a move made another way (mv, git mv, a script) fails `uv run pytest`.

The layout:
    docs/README.md                      the index (the rules for Claude are in CLAUDE.md)
    docs/proposal/...                   the proposal
    docs/presentation/...               the proposal presentation (slides, script, toy experiment)
    docs/reports/<report>/...           one folder per report (q1, q2_q3, data, train, results_tables, ...)
    docs/private/...                    personal notes and translations, git-ignored
Old places that must not come back: reports/, slides/, docs/q1_report/, docs/results_tables/.
"""

import json
import os
import re
import sys
from pathlib import PurePosixPath

TOP_FILES = {"README.md"}
TOP_FOLDERS = {"proposal", "presentation", "reports", "private"}
REPORT_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")  # q1, q2_q3, data, train, results_tables
MOVED = {  # old place -> new place
    "reports": "docs/private",
    "slides": "docs/presentation",
    "docs/q1_report": "docs/reports/q1",
    "docs/results_tables": "docs/reports/results_tables",
}


def check(path: str) -> str | None:
    """Why this repository-relative path breaks the layout, or None if it is fine."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    for old, new in MOVED.items():
        old_parts = PurePosixPath(old).parts
        if parts[: len(old_parts)] == old_parts:
            return f"{old}/ was moved to {new}/ (see docs/README.md); write there instead."
    if not parts or parts[0] != "docs":
        return None
    if len(parts) == 2:
        if parts[1] in TOP_FILES:
            return None
        return (f"no loose files in docs/: put {parts[1]} in docs/proposal/, docs/presentation/, "
                "docs/reports/<report>/ or docs/private/ (see docs/README.md).")
    if parts[1] not in TOP_FOLDERS:
        return (f"docs/{parts[1]}/ is not part of the layout. Use docs/proposal/, docs/presentation/, "
                "docs/reports/<report>/ or docs/private/ (see docs/README.md).")
    if parts[1] == "reports":
        if len(parts) == 3:
            return "no loose files in docs/reports/: each report has its own folder, docs/reports/<report>/."
        if not REPORT_NAME.match(parts[2]):
            return f"report folder names are lowercase with underscores (q1, q2_q3): not {parts[2]!r}."
    return None


def main() -> int:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input") or {}
    target = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not target:
        return 0
    root = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    path = os.path.relpath(os.path.abspath(os.path.join(root, target)), os.path.abspath(root))
    if path.startswith(".."):
        return 0  # outside the repository
    reason = check(path)
    if reason is None:
        return 0
    print(f"Docs layout: {path} refused. {reason} If the layout itself should change, update "
          ".claude/hooks/docs_layout.py, docs/README.md and CLAUDE.md together, and ask the user first.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
