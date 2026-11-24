"""The documents stay where docs/README.md says, and every link to them works.

The rules are those of the Claude Code hook (.claude/hooks/docs_layout.py), which refuses a
badly placed file as it is written; these tests catch what the hook cannot see: files moved
with mv or git mv, links left pointing to an old place, an index that misses a folder.
"""

import importlib.util
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("docs_layout", ROOT / ".claude" / "hooks" / "docs_layout.py")
docs_layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs_layout)

LINK = re.compile(r"\]\(([^)\s]+)\)")


def tracked(*paths: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.split()


@pytest.mark.parametrize("path", ["docs/reports/q2_q3/q2_q3_report_en.md", "docs/proposal/x.docx",
                                  "docs/reports/data/data_generation.pdf", "docs/reports/train/training_procedure.tex",
                                  "docs/presentation/toy_experiment/preview.tex", "docs/private/notes/a.md",
                                  "docs/README.md", "experiments/score.py"])
def test_allowed(path):
    assert docs_layout.check(path) is None


@pytest.mark.parametrize("path", ["docs/notes.md", "docs/notes/a.md", "docs/data/x.pdf", "docs/train/x.tex", "docs/reports/a.md", "docs/reports/Q4/a.md",
                                  "docs/q1_report/q1_report_en.md", "docs/results_tables/x.tex",
                                  "reports/src/step0.md", "slides/toy_experiment/preview.tex"])
def test_refused(path):
    assert docs_layout.check(path) is not None


def test_tracked_files_follow_the_layout():
    wrong = {p: docs_layout.check(p) for p in tracked() if docs_layout.check(p)}
    assert not wrong, wrong


def test_nothing_private_is_tracked():
    assert tracked("docs/private") == []


def test_the_index_lists_every_folder():
    index = (ROOT / "docs" / "README.md").read_text()
    folders = {f"{p.split('/')[1]}/" for p in tracked("docs") if p.count("/") >= 2}
    folders |= {f"reports/{p.split('/')[2]}/" for p in tracked("docs/reports")}
    missing = sorted(f for f in folders | {"private/"} if f not in index)
    assert not missing, f"add these folders to docs/README.md: {missing}"


def test_links_to_repository_files_work():
    broken = []
    for doc in ["README.md", "CLAUDE.md", *[p for p in tracked("docs") if p.endswith(".md")]]:
        path = ROOT / doc
        for target in LINK.findall(path.read_text()):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue  # web links and anchors
            file = unquote(target.split("#")[0])
            if file and not (path.parent / file).exists():
                broken.append(f"{doc} -> {target}")
    assert not broken, broken
