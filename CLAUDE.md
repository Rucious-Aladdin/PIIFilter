# PIIGuard: notes for Claude

A benchmark of PII detectors (Presidio, BERT, GLiNER) for a filter that masks personal data
before a prompt reaches a language model (course project, CSED490E). The three research
questions, the scripts and the results are in `README.md`; the documents are in `docs/`.

## Environment

- uv and Python 3.12: `uv sync`, then `uv run python experiments/<script>.py`.
- Tests: `uv run pytest` (the whole suite must pass before a commit).

## Never commit or print

- `.env`: API keys. Never print, copy or quote its values.
- `data/` (datasets and test sets, see the README on licenses), `models/` (fine-tuned weights),
  `docs/private/` (personal notes). All are git-ignored; keep them so.

## Documents: the layout is fixed

`docs/README.md` is the index. Documents go here and nowhere else:

| Folder | For |
| --- | --- |
| `docs/proposal/` | the proposal |
| `docs/presentation/` | the proposal presentation: slides, script, toy experiment frames |
| `docs/reports/<name>/` | one folder per report: `q1/`, `q2_q3/`, `results_tables/`, ... |
| `docs/private/` | personal notes, drafts and translations (git-ignored) |

- Put a new document in the matching folder. A new report gets a new `docs/reports/<name>/`
  folder (lowercase, words joined by `_`) and a line in `docs/README.md`.
- Do not move, rename or delete existing documents or folders, and do not create other folders
  for documents (in `docs/` or at the repository root), unless the user asks for it.
- A hook (`.claude/hooks/docs_layout.py`, run from `.claude/settings.json`) refuses files written
  in the wrong place. Do not get around it with shell commands. If the user wants the layout
  changed, update the hook, `docs/README.md` and this file together, fix every link, and run
  the tests.
- Each report belongs to its authors (`docs/reports/q1/` is Louis's). Edit one only when asked.
- Generated files are rebuilt, not edited by hand: `docs/reports/results_tables/tables/` by
  `experiments/results_tables.py`, `results/` by the experiment scripts.
- After any change in `docs/` or to a link: `uv run pytest tests/test_docs_layout.py`.
