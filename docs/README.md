# Documents

Every document of the project, by stage. The code and how to run it are in the
[repository README](../README.md); the files written by the experiment scripts are in `results/`.

| Folder | Content |
| --- | --- |
| [proposal/](proposal/) | [PIIGuard_Proposal_Merged.docx](proposal/PIIGuard_Proposal_Merged.docx): the plan (questions, datasets, labels, metrics, Appendix B on fine-tuning) |
| [presentation/](presentation/) | The proposal presentation of October 22: [slides](presentation/PII_Filter_long.pdf), [script](presentation/PII_Filter_presentation_script.docx) (English, with Korean and French), and [toy_experiment/](presentation/toy_experiment/), the Beamer frames of the toy experiment slides with a [preview](presentation/toy_experiment/preview.pdf) |
| [reports/q1/](reports/q1/) | Question 1, off-the-shelf detectors on clean text: [en](reports/q1/q1_report_en.md), [fr](reports/q1/q1_report_fr.md), [ko](reports/q1/q1_report_ko.md) |
| [reports/q2_q3/](reports/q2_q3/) | Questions 2 and 3, noisy text and fine-tuning BERT, with the noise, the data, the training recipe, the results and the tests: [en](reports/q2_q3/q2_q3_report_en.md), [fr](reports/q2_q3/q2_q3_report_fr.md), [ko](reports/q2_q3/q2_q3_report_ko.md) |
| [reports/data/](reports/data/) | [How the data is made](reports/data/data_generation.pdf): sources, label mapping, clean test sets, the noise algorithm, the fine-tuning data, and the checksum of every data file (LaTeX source and PDF) |
| [reports/train/](reports/train/) | [How BERT is fine-tuned](reports/train/training_procedure.pdf): labels and tags, settings, the six runs, outputs, reproduction (LaTeX source and PDF) |
| [reports/results_tables/](reports/results_tables/) | [Three pages of key tables](reports/results_tables/results_tables.pdf) for every model, questions 1 to 3. `tables/` is written by `experiments/results_tables.py` |
| `private/` | Personal notes, drafts and translations. Git-ignored: nothing in it is shared |

## Rules

1. Every document goes into one of these folders. No loose files in `docs/` (except this index)
   or in `docs/reports/`.
2. A new report gets its own folder, `docs/reports/<name>/` (lowercase, words joined by `_`),
   and a line in the table above. Its translations sit next to it: `<name>_report_en.md`,
   `_fr.md`, `_ko.md`, as for `q1/` and `q2_q3/`.
3. Existing files and folders are not moved or renamed: the README, the scripts and the other
   reports link to them. If the layout has to change, change `.claude/hooks/docs_layout.py`, this
   index and `CLAUDE.md` together, then fix the links.
4. Notes, drafts and translations that are not for the team go to `docs/private/`.
5. Generated files (`reports/results_tables/tables/`, the PDFs built from LaTeX) are rebuilt with
   their script or LaTeX source, not edited by hand.

Two checks keep this layout: a Claude Code hook (`.claude/settings.json`) refuses a file written
in the wrong place, and `uv run pytest tests/test_docs_layout.py` fails if a tracked file breaks
the rules, if a folder is missing from this index, or if a link in the READMEs or the reports
points to nothing.
