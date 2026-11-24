"""Sensitivity analyses for question 1 (S1, S2 and S4 of the Q1 report; S3 is in
experiments/contamination.py).

    uv run python experiments/sensitivity.py

- S1: micro F2 and recall under looser matching rules (benchmark/analysis.py, match_counts);
- S2: GLiNER's micro F2 as a function of its cut-off, from a run at cut-off 0.05 (the script
  stops, without writing anything, if that run is missing or incomplete). The cut-off used
  in the benchmark stays 0.35;
- S4: micro F2 on documents of at most 384 words (one GLiNER window) and on longer ones.

Writes results/analysis/sensitivity.md.
"""

import argparse
import json

import numpy as np

from baselines.windows import WORD_RE
from benchmark.analysis import DETECTORS, example_counts, load_comparison, match_counts, scores
from benchmark.data import ANALYSIS_DIR, ROOT, SOURCES, TEST_DIR, read_jsonl, read_predictions
from benchmark.scoring import f_score, fmt

RULES = ["strict", "same_span", "overlap_same_label", "overlap_any_label"]
LOW_CUTOFF_RUN = ROOT / "data" / "analysis" / "predictions" / "gliner_cutoff_0.05"
LOW_CUTOFF_COMMAND = ("uv run python experiments/run_benchmark.py gliner --threshold 0.05 "
                      "--out data/analysis/predictions --name gliner_cutoff_0.05")


def check_low_cutoff_run() -> None:
    """Stop if the GLiNER run at cut-off 0.05 (needed by S2) is missing or incomplete."""
    for dataset in SOURCES:
        path = LOW_CUTOFF_RUN / dataset / "clean.jsonl"
        done = set()
        if path.exists():
            # A line without its end of line was cut by an interruption: it does not count.
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            done = {json.loads(line)["id"] for line in lines if line.endswith("\n")}
        missing = {ex.id for ex in read_jsonl(TEST_DIR / f"{dataset}_clean.jsonl")} - done
        if missing:
            raise SystemExit(f"The GLiNER run at cut-off 0.05 lacks {len(missing)} examples of {dataset}. "
                             f"Nothing was written. Run first:\n  {LOW_CUTOFF_COMMAND}")


def f2_and_recall(gold_found: int, gold: int, found_ok: int, found: int) -> tuple:
    p = found_ok / found if found else None
    r = gold_found / gold if gold else None
    return f_score(p, r, 2), r


def s1() -> list[str]:
    lines = ["## S1: looser matching rules", "",
             "Micro F2 (recall in brackets). same_span: same start and end, any label; overlap: at least one "
             "shared character.", ""]
    for comparison in ("A", "A-dates"):
        lines += [f"### Comparison {comparison}", "", "| Dataset | Detector | " + " | ".join(RULES) + " |", "|---" * 6 + "|"]
        for dataset in SOURCES:
            examples, labels, predictions = load_comparison(dataset, comparison)
            for d in DETECTORS:
                cells = []
                for rule in RULES:
                    total = np.sum([match_counts(ex.text, ex.spans, predictions[d][ex.id], labels, rule) for ex in examples], axis=0)
                    f2, r = f2_and_recall(*total)
                    cells.append(f"{fmt(f2)} ({fmt(r)})")
                lines.append(f"| {dataset} | {d} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def s2() -> list[str]:
    lines = ["## S2: GLiNER's cut-off", ""]
    cutoffs = [round(0.05 * k, 2) for k in range(1, 20)]
    lines += ["Micro F2 of GLiNER, keeping the spans whose score is above each cut-off, on the clean test sets. "
              "BERT's F2 is given for reference; 0.35 is the cut-off of the benchmark.", "",
              "| Comparison | Dataset | BERT | " + " | ".join(f"{c:.2f}" for c in cutoffs) + " |",
              "|---" * (3 + len(cutoffs)) + "|"]
    for comparison in ("A", "B"):
        for dataset in SOURCES:
            examples, labels, predictions = load_comparison(dataset, comparison)
            low = read_predictions(LOW_CUTOFF_RUN / dataset / "clean.jsonl")
            bert = float(scores(example_counts(examples, predictions["bert"], labels).sum(axis=(0, 1)))["f2"])
            cells = []
            for c in cutoffs:
                kept = {i: [s for s in spans if s.score is not None and s.score > c] for i, spans in low.items()}
                cells.append(fmt(float(scores(example_counts(examples, kept, labels).sum(axis=(0, 1)))["f2"])))
            lines.append(f"| {comparison} | {dataset} | {bert:.3f} | " + " | ".join(cells) + " |")
    return lines + [""]


def s4() -> list[str]:
    lines = ["## S4: document length", "", "Micro F2 (comparison A) on documents of at most 384 words and on longer ones.", "",
             "| Dataset | Documents (short / long) | " + " | ".join(f"{d}: short / long" for d in DETECTORS) + " |",
             "|---" * (2 + len(DETECTORS)) + "|"]
    for dataset in SOURCES:
        examples, labels, predictions = load_comparison(dataset, "A")
        long = np.array([len(WORD_RE.findall(ex.text)) > 384 for ex in examples])
        cells = []
        for d in DETECTORS:
            counts = example_counts(examples, predictions[d], labels).sum(axis=1)
            short_f2 = float(scores(counts[~long].sum(0))["f2"])
            long_f2 = float(scores(counts[long].sum(0))["f2"]) if long.any() else float("nan")
            cells.append(f"{fmt(short_f2)} / {fmt(long_f2)}")
        lines.append(f"| {dataset} | {int((~long).sum())} / {int(long.sum())} | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    check_low_cutoff_run()
    lines = ["# Question 1: sensitivity analyses", ""] + s1() + s2() + s4()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "sensitivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
