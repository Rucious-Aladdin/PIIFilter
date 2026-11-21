"""Confidence intervals and paired tests for question 1 (sections 6.4 and 6.5 of the report).

    uv run python experiments/significance.py      # about 5 minutes on a laptop CPU

For each comparison (A, B, A-dates, B-dates; see benchmark/analysis.py) and each dataset:

- micro P, R, F1, F2 and leak rate of each detector, and F2 per label, with a 95% interval from
  a paired cluster bootstrap (10,000 draws, seed 0);
- for each pair of detectors, the difference of micro F1, of micro F2 and of leak rate, its 95%
  interval (same draws), and a two-sided paired approximate randomization test (10,000
  rounds, seed 0);
- Holm's correction over the 36 tests of each family (F1, F2, leak rate). F1 is a main
  score of the proposal, like F2; its tests were added after those of F2 and the leak rate.

A difference is called established when its corrected p-value is below 0.05, its interval
excludes 0 and it is at least 0.01. Writes results/analysis/significance.md and .json.
"""

import argparse
import json

import numpy as np

from benchmark.analysis import (
    COMPARISONS,
    DETECTORS,
    FN,
    PAIRS,
    TP,
    bootstrap_totals,
    cluster_ids,
    example_counts,
    holm,
    interval,
    load_comparison,
    per_cluster,
    randomization_test,
    scores,
)
from benchmark.data import ANALYSIS_DIR, SOURCES
from benchmark.scoring import fmt

DRAWS = ROUNDS = 10_000
ALPHA, SMALLEST = 0.05, 0.01
METRICS = ["precision", "recall", "f1", "f2", "leak_rate"]
TESTED = ["f1", "f2", "leak_rate"]  # one family of tests each


def analyse(dataset: str, comparison: str) -> tuple[dict, list[dict]]:
    examples, labels, predictions = load_comparison(dataset, comparison)
    clusters = cluster_ids(examples)
    counts = np.stack([example_counts(examples, predictions[d], labels) for d in DETECTORS], axis=1)
    by_cluster = per_cluster(counts, clusters)  # (clusters, detectors, labels, 4)
    draws = bootstrap_totals(by_cluster, DRAWS, seed=0)  # (draws, detectors, labels, 4)

    point_micro, draw_micro = scores(counts.sum(axis=(0, 2))), scores(draws.sum(axis=2))
    point_label, draw_label = scores(counts.sum(axis=0)), scores(draws)
    result = {"labels": labels, "examples": len(examples), "clusters": int(clusters.max() + 1), "detectors": {}}
    for k, d in enumerate(DETECTORS):
        result["detectors"][d] = {
            "micro": {m: [float(point_micro[m][k]), *interval(draw_micro[m][:, k])] for m in METRICS},
            "per_label": {
                label: {"f2": [float(point_label["f2"][k, j]), *interval(draw_label["f2"][:, k, j])],
                        "support": int(counts[:, k, j, TP].sum() + counts[:, k, j, FN].sum())}
                for j, label in enumerate(labels)
            },
        }

    tests = []
    micro_cluster = by_cluster.sum(axis=2)  # (clusters, detectors, 4)
    for x, y in PAIRS:
        i, j = DETECTORS.index(x), DETECTORS.index(y)
        for metric in TESTED:
            diff, p = randomization_test(micro_cluster[:, i], micro_cluster[:, j], metric, ROUNDS, seed=0)
            low, high = interval(draw_micro[metric][:, i] - draw_micro[metric][:, j])
            tests.append({"comparison": comparison, "dataset": dataset, "pair": [x, y], "metric": metric,
                          "difference": diff, "ci": [low, high], "p": p})
    return result, tests


def verdict(t: dict) -> str:
    x, y = t["pair"]
    better, worse = (x, y) if t["difference"] > 0 else (y, x)
    if t["metric"] == "leak_rate":  # lower is better
        better, worse = worse, better
    low, high = t["ci"]
    if t["p_holm"] < ALPHA and (low > 0 or high < 0) and abs(t["difference"]) >= SMALLEST:
        return f"{better} better"
    if t["p_holm"] < ALPHA:
        return "detectable but negligible"
    return "not established"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    results, tests = {}, []
    for comparison in COMPARISONS:
        for dataset in SOURCES:
            print(f"{comparison} {dataset} ...", flush=True)
            results[f"{comparison}/{dataset}"], new = analyse(dataset, comparison)
            tests += new
    for metric in TESTED:
        family = [t for t in tests if t["metric"] == metric]
        for t, adjusted in zip(family, holm([t["p"] for t in family])):
            t["p_holm"] = adjusted
            t["verdict"] = verdict(t)

    lines = ["# Question 1: scores with 95% intervals and paired tests", "",
             f"Paired cluster bootstrap ({DRAWS} draws) and approximate randomization ({ROUNDS} rounds), seed 0. "
             f"Holm's correction over the {len(tests) // len(TESTED)} tests of each family.", ""]
    for comparison in COMPARISONS:
        lines += [f"## Comparison {comparison}", "", "| Dataset | Detector | P | R | F1 | F2 | Leak rate |", "|---|---|---|---|---|---|---|"]
        for dataset in SOURCES:
            for d, r in results[f"{comparison}/{dataset}"]["detectors"].items():
                m = r["micro"]
                lines.append(f"| {dataset} | {d} | " + " | ".join(fmt(m[k][0], m[k][1:]) for k in METRICS) + " |")
        lines.append("")
    for metric, title in (("f1", "micro F1"), ("f2", "micro F2"), ("leak_rate", "leak rate")):
        lines += [f"## Differences of {title} (first minus second)", "",
                  "| Comparison | Dataset | Pair | Difference [95% CI] | p | p (Holm) | Verdict |", "|---|---|---|---|---|---|---|"]
        for t in (t for t in tests if t["metric"] == metric):
            lines.append(f"| {t['comparison']} | {t['dataset']} | {t['pair'][0]} - {t['pair'][1]} | "
                         f"{fmt(t['difference'], t['ci'])} | {t['p']:.4f} | {t['p_holm']:.4f} | {t['verdict']} |")
        lines.append("")
    lines += ["## F2 per label [95% CI] (support)", ""]
    for comparison in COMPARISONS:
        for dataset in SOURCES:
            r = results[f"{comparison}/{dataset}"]
            lines += [f"### {comparison}, {dataset}", "", "| Detector | " + " | ".join(r["labels"]) + " |",
                      "|---" * (len(r["labels"]) + 1) + "|"]
            for d, v in r["detectors"].items():
                lines.append(f"| {d} | " + " | ".join(
                    f"{fmt(v['per_label'][label]['f2'][0], v['per_label'][label]['f2'][1:])} ({v['per_label'][label]['support']})"
                    for label in r["labels"]) + " |")
            lines.append("")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "significance.json").write_text(json.dumps({"scores": results, "tests": tests}, indent=1) + "\n")
    (ANALYSIS_DIR / "significance.md").write_text("\n".join(lines) + "\n")
    print(f"Saved to {ANALYSIS_DIR / 'significance.md'} and .json")


if __name__ == "__main__":
    main()
