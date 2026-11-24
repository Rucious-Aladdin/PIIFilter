"""Tables for questions 2 and 3, from results/scores.json (experiments/score.py).

Question 2: how much each detector loses on the noisy test sets. One table per dataset:
micro F2 on the clean test set and on each noisy copy, and the change from clean; then the
same for the leak rate.

Question 3: does fine-tuning BERT on noisy copies win back that loss, and does it cost
anything on clean text? The original BERT, BERT fine-tuned on clean text and BERT fine-tuned
on clean + noisy text (mean and standard deviation over the training seeds), with:

    loss       F2 on clean - F2 on noisy, for the BERT fine-tuned on clean text
    won back   F2 of the noisy-trained BERT on noisy - F2 of the clean-trained BERT on noisy
    share      won back / loss
    clean cost F2 of the clean-trained BERT on clean - F2 of the noisy-trained BERT on clean

Both fine-tuned versions saw ai4privacy-style sentences, so they are compared with each
other, not with the other detectors (proposal, Appendix B).

    uv run python experiments/summarize_noise.py                  # comparison A (the six labels)
    uv run python experiments/summarize_noise.py --dates          # comparison A-dates
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from benchmark.data import ROOT, SOURCES
from benchmark.noise import NOISE_KINDS
from benchmark.scoring import fmt

VERSIONS = ("clean",) + tuple(f"noisy_{k}" for k in NOISE_KINDS + ("mixed",))
DETECTORS = ("presidio", "bert", "gliner")
FINE_TUNED = {"bert-clean": "BERT, fine-tuned on clean", "bert-noisy": "BERT, fine-tuned on clean + noisy"}


def short(version: str) -> str:
    return version.removeprefix("noisy_")


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=ROOT / "results" / "scores.json")
    parser.add_argument("--dates", action="store_true", help="comparison A-dates: the _dates test sets")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "noise")
    args = parser.parse_args()
    suffix = "_dates" if args.dates else ""

    # (detector, dataset, version) -> micro scores
    micro = {(r["detector"], r["dataset"], r["version"]): r["scores"]["micro"] for r in json.loads(args.scores.read_text())}

    def get(detector, dataset, version, key):
        scores = micro.get((detector, dataset, version + suffix))
        return scores[key] if scores else None

    seeds = {}
    for detector, _, _ in micro:
        match = re.fullmatch(r"(bert-(?:clean|noisy))-s(\d+)", detector)
        if match:
            seeds.setdefault(match.group(1), set()).add(detector)

    lines = [f"# Questions 2 and 3: noisy test sets{', comparison A-dates' if args.dates else ', comparison A'}", ""]
    lines += [
        "Micro scores with strict matching. Noisy copies: about one edit per ten words "
        "(benchmark/noise.py, experiments/make_noisy.py).", "",
    ]

    # Question 2
    lines += ["## Question 2: F2 of each detector", ""]
    for key, name in (("f2", "F2"), ("leak_rate", "Leak rate")):
        for dataset in SOURCES:
            lines += [f"### {name}, {dataset}", ""]
            lines += ["| Detector | " + " | ".join(short(v) for v in VERSIONS) + " |", "|---" + "|---:" * len(VERSIONS) + "|"]
            for detector in DETECTORS:
                clean = get(detector, dataset, "clean", key)
                cells = [fmt(clean)]
                for v in VERSIONS[1:]:
                    value = get(detector, dataset, v, key)
                    delta = None if value is None or clean is None else value - clean
                    cells.append(f"{fmt(value)} ({signed(delta)})")
                lines.append(f"| {detector} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["In brackets: the change from the clean test set.", ""]

    # Question 3
    if seeds:
        lines += ["## Question 3: fine-tuned BERT (F2, mean ± sd over seeds)", ""]
        for prefix, names in sorted(seeds.items()):
            lines.append(f"- {FINE_TUNED[prefix]}: {', '.join(sorted(names))}")
        lines.append("")
        for dataset in SOURCES:
            lines += [f"### {dataset}", ""]
            lines += ["| Model | " + " | ".join(short(v) for v in VERSIONS) + " |", "|---" + "|---:" * len(VERSIONS) + "|"]
            lines.append("| BERT, original | " + " | ".join(fmt(get("bert", dataset, v, "f2")) for v in VERSIONS) + " |")
            stats = {}
            for prefix in sorted(seeds):
                row = []
                for v in VERSIONS:
                    m, sd = mean_sd([get(d, dataset, v, "f2") for d in seeds[prefix]])
                    stats[(prefix, v)] = m
                    row.append("n/a" if m is None else f"{m:.3f} ± {sd:.3f}")
                lines.append(f"| {FINE_TUNED[prefix]} | " + " | ".join(row) + " |")
            lines.append("")
            if all(stats.get((p, "clean")) is not None for p in FINE_TUNED):
                lines += ["| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |",
                          "|---|---:|---:|---:|"]
                for v in VERSIONS[1:]:
                    c_clean, c_noisy = stats[("bert-clean", "clean")], stats[("bert-clean", v)]
                    n_noisy = stats[("bert-noisy", v)]
                    if None in (c_clean, c_noisy, n_noisy):
                        continue
                    loss, won = c_clean - c_noisy, n_noisy - c_noisy
                    share = f"{won / loss:.0%}" if loss > 0.005 else "n/a (no loss)"
                    lines.append(f"| {short(v)} | {loss:.3f} | {won:+.3f} | {share} |")
                cost = stats[("bert-clean", "clean")] - stats[("bert-noisy", "clean")]
                lines += ["", f"Clean cost (F2 on clean, clean-trained minus noisy-trained): {cost:+.3f}", ""]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"summary{suffix}.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
