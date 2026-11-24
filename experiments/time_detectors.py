"""Time each detector on this computer, to choose the size of the test sets.

Each detector runs on the first examples of each clean test set (model loading and a
first warm-up call are not counted). The script prints the seconds per example, and the
hours the whole plan needs per 1,000 examples of a dataset: Presidio, GLiNER and three
BERT versions (original + two fine-tuned) on 6 versions of each test set (clean, 4 noise
types, mixed noise).

    uv run python experiments/time_detectors.py
    uv run python experiments/time_detectors.py gliner --n 50

Writes results/timing.json, with the computer's details.
"""

import argparse
import json
import time

from baselines import DETECTORS
from benchmark.data import ROOT, SOURCES, TEST_DIR, read_jsonl
from PIIFilter.device import computer_info

VERSIONS = 6  # clean + 4 noise types + mixed
RUNS = {"presidio": 1, "gliner": 1, "bert": 3}  # BERT: original + 2 fine-tuned versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"detectors: {', '.join(DETECTORS)} (default: all)")
    parser.add_argument("--n", type=int, default=100, help="examples timed per test set (default: 100)")
    args = parser.parse_args()

    seconds = {}
    for name in args.names or DETECTORS:
        detector = DETECTORS[name]()
        seconds[name] = {}
        for dataset in SOURCES:
            examples = read_jsonl(TEST_DIR / f"{dataset}_clean.jsonl")[: args.n]
            detector.detect(examples[0].text)  # warm-up
            start = time.perf_counter()
            for ex in examples:
                detector.detect(ex.text)
            seconds[name][dataset] = (time.perf_counter() - start) / len(examples)
            print(f"{name:>8} {dataset:>10}: {seconds[name][dataset]:.3f} s per example")

    print("\nHours for the whole plan, per 1,000 examples of a dataset:")
    plan = {}
    for dataset in SOURCES:
        per_example = sum(RUNS[name] * seconds[name][dataset] for name in seconds)
        plan[dataset] = per_example * 1000 * VERSIONS / 3600
        print(f"  {dataset:>10}: {plan[dataset]:.1f} h  ({', '.join(seconds)})")

    result = {"computer": computer_info(), "examples_timed": args.n, "seconds_per_example": seconds,
              "plan_hours_per_1000_examples": plan, "plan": {"versions": VERSIONS, "runs": RUNS}}
    path = ROOT / "results" / "timing.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
