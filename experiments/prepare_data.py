"""Draw the clean test sets: data/test/<dataset>_clean.jsonl.

Each test set is a random sample of the dataset's English test records, drawn with a fixed
seed, so every computer that runs this script gets the same examples. Gold spans use our
six labels (benchmark/data.py). Records without any of our labels are kept: they test
whether a detector masks things that are not PII.

<dataset>_clean_dates.jsonl has the same examples for comparison C: every date is gold,
under DATE, and nothing is DATE_OF_BIRTH.

    uv run python experiments/prepare_data.py                    # all three, at our sizes
    uv run python experiments/prepare_data.py gretel --n 500     # one dataset, another size

Our sizes ("test_size" in benchmark/data.py) were chosen after timing the detectors:
ai4privacy 10,000, Nemotron-PII 2,000, Gretel 2,962 (all its English test documents).

Next to each test set, a .meta.json records the source, the size, the seed, the labels
scored on it, the number of gold spans per label and a SHA-256 checksum of the file, so we
can check that two computers have the same test set.
"""

import argparse
import json
import random
from collections import Counter

from benchmark.data import SOURCES, TEST_DIR, file_sha256, load_examples, scored_labels, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"datasets: {', '.join(SOURCES)} (default: all)")
    parser.add_argument("--n", type=int, help="examples per test set (default: our size for each dataset)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    args = parser.parse_args()
    unknown = set(args.names) - set(SOURCES)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    for name in args.names or SOURCES:
        for dates, version in [(False, "clean"), (True, "clean_dates")]:
            # Same records in the same order for both versions, so the same seed picks the same examples.
            examples = load_examples(name, "test", dates)
            n = min(args.n or SOURCES[name]["test_size"], len(examples))
            picked = sorted(random.Random(args.seed).sample(range(len(examples)), n))
            sample = [examples[i] for i in picked]

            path = TEST_DIR / f"{name}_{version}.jsonl"
            write_jsonl(path, sample)
            labels = scored_labels(name, dates)
            counts = Counter(s.label for ex in sample for s in ex.spans)
            meta = {
                "dataset": name,
                "version": version,
                "repo": SOURCES[name]["repo"],
                "revision": SOURCES[name]["revision"],
                "split": SOURCES[name]["splits"]["test"],
                "english_records": len(examples),
                "n": n,
                "seed": args.seed,
                "labels": labels,
                "spans_per_label": {label: counts[label] for label in labels},
                "examples_without_spans": sum(not ex.spans for ex in sample),
                "sha256": file_sha256(path),
            }
            path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

            print(f"{name} {version}: {n} of {len(examples)} English records -> {path}")
            print("  spans: " + ", ".join(f"{label} {counts[label]}" for label in labels))
            print(f"  without spans: {meta['examples_without_spans']} | sha256 {meta['sha256'][:12]}")


if __name__ == "__main__":
    main()
