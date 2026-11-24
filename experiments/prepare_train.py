"""Draw the fine-tuning data for question 3: data/train/ai4privacy_{train,dev}_{clean,noisy}.jsonl.

Both fine-tuned BERT versions learn from the English records of the ai4privacy *train*
split (our ai4privacy test set comes from its validation split, so they never see a test
sentence). A fixed sample of DEV_SIZE records is held out as a dev set; the rest is the
training set.

    clean   the records as they are
    noisy   one noisy copy of each record: mixed noise at the test sets' rate, with another
            seed than the noisy test sets (benchmark/noise.py)

Version (a) of BERT learns from train_clean; version (b) from train_clean + train_noisy.

Gold spans use our labels with every date under DATE (comparison C labels): ai4privacy has
no birth-date label, and BERT's date class (DATE_TIME) means any date, so learning dates as
DATE keeps that class alive. On the clean test sets the detector still maps DATE_TIME to
DATE_OF_BIRTH, as for the original model.

    uv run python experiments/prepare_train.py
"""

import argparse
import json
import random
from collections import Counter

from benchmark.data import ROOT, Example, file_sha256, load_examples, write_jsonl
from benchmark.noise import RATE, example_rng, noisy_copy

TRAIN_DIR = ROOT / "data" / "train"
DEV_SIZE = 5_000
SPLIT_SEED = 0
TRAIN_NOISE_SEED = 2  # the noisy test sets use 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=int, default=DEV_SIZE, help=f"held-out records (default: {DEV_SIZE})")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help=f"train/dev split seed (default: {SPLIT_SEED})")
    parser.add_argument("--noise-seed", type=int, default=TRAIN_NOISE_SEED)
    parser.add_argument("--rate", type=float, default=RATE)
    args = parser.parse_args()

    records = load_examples("ai4privacy", "train", dates=True)
    dev_ids = set(random.Random(args.seed).sample(range(len(records)), args.dev))
    splits = {
        "train": [ex for i, ex in enumerate(records) if i not in dev_ids],
        "dev": [ex for i, ex in enumerate(records) if i in dev_ids],
    }
    for split, examples in splits.items():
        noisy = []
        for ex in examples:
            text, (spans,), edits = noisy_copy(
                ex.text, [ex.spans], "mixed", example_rng(args.noise_seed, "mixed", ex.id), args.rate
            )
            noisy.append(Example(ex.id, text, spans, {**ex.meta, "noise": {"kind": "mixed", "edits": edits}}))
        for version, items in (("clean", examples), ("noisy", noisy)):
            path = TRAIN_DIR / f"ai4privacy_{split}_{version}.jsonl"
            write_jsonl(path, items)
            counts = Counter(s.label for ex in items for s in ex.spans)
            meta = {
                "dataset": "ai4privacy", "split": "train", "part": split, "version": version,
                "english_records": len(records), "n": len(items), "dev_size": args.dev, "split_seed": args.seed,
                "noise": {"kind": "mixed", "rate": args.rate, "seed": args.noise_seed} if version == "noisy" else None,
                "spans_per_label": dict(sorted(counts.items())),
                "sha256": file_sha256(path),
            }
            path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"{path.name}: {len(items)} records, spans " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
