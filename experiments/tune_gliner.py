"""Choose GLiNER's confidence cut-off once, before any test (proposal, Section 5.2).

GLiNER runs on 300 English sentences drawn from the ai4privacy *train* split (never from a
test set), with a very low cut-off. Then, for each cut-off from 0.05 to 0.95, we keep the
spans whose confidence is above it and compute the micro F2 (benchmark/scoring.py). The
best cut-off is written to results/gliner_threshold.json and used for every test.

Filtering afterwards gives the same spans as running GLiNER again with the higher cut-off
(up to the rounding of scores to 3 decimals): GLiNER keeps the spans above the cut-off,
then picks them from the highest score down, so lower-scored spans never change which
higher-scored ones are picked.

    uv run python experiments/tune_gliner.py
"""

import argparse
import json
import random

from baselines.gliner_detector import GlinerDetector
from benchmark.data import THRESHOLD_FILE, load_examples, scored_labels
from benchmark.scoring import score

GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="training sentences used (default: 300)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="cuda or cpu (default: cuda if a GPU is available, else cpu)")
    args = parser.parse_args()

    train = load_examples("ai4privacy", "train")
    examples = [train[i] for i in sorted(random.Random(args.seed).sample(range(len(train)), args.n))]
    detector = GlinerDetector(device=args.device)
    found = {ex.id: detector.detect(ex.text, threshold=min(GRID)) for ex in examples}

    labels = scored_labels("ai4privacy")
    rows = []
    for cut in GRID:
        kept = {id_: [s for s in spans if s.score > cut] for id_, spans in found.items()}
        micro = score(examples, kept, labels)["micro"]
        rows.append({"threshold": cut, **{k: micro[k] for k in ("precision", "recall", "f1", "f2")}})
        print(f"cut-off {cut:.2f}: P {micro['precision']:.3f}  R {micro['recall']:.3f}  F2 {micro['f2']:.3f}")

    best = max(rows, key=lambda r: r["f2"])  # the first (lowest) cut-off wins a tie
    result = {
        "threshold": best["threshold"],
        "model": detector.model_id,
        "revision": detector.revision,
        "data": f"ai4privacy train split, {args.n} English sentences, seed {args.seed}",
        "labels": labels,
        "grid": rows,
    }
    THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nBest cut-off: {best['threshold']} (F2 {best['f2']:.3f}) -> {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()
