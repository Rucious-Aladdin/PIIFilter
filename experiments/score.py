"""Score every detector output against its test set.

Each predictions file results/predictions/<detector>/<dataset>/<version>.jsonl (written by
experiments/run_benchmark.py) is compared with data/test/<dataset>_<version>.jsonl, where
<version> is "clean" or a noisy copy. Rules and formulas: benchmark/scoring.py.

    uv run python experiments/score.py
    uv run python experiments/score.py --labels EMAIL PHONE DATE_OF_BIRTH DATE CARD_NUMBER \
        --out results/without_name_address

Writes results/scores.json (every number) and results/scores.md: one table per detector and
version, like Table 7 of the proposal. Label cells give F2 (support); P, R, F1, F2 and the
leak rate on the right are micro averages. "-" means the dataset has no such label; "n/a"
means the score does not exist (e.g. no gold span of that label in the test set).

--labels scores only some labels, for example without NAME and ADDRESS, which Presidio's
rules cannot find, so that every detector is compared on the same ground. The micro
averages then cover these labels only.

Each test set's .meta.json lists the labels scored on it: our six on the clean versions,
DATE in place of DATE_OF_BIRTH on the _dates versions (comparison C).
"""

import argparse
import json
from pathlib import Path

from benchmark.data import ALL_LABELS, PRED_DIR, ROOT, SOURCES, TEST_DIR, file_sha256, read_jsonl, read_predictions
from benchmark.scoring import fmt, score


def to_markdown(results: list[dict], labels: list[str]) -> str:
    title = "# Benchmark scores" if labels == list(ALL_LABELS) else f"# Benchmark scores, {', '.join(labels)} only"
    lines = [title, ""]
    groups = sorted({(r["detector"], r["version"]) for r in results})
    for detector, version in groups:
        rows = [r for r in results if (r["detector"], r["version"]) == (detector, version)]
        # Columns: the labels scored on at least one of these test sets (DATE only on _dates).
        columns = [label for label in labels if any(label in r["scores"]["per_label"] for r in rows)]
        lines += [
            f"## {detector}, {version}", "",
            "| Dataset | " + " | ".join(columns) + " | P | R | F1 | F2 | Leak |",
            "|---" * (len(columns) + 6) + "|",
        ]
        for r in rows:
            per_label, micro = r["scores"]["per_label"], r["scores"]["micro"]
            cells = [
                f"{fmt(per_label[label]['f2'])} ({per_label[label]['support']})" if label in per_label else "-"
                for label in columns
            ]
            cells += [fmt(micro[k]) for k in ("precision", "recall", "f1", "f2", "leak_rate")]
            lines.append(f"| {r['dataset']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=PRED_DIR, help="folder of predictions files")
    parser.add_argument("--out", type=Path, help="where scores.json and scores.md go (default: results/)")
    parser.add_argument("--labels", nargs="+", choices=ALL_LABELS, help="score only these labels (needs --out)")
    args = parser.parse_args()
    if args.labels and not args.out:
        parser.error("--labels needs --out, so the scores of all labels in results/ are kept")
    labels = [label for label in ALL_LABELS if label in (args.labels or ALL_LABELS)]  # in a fixed order
    out = args.out or ROOT / "results"

    results = []
    for path in sorted(args.predictions.glob("*/*/*.jsonl")):
        detector, dataset, version = path.parent.parent.name, path.parent.name, path.stem
        gold_path = TEST_DIR / f"{dataset}_{version}.jsonl"
        if dataset not in SOURCES or not gold_path.exists():
            print(f"skipped {path}: no test set {gold_path.name}")
            continue
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists() and json.loads(meta_path.read_text())["test_sha256"] != file_sha256(gold_path):
            print(f"skipped {path}: made on another version of {gold_path.name}")
            continue
        test_meta = json.loads(gold_path.with_suffix(".meta.json").read_text())
        if "labels" not in test_meta:
            raise SystemExit(f"{gold_path.name} was drawn by an older script; run experiments/prepare_data.py again")
        dataset_labels = [label for label in test_meta["labels"] if label in labels]
        scores = score(read_jsonl(gold_path), read_predictions(path), dataset_labels)
        results.append({"detector": detector, "dataset": dataset, "version": version, "scores": scores})
        micro = scores["micro"]
        print(f"{detector:>10} {dataset:>10} {version:>8}: F1 {fmt(micro['f1'])}  F2 {fmt(micro['f2'])}  leak {fmt(micro['leak_rate'])}")

    if not results:
        print(f"no predictions found in {args.predictions}")
        return
    results.sort(key=lambda r: list(SOURCES).index(r["dataset"]))  # rows in the order of Table 7
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "scores.md").write_text(to_markdown(results, labels) + "\n")
    print(f"\nSaved to {out}/scores.json and scores.md")


if __name__ == "__main__":
    main()
