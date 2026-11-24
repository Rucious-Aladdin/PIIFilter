"""Statistics of the data of questions 2 and 3, from the metadata files (no dataset text).

    uv run python experiments/summarize_data.py

Writes:
- results/noise/test_sets.md: the edits made in each noisy test set (data/test/*.meta.json,
  written by experiments/make_noisy.py). The _dates versions have the same edits.
- results/training/summary.md: the fine-tuning data (data/train/*.meta.json, written by
  experiments/prepare_train.py) and the training runs (results/training/*.json, written by
  experiments/train_bert.py): time, final dev losses, and the dev losses during training.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark.data import ROOT, SOURCES, TEST_DIR
from benchmark.noise import NOISE_KINDS

TRAIN_DIR = ROOT / "data" / "train"
RECORDS_DIR = ROOT / "results" / "training"
KINDS = NOISE_KINDS + ("mixed",)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_sets() -> list[str]:
    lines = ["# Noisy test sets: the edits", "",
             "From `data/test/<dataset>_noisy_<kind>.meta.json` (experiments/make_noisy.py, seed 1, "
             "round(0.10 x words) edits per example, at least one). The `_dates` versions have the same edits.", "",
             "| Dataset | Examples | Copy | Examples changed | Edits | Edits per example | Inside a PII value |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for dataset in SOURCES:
        for kind in KINDS:
            m = read(TEST_DIR / f"{dataset}_noisy_{kind}.meta.json")
            assert m["spans_per_label"] == m["spans_per_label_clean"], (dataset, kind)
            lines.append(f"| {dataset} | {m['n']:,} | {kind} | {m['examples_changed']:,} | {m['edits']:,} | "
                         f"{m['edits'] / m['n']:.1f} | {m['edits_inside_spans'] / m['edits']:.1%} |")
    lines += ["", "Edits of each kind in the mixed copies:", "",
              "| Dataset | " + " | ".join(NOISE_KINDS) + " |", "|---|" + "---:|" * len(NOISE_KINDS)]
    for dataset in SOURCES:
        per_kind = read(TEST_DIR / f"{dataset}_noisy_mixed.meta.json")["edits_per_kind"]
        lines.append(f"| {dataset} | " + " | ".join(f"{per_kind.get(k, 0):,}" for k in NOISE_KINDS) + " |")
    lines += ["", "Every noisy test set has the same gold spans per label as its clean version (checked here).", ""]
    return lines


def training() -> list[str]:
    lines = ["# Fine-tuning for question 3: data and runs", "",
             "## Data", "",
             "From `data/train/*.meta.json` (experiments/prepare_train.py): English records of the ai4privacy "
             "train split, a dev sample held out (seed 0), and one mixed-noise copy of each record (seed 2). "
             "Every date is under DATE.", "",
             "| File | Records | Noise | " + " | ".join(("NAME", "EMAIL", "PHONE", "ADDRESS", "DATE", "CARD_NUMBER")) + " |",
             "|---|---:|---|" + "---:|" * 6]
    for part in ("train", "dev"):
        for version in ("clean", "noisy"):
            m = read(TRAIN_DIR / f"ai4privacy_{part}_{version}.meta.json")
            noise = "none" if m["noise"] is None else f"{m['noise']['kind']}, rate {m['noise']['rate']}, seed {m['noise']['seed']}"
            counts = m["spans_per_label"]
            lines.append(f"| ai4privacy_{part}_{version} | {m['n']:,} | {noise} | "
                         + " | ".join(f"{counts.get(label, 0):,}" for label in ("NAME", "EMAIL", "PHONE", "ADDRESS", "DATE", "CARD_NUMBER")) + " |")
    english = read(TRAIN_DIR / "ai4privacy_train_clean.meta.json")["english_records"]
    lines += ["", f"English records in the train split: {english:,}.", ""]

    records = {p.stem: read(p) for p in sorted(RECORDS_DIR.glob("bert-*.json"))}
    if not records:
        return lines
    first = next(iter(records.values()))
    lines += ["## Runs", "",
              f"Base model: {first['base_model']} (revision {first['base_revision'][:10]}). "
              f"Settings: " + ", ".join(f"{k} {v}" for k, v in first["settings"].items()) + ".", "",
              "| Model | Training records | Steps | Minutes | GPU | Dev loss, clean | Dev loss, noisy |",
              "|---|---:|---:|---:|---|---:|---:|"]
    for name, r in records.items():
        loss = r["final_dev_loss"]
        lines.append(f"| {name} | {r['train_examples']:,} | {r['steps']:,} | {r['seconds'] / 60:.1f} | {r['gpu']} | "
                     f"{loss['eval_clean_loss']:.4f} | {loss['eval_noisy_loss']:.4f} |")
    packages = sorted({json.dumps(r["packages"], sort_keys=True) for r in records.values()})
    lines += ["", "Packages: " + "; ".join(", ".join(f"{k} {v}" for k, v in json.loads(p).items()) for p in packages) + ".", ""]

    # dev loss during training, mean over the seeds of each version
    curves = defaultdict(lambda: defaultdict(list))  # (version, dev set) -> step -> losses
    for name, r in records.items():
        version = name.rsplit("-s", 1)[0]
        for entry in r["log"]:
            if entry["step"] == r["steps"]:
                continue  # the evaluation after training: same model as the last one logged
            for dev in ("clean", "noisy"):
                if f"eval_{dev}_loss" in entry:
                    curves[(version, dev)][entry["step"]].append(entry[f"eval_{dev}_loss"])
    steps = sorted({s for c in curves.values() for s in c})
    lines += ["## Dev loss during training (mean over the seeds)", "",
              "| Model | Dev set | " + " | ".join(f"step {s:,}" for s in steps) + " |", "|---|---|" + "---:|" * len(steps)]
    for (version, dev), by_step in sorted(curves.items()):
        lines.append(f"| {version} | {dev} | " + " | ".join(
            f"{statistics.mean(by_step[s]):.4f}" if s in by_step else "" for s in steps) + " |")
    lines += ["", "The dev sets are only logged: the model of the last step is kept.", ""]
    return lines


def main() -> None:
    for path, lines in ((ROOT / "results" / "noise" / "test_sets.md", test_sets()),
                        (RECORDS_DIR / "summary.md", training())):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
