"""Run one detector on test sets and save what it finds.

    uv run python experiments/run_benchmark.py gliner                          # every dataset, clean
    uv run python experiments/run_benchmark.py bert --datasets nemotron gretel
    uv run python experiments/run_benchmark.py bert --model models/bert-noisy --name bert-noisy  # question 3
    uv run python experiments/run_benchmark.py gliner --dates                  # comparison C

For each dataset and version, it reads data/test/<dataset>_<version>.jsonl and writes
results/predictions/<name>/<dataset>/<version>.jsonl, one line per example as soon as it is
done. After an interruption, run the same command again: examples already in the file are
skipped. Then score everything with experiments/score.py.

--dates (comparison C) reads the <version>_dates test sets, where every date is gold under
DATE. Presidio and BERT then call their date class DATE, and GLiNER is asked for "date"
instead of "date of birth".

Next to each predictions file, a .meta.json records the model, its settings, the computer,
the time taken, and the checksum of the test set, so score.py can tell if the test set has
changed since. It is written before the first prediction: a run only resumes a file made
with the same test set, model, revision and cut-off. GLiNER uses the cut-off chosen once by
experiments/tune_gliner.py.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from baselines import DETECTORS, BertDetector, GlinerDetector, PresidioDetector
from benchmark.data import PRED_DIR, SOURCES, TEST_DIR, THRESHOLD_FILE, file_sha256, predicted_ids, read_jsonl, spans_to_json
from PIIFilter.device import computer_info

RESUME_KEYS = ("model", "revision", "threshold")  # must match to continue a predictions file


def make_detector(name: str, model: str | None, device: str | None, dates: bool, threshold: float | None):
    if name == "presidio":
        return PresidioDetector(dates=dates)
    if name == "bert":
        return BertDetector(model=model, device=device, dates=dates)
    if threshold is None:
        if not THRESHOLD_FILE.exists():
            raise SystemExit(f"{THRESHOLD_FILE} is missing; run: uv run python experiments/tune_gliner.py")
        threshold = json.loads(THRESHOLD_FILE.read_text())["threshold"]
    return GlinerDetector(model=model, threshold=threshold, device=device, dates=dates)


def check_resume(out: Path, meta_path: Path, config: dict) -> None:
    """Stop if the predictions already in `out` were not made with this configuration."""
    if not meta_path.exists():
        if out.exists():
            raise SystemExit(f"{out} has no {meta_path.name}, so its settings are unknown; delete it to run again")
        return
    saved = json.loads(meta_path.read_text())
    if saved["test_sha256"] != config["test_sha256"]:
        raise SystemExit(f"{out} was made on another test set; delete it to run again")
    changed = [f"{key}: {saved.get(key)} -> {config[key]}" for key in RESUME_KEYS if saved.get(key) != config[key]]
    if changed:
        raise SystemExit(f"{out} was made with other settings ({'; '.join(changed)}); "
                         "delete it or use another --name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detector", choices=list(DETECTORS))
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--versions", nargs="+", default=["clean"], help="test set versions (default: clean)")
    parser.add_argument("--model", help="another model id or a local folder (e.g. a fine-tuned BERT)")
    parser.add_argument("--name", help="folder name in results/predictions/ (default: the detector name)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if a GPU is available, else cpu)")
    parser.add_argument("--dates", action="store_true", help="comparison C: every date is gold, under DATE")
    parser.add_argument("--threshold", type=float, help="GLiNER only: another cut-off, for the threshold analysis (needs --out)")
    parser.add_argument("--out", type=Path, default=PRED_DIR, help="where predictions go (default: results/predictions)")
    args = parser.parse_args()
    name = args.name or args.detector
    if args.model and not args.name:
        parser.error("--model needs --name, so the results do not mix with the original model's")
    if args.threshold is not None and (args.detector != "gliner" or args.out == PRED_DIR):
        parser.error("--threshold is for GLiNER only and needs --out, so the benchmark predictions are kept")
    versions = [f"{v}_dates" for v in args.versions] if args.dates else args.versions

    detector = make_detector(args.detector, args.model, args.device, args.dates, args.threshold)
    for dataset in args.datasets:
        for test_version in versions:
            test_path = TEST_DIR / f"{dataset}_{test_version}.jsonl"
            if not test_path.exists():
                raise SystemExit(f"{test_path} is missing; run experiments/prepare_data.py first")
            out = args.out / name / dataset / f"{test_version}.jsonl"
            meta_path = out.with_suffix(".meta.json")
            examples = read_jsonl(test_path)
            config = {
                "detector": args.detector, "name": name,
                "model": detector.model_id, "revision": getattr(detector, "revision", None),
                "threshold": getattr(detector, "threshold", None),
                "dates": args.dates,
                "dataset": dataset, "version": test_version,
                "test_set": test_path.name, "test_sha256": file_sha256(test_path), "examples": len(examples),
            }
            check_resume(out, meta_path, config)

            done = predicted_ids(out)
            todo = [ex for ex in examples if ex.id not in done]
            print(f"{name} {dataset} {test_version}: {len(todo)} of {len(examples)} examples to run")
            if not todo:
                continue

            out.parent.mkdir(parents=True, exist_ok=True)
            if not meta_path.exists():
                meta_path.write_text(json.dumps(config, indent=2) + "\n")
            start = time.perf_counter()
            every = max(1, len(todo) // 10)  # progress every 10%
            with open(out, "a", encoding="utf-8") as f:
                for i, ex in enumerate(todo, 1):
                    spans = detector.detect(ex.text)
                    f.write(json.dumps({"id": ex.id, "spans": spans_to_json(spans)}, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % every == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)}  {time.perf_counter() - start:.0f} s", flush=True)
            meta = {
                **config,
                # this run only: after an interruption, the earlier part is not counted
                "examples_run": len(todo), "seconds": round(time.perf_counter() - start, 1),
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "computer": computer_info(),
                "packages": {p: version(p) for p in ("presidio-analyzer", "transformers", "gliner", "torch")},
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
