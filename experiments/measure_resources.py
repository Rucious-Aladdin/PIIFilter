"""Time and memory of each detector on CPU, each one in a fresh process.

    uv run python experiments/measure_resources.py
    uv run python experiments/measure_resources.py gliner --n 20

Each detector runs on the first 100 examples of each clean test set (loading the model and
a first warm-up call are not timed). GLiNER uses the cut-off of results/gliner_threshold.json.
Writes results/paper/resources.md and .json, with the mean and median time per example,
the peak RAM of the process (model, Python and libraries), the number of parameters and the
number of threads.

This script imports no heavy library outside the measured process, because a child process
starts with the peak RAM of its parent.
"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"
OUT = ROOT / "results" / "paper"
DATASETS = ("ai4privacy", "nemotron", "gretel")
DETECTORS = ("presidio", "bert", "gliner")
NAMES = {"presidio": "Presidio", "bert": "BERT", "gliner": "GLiNER", "ai4privacy": "ai4privacy",
         "nemotron": "Nemotron-PII", "gretel": "Gretel"}


def first_texts(dataset: str, n: int) -> list[str]:
    with open(TEST_DIR / f"{dataset}_clean.jsonl", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in islice(f, n)]


def load_detector(name: str):
    from baselines import DETECTORS as CLASSES

    if name == "presidio":
        return CLASSES[name]()
    if name == "bert":
        return CLASSES[name](device="cpu")
    threshold = json.loads((ROOT / "results" / "gliner_threshold.json").read_text())["threshold"]
    return CLASSES[name](threshold=threshold, device="cpu")


def count_parameters(name: str, detector) -> int:
    if name == "presidio":
        return 0  # rules only
    model = detector.pipe.model if name == "bert" else detector.model
    return sum(p.numel() for p in model.parameters())


def measure(name: str, n: int) -> dict:
    """Runs inside the fresh process."""
    detector = load_detector(name)
    times = {}
    for dataset in DATASETS:
        texts = first_texts(dataset, n)
        detector.detect(texts[0])  # warm-up
        seconds = []
        for text in texts:
            start = time.perf_counter()
            detector.detect(text)
            seconds.append(time.perf_counter() - start)
        times[dataset] = {"mean_ms": 1000 * statistics.mean(seconds), "median_ms": 1000 * statistics.median(seconds)}
    return {
        "parameters": count_parameters(name, detector),
        # Presidio's rules run in plain Python, even if one of its imports loads torch.
        "threads": 1 if name == "presidio" else sys.modules["torch"].get_num_threads(),
        "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # KB on Linux
        "times": times,
    }


def run_in_fresh_process(name: str, n: int) -> dict:
    command = [sys.executable, __file__, name, "--n", str(n), "--worker"]
    done = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    results = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if done.returncode or not results:
        raise SystemExit(f"{name} failed:\n{done.stderr[-2000:]}")
    return json.loads(results[-1].removeprefix("RESULT "))


def to_markdown(results: dict, n: int) -> str:
    lines = ["# Time and memory on CPU", "",
             f"{platform.machine()}, {os.cpu_count()} cores. First {n} examples of each clean test set; each detector "
             "in a fresh process. The peak RAM includes Python and the libraries.", "",
             "| Detector | Parameters | Threads | Peak RAM (MB) | " + " | ".join(f"{NAMES[d]}: mean / median (ms)" for d in DATASETS) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(DATASETS)]
    for name, r in results.items():
        times = [f"{r['times'][d]['mean_ms']:.1f} / {r['times'][d]['median_ms']:.1f}" for d in DATASETS]
        parameters = f"{r['parameters'] / 1e6:.1f} M" if r["parameters"] else "none (rules)"
        lines.append(f"| {NAMES[name]} | {parameters} | {r['threads']} | {r['peak_ram_mb']:.0f} | " + " | ".join(times) + " |")
    lines += ["", "Presidio loads no model: almost all of its RAM comes from importing presidio-analyzer, which "
              "imports spaCy and torch."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"detectors: {', '.join(DETECTORS)} (default: all)")
    parser.add_argument("--n", type=int, default=100, help="examples per test set (default: 100)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if set(args.names) - set(DETECTORS):
        parser.error(f"unknown detector(s): {args.names}")
    if args.worker:
        print("RESULT " + json.dumps(measure(args.names[0], args.n)))
        return

    results = {}
    for name in args.names or DETECTORS:
        print(f"{name}...", flush=True)
        results[name] = run_in_fresh_process(name, args.n)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resources.json").write_text(json.dumps({"examples": args.n, "detectors": results}, indent=2) + "\n")
    markdown = to_markdown(results, args.n)
    (OUT / "resources.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
