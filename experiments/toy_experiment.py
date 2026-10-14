"""Toy experiment (proposal, Section 5.4): the whole filter on one clean and one noisy prompt.

The prompts are read from experiments/scenario_samples/ (one JSON file per prompt).
It produces no scores. For each example it records the output of every step and the
three checks of Section 5.3. "Private?" is decided automatically from the gold spans;
"Enough context?" and "Usable answer?" are automatic proxies that we confirm by reading
the answers.

    uv run python experiments/toy_experiment.py --backend local
    uv run python experiments/toy_experiment.py --backend openai --model gpt-5-mini
    uv run python experiments/toy_experiment.py --backend anthropic --model claude-opus-5-5
"""

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from baselines.gliner_detector import GlinerDetector
from PIIFilter.llm import SYSTEM_PROMPT, make_llm
from PIIFilter.pipeline import FilterTrace, run_filter

SAMPLES_DIR = Path(__file__).parent / "scenario_samples"


def load_samples(samples_dir: Path) -> list[dict]:
    """One JSON file per prompt: {"id", "title", "prompt", "gold": [{"label", "value"}]}."""
    return [json.loads(f.read_text()) for f in sorted(samples_dir.glob("*.json"))]


# Placeholder-like text left in the final answer: unknown <...> tags, or a placeholder
# the LLM rewrote (NAME_1, Name 1, [PHONE_1]).
LEFTOVER_RE = re.compile(
    r"<[A-Za-z_]+(?:_\d+)?>|\b(?:NAME|EMAIL|PHONE|ADDRESS|DATE_OF_BIRTH|CARD_NUMBER)[_ ]?\d+\b",
    re.IGNORECASE,
)


def gold_status(trace: FilterTrace, label: str, value: str) -> dict:
    """How the detector handled one gold span: exact, covered, partial or missed."""
    start = trace.prompt.index(value)
    end = start + len(value)
    masked_chars = {i for s in trace.spans for i in range(s.start, s.end)}
    leaked = "".join(trace.prompt[i] if i not in masked_chars else "_" for i in range(start, end))
    if any((s.start, s.end, s.label) == (start, end, label) for s in trace.spans):
        status = "exact"
    elif "_" * len(value) == leaked:
        status = "covered"  # fully masked, but not an exact strict match
    elif "_" in leaked:
        status = "partial"
    else:
        status = "missed"
    return {"label": label, "value": value, "start": start, "end": end, "status": status}


def check(trace: FilterTrace, gold: list[dict]) -> dict:
    golds = [gold_status(trace, g["label"], g["value"]) for g in gold]
    leaked = [g["value"] for g in golds if g["status"] in ("partial", "missed")]
    leaked += [
        g["value"] for g in gold
        if g["value"].lower() in trace.masked_prompt.lower() and g["value"] not in leaked
    ]
    gold_offsets = {(g["start"], g["end"]) for g in golds}
    extra = [trace.prompt[s.start : s.end] for s in trace.spans if (s.start, s.end) not in gold_offsets]
    missing = [p for p in trace.mapping if p not in trace.masked_answer]
    leftover = LEFTOVER_RE.findall(trace.answer)
    return {
        "gold": golds,
        "extra_detections": extra,
        "leaked_values": leaked,
        "placeholders_missing_from_answer": missing,
        "leftover_placeholders": leftover,
        "private": not leaked,
        "enough_context": bool(trace.masked_answer) and not missing,
        "usable_answer": bool(trace.answer) and not leftover,
    }


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


def to_markdown(meta: dict, rows: list[dict]) -> str:
    def cell(text: str) -> str:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "# Toy experiment results",
        "",
        f"- Date: {meta['date']}",
        f"- Detector: `{meta['detector']['model']}` (threshold {meta['detector']['threshold']})",
        f"- LLM: `{meta['llm']['model']}` ({meta['llm']['backend']} backend)",
        f"- System prompt: {cell(meta['llm']['system_prompt'])}",
        "",
        "## Detected spans (GLiNER confidence)",
        "",
        "| Example | Gold value | Label | Result | Confidence |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        scores = {(s["start"], s["end"]): s["score"] for s in row["spans"]}
        for g in row["checks"]["gold"]:
            score = scores.get((g["start"], g["end"]), "-")
            lines.append(f"| {row['title']} | {g['value']} | {g['label']} | {g['status']} | {score} |")
    lines += ["", "## Output of each step (Table 9)", "", "| | " + " | ".join(r["title"] for r in rows) + " |"]
    lines.append("|---" * (len(rows) + 1) + "|")
    for key, name in [
        ("prompt", "User prompt"),
        ("masked_prompt", "Prompt sent to the LLM"),
        ("masked_answer", "LLM answer (masked)"),
        ("answer", "Answer shown to the user"),
    ]:
        lines.append(f"| {name} | " + " | ".join(cell(r[key]) for r in rows) + " |")
    verdicts = [
        " / ".join(yes_no(r["checks"][k]) for k in ("private", "enough_context", "usable_answer"))
        for r in rows
    ]
    lines.append("| Private? / Enough context? / Usable answer? | " + " | ".join(verdicts) + " |")
    lines += ["", "## Check details", ""]
    for row in rows:
        c = row["checks"]
        lines += [
            f"**{row['title']}**",
            f"- Leaked values: `{c['leaked_values']}`",
            f"- Extra detections: `{c['extra_detections']}`",
            f"- Placeholders missing from the LLM answer: `{c['placeholders_missing_from_answer']}`",
            f"- Placeholder-like text left after unmasking: `{c['leftover_placeholders']}`",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["local", "openai", "anthropic"], default="local")
    parser.add_argument("--model", help="LLM model id (default depends on the backend)")
    parser.add_argument("--threshold", type=float, default=0.5, help="GLiNER confidence cut-off")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=Path, default=SAMPLES_DIR, help="folder of scenario JSON files")
    parser.add_argument("--out", type=Path, default=Path("results/toy"))
    args = parser.parse_args()
    load_dotenv()  # API keys may be kept in .env

    detector = GlinerDetector(threshold=args.threshold, device=args.device)
    llm = make_llm(args.backend, args.model)

    rows = []
    for example in load_samples(args.samples):
        trace = run_filter(example["prompt"], detector, llm)
        rows.append({
            "id": example["id"],
            "title": example["title"],
            **asdict(trace),
            "checks": check(trace, example["gold"]),
        })

    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": str(args.samples),
        "detector": {"model": detector.model_id, "threshold": args.threshold},
        "llm": {"backend": args.backend, "model": llm.model_id, "system_prompt": SYSTEM_PROMPT},
        "packages": {p: version(p) for p in ("gliner", "torch", "transformers", "openai", "anthropic")},
    }
    out_dir = args.out / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "toy_results.json").write_text(json.dumps({"meta": meta, "examples": rows}, indent=2, ensure_ascii=False))
    markdown = to_markdown(meta, rows)
    (out_dir / "toy_results.md").write_text(markdown + "\n")
    print(markdown)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
