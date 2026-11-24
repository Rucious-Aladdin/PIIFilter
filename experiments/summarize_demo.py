"""Summarize a toy_results.json written by experiments/toy_experiment.py.

    uv run python experiments/summarize_demo.py results/demo/openai/toy_results.json
    uv run python experiments/summarize_demo.py results/demo/openai/toy_results.json \
        --out results/demo/openai/summary.md

Counts how many prompts pass each of the three checks of the proposal (Section 5.3), apart
for the clean and the noisy prompts, whose ids end in _clean and _noisy. "Private?" comes
from the gold values. "Enough context?" and "Usable answer?" are automatic proxies, so the
output lists each answer: confirm them by reading it.
"""

import argparse
import json
from pathlib import Path

CHECKS = {"private": "Private?", "enough_context": "Enough context?", "usable_answer": "Usable answer?"}
VERSIONS = ("clean", "noisy")


def version_of(example: dict) -> str:
    version = example["id"].rsplit("_", 1)[-1]
    if version not in VERSIONS:
        raise SystemExit(f"{example['id']}: the id must end in _clean or _noisy")
    return version


def summarize(examples: list[dict]) -> str:
    lines = ["# Demo summary", "", "| Check | " + " | ".join(VERSIONS) + " |", "|---|" + "---:|" * len(VERSIONS)]
    groups = {v: [ex for ex in examples if version_of(ex) == v] for v in VERSIONS}
    for key, name in CHECKS.items():
        cells = [f"{sum(ex['checks'][key] for ex in groups[v])} / {len(groups[v])}" for v in VERSIONS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "*Enough context?* and *Usable answer?* are automatic proxies: confirm them by reading the answers.", "",
              "## Failed checks", ""]
    failed = [f"- `{ex['id']}`: " + ", ".join(name for key, name in CHECKS.items() if not ex["checks"][key])
              for ex in examples if not all(ex["checks"][key] for key in CHECKS)]
    lines += failed or ["None."]
    lines += ["", "## Answers to read", ""]
    for ex in examples:
        lines += [f"### {ex['id']}", "", f"- Sent to the LLM: {ex['masked_prompt']}", f"- Answer shown: {ex['answer']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="a toy_results.json file")
    parser.add_argument("--out", type=Path, help="also write the summary to this Markdown file")
    args = parser.parse_args()
    markdown = summarize(json.loads(args.results.read_text())["examples"])
    print(markdown)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown + "\n")


if __name__ == "__main__":
    main()
