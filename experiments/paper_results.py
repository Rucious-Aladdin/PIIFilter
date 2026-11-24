"""Tables of the paper for one comparison of question 1, written to results/paper/.

    uv run python experiments/paper_results.py                       # comparison A
    uv run python experiments/paper_results.py --comparison A-dates

1. Main table (Markdown and LaTeX booktabs): P, R, F1, F2 and leak rate, read from
   results/analysis/significance.json. A value is bold only if the paired tests establish
   that this detector is better than both others. P and R were not tested: never bold.
2. Over-masking: the share of non-space characters outside every gold span that the filter
   masks anyway.
3. Error types: each strict FN and FP is "boundary", "wrong_label", or "missed" / "spurious".
   The totals must equal the FN and FP of the saved scores.

Files have no suffix for comparison A and end in _<comparison> otherwise. One short example
per error type goes to data/analysis/error_examples.md, which is git-ignored because it
quotes dataset text.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

from benchmark.analysis import (
    COMPARISONS,
    DETECTORS,
    FN_TYPES,
    FP_TYPES,
    classify_errors,
    load_comparison,
    overmasked_chars,
)
from benchmark.data import ANALYSIS_DIR, RAW_DIR, ROOT, SOURCES
from benchmark.spans import normalize_spans

OUT = ROOT / "results" / "paper"
EXAMPLES_DIR = ROOT / "data" / "analysis"
DATASET_NAMES = {"ai4privacy": "ai4privacy", "nemotron": "Nemotron-PII", "gretel": "Gretel"}
DETECTOR_NAMES = {"presidio": "Presidio", "bert": "BERT", "gliner": "GLiNER"}
METRICS = {"precision": "P", "recall": "R", "f1": "F1", "f2": "F2", "leak_rate": "Leak"}
TESTED = ("f1", "f2", "leak_rate")  # the only scores with paired tests
COMPARISON_TEXT = {
    "A": "the six labels",
    "B": "email, phone, date of birth and card number",
    "A-dates": "the six labels, with every date in place of date of birth",
    "B-dates": "email, phone, every date and card number",
}


def suffix(comparison: str) -> str:
    return "" if comparison == "A" else f"_{comparison}"


def best_detector(tests: list[dict], comparison: str, dataset: str, metric: str) -> str | None:
    """The detector established as better than both others on this score, if there is one."""
    verdicts = {frozenset(t["pair"]): t["verdict"] for t in tests
                if (t["comparison"], t["dataset"], t["metric"]) == (comparison, dataset, metric)}
    for d in DETECTORS:
        if all(verdicts[frozenset((d, other))] == f"{d} better" for other in DETECTORS if other != d):
            return d
    return None


def main_table(comparison: str) -> None:
    data = json.loads((ANALYSIS_DIR / "significance.json").read_text())
    names = list(METRICS.values())
    md = ["| Dataset | Detector | " + " | ".join(names) + " |", "|---|---|" + "---:|" * len(names)]
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{ll" + "r" * len(names) + "}", r"\toprule",
           "Dataset & Detector & " + " & ".join(names) + r" \\", r"\midrule"]
    for k, dataset in enumerate(SOURCES):
        best = {m: best_detector(data["tests"], comparison, dataset, m) for m in TESTED}
        for j, d in enumerate(DETECTORS):
            micro = data["scores"][f"{comparison}/{dataset}"]["detectors"][d]["micro"]
            values = [(f"{micro[m][0]:.3f}", best.get(m) == d) for m in METRICS]
            first = DATASET_NAMES[dataset] if j == 0 else ""
            md.append(f"| {first} | {DETECTOR_NAMES[d]} | " + " | ".join(f"**{v}**" if b else v for v, b in values) + " |")
            tex.append(f"{first} & {DETECTOR_NAMES[d]} & " + " & ".join(rf"\textbf{{{v}}}" if b else v for v, b in values) + r" \\")
        if k < len(SOURCES) - 1:
            tex.append(r"\midrule")
    note = "Bold: established as better than both other detectors (paired tests, Holm's correction). P and R were not tested."
    tex += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{Micro scores on the clean test sets, strict matching ({COMPARISON_TEXT[comparison]}). {note}}}",
            rf"\label{{tab:main{suffix(comparison).replace('_', '-')}}}", r"\end{table}"]
    head = [f"# Main table, comparison {comparison} ({COMPARISON_TEXT[comparison]})", ""]
    (OUT / f"main_table{suffix(comparison)}.md").write_text("\n".join(head + md + ["", note]) + "\n")
    (OUT / f"main_table{suffix(comparison)}.tex").write_text("\n".join(tex) + "\n")


def overmasking(comparison: str) -> None:
    lines = [f"# Over-masking, comparison {comparison}", "",
             "Share of the non-space characters outside every gold span, of any label of the dataset, that the "
             "detections mask (after the span clean-up).", "",
             "| Dataset | " + " | ".join(DETECTOR_NAMES[d] for d in DETECTORS) + " |", "|---|" + "---:|" * len(DETECTORS)]
    for dataset, source in SOURCES.items():
        examples, _, predictions = load_comparison(dataset, comparison)
        ids = {ex.id for ex in examples}
        raw_gold = {id_: spans for id_, _, spans, _ in source["read"](RAW_DIR / dataset / source["splits"]["test"]) if id_ in ids}
        cells = []
        for d in DETECTORS:
            counts = [overmasked_chars(ex.text, raw_gold[ex.id], predictions[d][ex.id]) for ex in examples]
            cells.append(f"{100 * sum(c[0] for c in counts) / sum(c[1] for c in counts):.2f}%")
        lines.append(f"| {DATASET_NAMES[dataset]} | " + " | ".join(cells) + " |")
    lines += ["", "Gretel contains PII left unannotated, so its values are upper bounds: some of the text counted "
              "as masked by mistake is real PII."]
    (OUT / f"overmasking{suffix(comparison)}.md").write_text("\n".join(lines) + "\n")


def saved_errors(comparison: str) -> dict[tuple[str, str], tuple[int, int]]:
    """(FN, FP) of the saved scores, by (dataset, detector)."""
    version, keep = COMPARISONS[comparison]
    path = ROOT / "results" / ("scores.json" if keep is None else "without_name_address/scores.json")
    return {(r["dataset"], r["detector"]): (r["scores"]["micro"]["fn"], r["scores"]["micro"]["fp"])
            for r in json.loads(path.read_text()) if r["version"] == version}


def write_examples(comparison: str, candidates: dict) -> None:
    """One short random example per error type (seed 0), with the span it overlaps."""
    rng = random.Random(0)
    lines = [f"# Error examples, comparison {comparison}", "", "Quotes dataset text: keep out of git.", ""]
    for (side, kind), items in sorted(candidates.items()):
        short = [item for item in items if item[3].end - item[3].start <= 40] or items
        dataset, d, ex, span, detected = rng.choice(short)
        others = ex.spans if side == "FP" else normalize_spans(ex.text, detected)
        near = [f"{ex.text[o.start:o.end]} ({o.label})" for o in others if o.start < span.end and span.start < o.end]
        context = " ".join(ex.text[max(0, span.start - 40) : span.end + 40].split())
        lines += [f"## {side}, {kind}", "", f"- {DETECTOR_NAMES[d]} on {DATASET_NAMES[dataset]}, label {span.label}",
                  f"- Span: `{ex.text[span.start:span.end]}`", f"- Overlaps: {', '.join(near) or 'nothing'}",
                  f"- Context: `{context}`", ""]
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (EXAMPLES_DIR / f"error_examples{suffix(comparison)}.md").write_text("\n".join(lines))


def error_types(comparison: str) -> None:
    saved = saved_errors(comparison)
    columns = [("FN", k) for k in FN_TYPES] + [("FP", k) for k in FP_TYPES]
    head = ["FN"] + [k.replace("_", " ") for k in FN_TYPES] + ["FP"] + [k.replace("_", " ") for k in FP_TYPES]
    lines = [f"# Error types, comparison {comparison}", "",
             "FN: *boundary* if a detection with the same label overlaps the gold span, *wrong label* if only "
             "detections with another label do, else *missed*. FP: *boundary*, *wrong label* or *spurious*, on the "
             "same principle against the gold spans.", "",
             "| Dataset | Detector | " + " | ".join(head) + " |", "|---|---|" + "---:|" * len(head)]
    candidates = defaultdict(list)
    for dataset in SOURCES:
        examples, labels, predictions = load_comparison(dataset, comparison)
        for d in DETECTORS:
            counts = Counter()
            for ex in examples:
                for side, kind, span in classify_errors(ex.text, ex.spans, predictions[d][ex.id], labels):
                    counts[side, kind] += 1
                    candidates[side, kind].append((dataset, d, ex, span, predictions[d][ex.id]))
            fn, fp = (sum(counts[side, k] for k in kinds) for side, kinds in (("FN", FN_TYPES), ("FP", FP_TYPES)))
            if (fn, fp) != saved[(dataset, d)]:
                raise SystemExit(f"{d} on {dataset}: {fn} FN and {fp} FP, but the saved scores have {saved[(dataset, d)]}")
            cells = [fn] + [counts["FN", k] for k in FN_TYPES] + [fp] + [counts["FP", k] for k in FP_TYPES]
            lines.append(f"| {DATASET_NAMES[dataset]} | {DETECTOR_NAMES[d]} | " + " | ".join(map(str, cells)) + " |")
    lines += ["", "The FN and FP totals equal those of the saved scores (checked by the script)."]
    (OUT / f"error_types{suffix(comparison)}.md").write_text("\n".join(lines) + "\n")
    write_examples(comparison, candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", choices=list(COMPARISONS), default="A", help="default: A")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    main_table(args.comparison)
    overmasking(args.comparison)
    error_types(args.comparison)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
