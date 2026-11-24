"""LaTeX tables of the key results of questions 1 to 3, for docs/results_tables/.

    uv run python experiments/results_tables.py

Reads results/analysis/significance.json (question 1) and significance_noise.json
(questions 2 and 3) and writes one tabular per file to docs/results_tables/tables/. The
document docs/results_tables/results_tables.tex inputs them:

    cd docs/results_tables && latexmk -pdf results_tables.tex

The tables hold numbers and model names only. Every other word is a macro (\\hModel,
\\kTypo, ...) that the document defines, so the same tables serve a translated document.
"""

import json
from pathlib import Path

from benchmark.data import ANALYSIS_DIR, ROOT, SOURCES
from benchmark.noise import NOISE_KINDS

OUT = ROOT / "docs" / "results_tables" / "tables"
DATASETS = {"ai4privacy": "ai4privacy", "nemotron": "Nemotron-PII", "gretel": "Gretel"}
MODELS = {"presidio": "Presidio", "bert": "BERT", "gliner": "GLiNER",
          "bert-clean": r"\mFTclean", "bert-noisy": r"\mFTnoisy"}
DETECTORS = ["presidio", "bert", "gliner"]
KINDS = NOISE_KINDS + ("mixed",)
KIND_MACROS = {k: "\\k" + k.capitalize() for k in KINDS}  # \kTypo, \kSpace, ...
NOISY = [f"noisy_{k}" for k in KINDS]

q1 = json.loads((ANALYSIS_DIR / "significance.json").read_text())
q23 = json.loads((ANALYSIS_DIR / "significance_noise.json").read_text())
TESTS = {(t["family"], t["comparison"], t["dataset"], t["model"], t["version"], t["metric"]): t for t in q23["tests"]}


def score(comparison, dataset, model, version, metric):
    return q23["scores"][f"{comparison}/{dataset}"]["models"][model][version][metric]


def established(t: dict) -> bool:
    return t["verdict"] not in ("not established", "detectable but negligible")


def best(comparison: str, dataset: str, metric: str) -> str | None:
    """The detector of question 1 established as better than both others (as in paper_results.py)."""
    verdicts = {frozenset(t["pair"]): t["verdict"] for t in q1["tests"]
                if (t["comparison"], t["dataset"], t["metric"]) == (comparison, dataset, metric)}
    for d in DETECTORS:
        if all(verdicts[frozenset((d, o))] == f"{d} better" for o in DETECTORS if o != d):
            return d
    return None


def num(v: float) -> str:
    return f"{v:.3f}"


def signed(v: float, scale: float = 1, digits: int = 3) -> str:
    v = round(v * scale, digits)
    if v == 0:
        return f"{0:.{digits}f}"
    return ("$+$" if v > 0 else "$-$") + f"{abs(v):.{digits}f}"


def plain(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}".replace("-", "$-$")


def ci(low: float, high: float, scale: float = 1, digits: int = 3) -> str:
    return rf"{{\scriptsize[{plain(low * scale, digits)}, {plain(high * scale, digits)}]}}"


def ns(text: str, faint: bool) -> str:
    return rf"\ns{{{text}}}" if faint else text


def bold(text: str, on: bool) -> str:
    return rf"\textbf{{{text}}}" if on else text


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / name}.tex")


def check_question_1() -> None:
    """On the clean test sets, both analyses use the same draws: the intervals must agree."""
    for key, r in q1["scores"].items():
        for d in DETECTORS:
            for metric, values in r["detectors"][d]["micro"].items():
                ours = q23["scores"][key]["models"][d]["clean"][metric]
                assert all(abs(a - b) < 1e-9 for a, b in zip(values, ours)), (key, d, metric, values, ours)


def clean_table() -> None:
    """Table 1: every model on the clean test sets, comparison A."""
    lines = [r"\begin{tabular}{llrrrll}", r"\toprule",
             r"\hDataset & \hModel & P & R & F1 & F2 \hCI & \hLeak{} \hCI \\", r"\midrule"]
    for k, dataset in enumerate(DATASETS):
        top = {m: best("A", dataset, m) for m in ("f1", "f2", "leak_rate")}
        for j, model in enumerate(MODELS):
            if model == "bert-clean":
                lines.append(r"\cmidrule(l){2-7}")
            s = {m: score("A", dataset, model, "clean", m) for m in ("precision", "recall", "f1", "f2", "leak_rate")}
            cells = [num(s["precision"][0]), num(s["recall"][0]), bold(num(s["f1"][0]), top["f1"] == model),
                     bold(num(s["f2"][0]), top["f2"] == model) + " " + ci(*s["f2"][1:]),
                     bold(f"{100 * s['leak_rate'][0]:.1f}", top["leak_rate"] == model) + " " + ci(*s["leak_rate"][1:], 100, 1)]
            lines.append(f"{DATASETS[dataset] if j == 0 else ''} & {MODELS[model]} & " + " & ".join(cells) + r" \\")
        if k < len(DATASETS) - 1:
            lines.append(r"\midrule")
    write("clean_A", lines + [r"\bottomrule", r"\end{tabular}"])


def other_comparisons_table() -> None:
    """Table 2: F2 on the clean test sets in comparisons B, A-dates and B-dates."""
    comparisons = ["B", "A-dates", "B-dates"]
    lines = [r"\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{DATASETS[d]}}}" for d in DATASETS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DATASETS))),
             r"\hModel & " + " & ".join(r"\cB & \cAd & \cBd" for _ in DATASETS) + r" \\", r"\midrule"]
    top = {(c, d): best(c, d, "f2") for c in comparisons for d in DATASETS}
    for model in MODELS:
        if model == "bert-clean":
            lines.append(r"\midrule")
        cells = [bold(num(score(c, d, model, "clean", "f2")[0]), top[(c, d)] == model) for d in DATASETS for c in comparisons]
        lines.append(f"{MODELS[model]} & " + " & ".join(cells) + r" \\")
    write("clean_other", lines + [r"\bottomrule", r"\end{tabular}"])


def change_table(name: str, comparison: str, metric: str, scale: float, digits: int, gap: bool) -> None:
    """Question 2: the score on the clean test set and its change on each noisy copy."""
    lines = [r"\begin{tabular}{ll" + "r" * (1 + len(KINDS)) + "}", r"\toprule",
             r"& & & \multicolumn{" + str(len(KINDS)) + r"}{c}{\hChange} \\",
             rf"\cmidrule(l){{4-{3 + len(KINDS)}}}",
             r"\hDataset & \hModel & \hClean & " + " & ".join(KIND_MACROS.values()) + r" \\", r"\midrule"]
    for k, dataset in enumerate(DATASETS):
        for j, model in enumerate(DETECTORS):
            cells = [f"{score(comparison, dataset, model, 'clean', metric)[0] * scale:.{digits}f}"]
            for version in NOISY:
                t = TESTS[("q2_change", comparison, dataset, model, version, metric)]
                cells.append(ns(signed(t["difference"], scale, digits), not established(t)))
            lines.append(f"{DATASETS[dataset] if j == 0 else ''} & {MODELS[model]} & " + " & ".join(cells) + r" \\")
        if gap:
            cells = []
            for version in NOISY:
                value, low, high = q23["scores"][f"{comparison}/{dataset}"]["change_gap"][version][metric]
                cells.append(ns(signed(value, scale, digits), low <= 0 <= high))
            lines.append(r"& \textit{\hGap} & & " + " & ".join(rf"\textit{{{c}}}" for c in cells) + r" \\")
        if k < len(DATASETS) - 1:
            lines.append(r"\midrule")
    write(name, lines + [r"\bottomrule", r"\end{tabular}"])


def fine_tuned_table(name: str, comparison: str) -> None:
    """Question 3: F2 of both fine-tuned versions on every test set, and their difference."""
    versions = ["clean"] + NOISY
    lines = [r"\begin{tabular}{ll" + "r" * len(versions) + "}", r"\toprule",
             r"\hDataset & \hModel & \hClean & " + " & ".join(KIND_MACROS.values()) + r" \\", r"\midrule"]
    for k, dataset in enumerate(DATASETS):
        for j, model in enumerate(["bert", "bert-clean", "bert-noisy"]):
            cells = [num(score(comparison, dataset, model, v, "f2")[0]) for v in versions]
            if model == "bert":
                cells = [rf"\refv{{{c}}}" for c in cells]
                label = r"\mOriginal"
            else:
                label = MODELS[model]
            lines.append(f"{DATASETS[dataset] if j == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
        cells = []
        for v in versions:
            t = TESTS[("q3_difference", comparison, dataset, "bert-noisy - bert-clean", v, "f2")]
            cells.append(ns(r"\textbf{" + signed(t["difference"]) + "}" if established(t) else signed(t["difference"]), not established(t)))
        lines.append(r"& \hDiff & " + " & ".join(cells) + r" \\")
        if k < len(DATASETS) - 1:
            lines.append(r"\midrule")
    write(name, lines + [r"\bottomrule", r"\end{tabular}"])


def won_back_table() -> None:
    """Question 3 in short: loss, gain and cost with 95% intervals (comparison A, F2), and leak rates."""
    lines = [r"\begin{tabular}{lllll}", r"\toprule",
             r"\hDataset & \hLossMixed & \hWonMixed & \hCleanCost & \hLeakMixed \\", r"\midrule"]
    for dataset in DATASETS:
        loss = TESTS[("ft_change", "A", dataset, "bert-clean", "noisy_mixed", "f2")]
        won = TESTS[("q3_difference", "A", dataset, "bert-noisy - bert-clean", "noisy_mixed", "f2")]
        cost = TESTS[("q3_difference", "A", dataset, "bert-noisy - bert-clean", "clean", "f2")]
        leak = {m: score("A", dataset, m, "noisy_mixed", "leak_rate")[0] for m in ("bert-clean", "bert-noisy")}
        leak_test = TESTS[("q3_difference", "A", dataset, "bert-noisy - bert-clean", "noisy_mixed", "leak_rate")]
        share = won["difference"] / -loss["difference"] if -loss["difference"] > 0.005 else None
        cells = [
            ns(signed(-loss["difference"]) + " " + ci(-loss["ci"][1], -loss["ci"][0]), not established(loss)),
            ns(signed(won["difference"]) + " " + ci(*won["ci"]), not established(won))
            + (rf" {{\scriptsize({share:.0%})}}".replace("%", r"\%") if share is not None else ""),
            ns(signed(-cost["difference"]) + " " + ci(-cost["ci"][1], -cost["ci"][0]), not established(cost)),
            ns(f"{100 * leak['bert-clean']:.1f} $\\to$ {100 * leak['bert-noisy']:.1f}", not established(leak_test)),
        ]
        lines.append(f"{DATASETS[dataset]} & " + " & ".join(cells) + r" \\")
    write("q3_summary", lines + [r"\bottomrule", r"\end{tabular}"])


def main() -> None:
    check_question_1()
    clean_table()
    other_comparisons_table()
    change_table("q2_f2", "A", "f2", 1, 3, gap=True)
    change_table("q2_leak", "A", "leak_rate", 100, 1, gap=False)
    change_table("q2_f2_dates", "A-dates", "f2", 1, 3, gap=False)
    fine_tuned_table("q3_f2", "A")
    fine_tuned_table("q3_f2_dates", "A-dates")
    won_back_table()


if __name__ == "__main__":
    main()
