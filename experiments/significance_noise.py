"""Confidence intervals and paired tests for questions 2 and 3.

    uv run python experiments/significance_noise.py      # a few minutes (one process per dataset and date version)

Same method as experiments/significance.py (question 1): a paired cluster bootstrap (10,000
draws, seed 0) for the 95% intervals, paired approximate randomization for the tests, and
Holm's correction within each family. The clusters are those of the clean test set, so the
noisy copies of an example are always drawn with it, and every model and test set of a
dataset is scored on the same draws. On the clean test sets the intervals of Presidio, BERT
and GLiNER are therefore those of significance.py.

The fine-tuned BERT has three training seeds per version. Its score is the mean of the
three seeds' scores, in each bootstrap draw and each randomization round too.

For each comparison (A, B, A-dates, B-dates) and each dataset:

- micro P, R, F1, F2 and leak rate of every model on the clean test set and on each noisy
  copy, with a 95% interval;
- question 2: for Presidio, BERT and GLiNER, the change on each noisy copy (noisy minus
  clean). The test swaps, cluster by cluster, the detector's output on the clean and on the
  noisy text;
- question 3: BERT fine-tuned on clean + noisy text minus BERT fine-tuned on clean text, on
  every test set. On a noisy copy this is what the noisy copies won back, on the clean test
  set it is minus the clean cost. The test swaps the two versions' outputs cluster by
  cluster, for the three seed pairs together (seed k with seed k);
- the change on each noisy copy of each fine-tuned version, tested as in question 2;
- BERT's change minus GLiNER's change on each noisy copy (interval only, not tested).

Three families of tests (question 2 changes, question 3 differences, fine-tuned changes),
each one for F1, F2 and leak rate. The families are larger than in question 1, so the
randomization uses 20,000 rounds. A difference is called established, as in question 1,
when its corrected p-value is below 0.05, its interval excludes 0 and it is at least 0.01.
Writes results/analysis/significance_noise.md and .json.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchmark.analysis import (
    COMPARISONS,
    bootstrap_totals,
    cluster_ids,
    example_counts,
    holm,
    interval,
    per_cluster,
    scores,
)
from benchmark.data import ANALYSIS_DIR, PRED_DIR, SOURCES, TEST_DIR, read_jsonl, read_predictions
from benchmark.noise import NOISE_KINDS
from benchmark.scoring import fmt

DRAWS, ROUNDS = 10_000, 20_000
ALPHA, SMALLEST = 0.05, 0.01
METRICS = ["precision", "recall", "f1", "f2", "leak_rate"]
TESTED = ["f1", "f2", "leak_rate"]
KINDS = NOISE_KINDS + ("mixed",)
VERSIONS = ["clean"] + [f"noisy_{k}" for k in KINDS]  # without the _dates suffix
SEEDS = (0, 1, 2)
MODELS = {  # model -> its runs (prediction folders)
    "presidio": ["presidio"],
    "bert": ["bert"],
    "gliner": ["gliner"],
    "bert-clean": [f"bert-clean-s{s}" for s in SEEDS],
    "bert-noisy": [f"bert-noisy-s{s}" for s in SEEDS],
}
RUNS = [run for runs in MODELS.values() for run in runs]
DETECTORS = ["presidio", "bert", "gliner"]  # question 2
FINE_TUNED = ["bert-clean", "bert-noisy"]  # question 3
FAMILIES = {"q2_change": "Question 2: change on the noisy copy (noisy minus clean)",
            "q3_difference": "Question 3: noisy-trained minus clean-trained BERT",
            "ft_change": "Fine-tuned BERT: change on the noisy copy (noisy minus clean)"}


def model_scores(totals: np.ndarray, model: str) -> dict[str, np.ndarray]:
    """Scores of a model from totals (..., runs, 4): the mean over its runs (training seeds)."""
    idx = [RUNS.index(run) for run in MODELS[model]]
    per_run = scores(totals[..., idx, :])
    return {m: np.nanmean(v, axis=-1) for m, v in per_run.items()}


def randomization(x: np.ndarray, y: np.ndarray, rounds: int = ROUNDS, seed: int = 0, chunk: int = 500) -> dict:
    """Paired approximate randomization test of score(x) - score(y), for each metric of TESTED.

    x and y are per-cluster counts (clusters, seeds, 4); a score is the mean over the seeds.
    In each round the outputs of x and y are swapped in each cluster with probability 1/2,
    for all seeds at once. With one seed this is benchmark.analysis.randomization_test.
    Returns {metric: (difference, p-value)}, p = (c + 1) / (rounds + 1), two-sided.
    """
    def stat(tx, ty):
        sx, sy = scores(tx), scores(ty)
        return {m: np.nanmean(sx[m], axis=-1) - np.nanmean(sy[m], axis=-1) for m in TESTED}

    n, shape = x.shape[0], x.shape[1:]
    observed = {m: float(v) for m, v in stat(x.sum(0), y.sum(0)).items()}
    gap = (y - x).reshape(n, -1).astype(float)
    total_x, total_y = x.sum(0), y.sum(0)
    rng = np.random.default_rng(seed)
    hits = dict.fromkeys(TESTED, 0)
    for start in range(0, rounds, chunk):
        size = min(chunk, rounds - start)
        swap = rng.integers(0, 2, size=(size, n)).astype(float)
        moved = np.rint(swap @ gap).reshape(size, *shape)  # what moves from y to x in each round
        diff = stat(total_x + moved, total_y - moved)
        for m in TESTED:
            hits[m] += int(np.sum(np.abs(diff[m]) >= abs(observed[m]) - 1e-12))
    return {m: (observed[m], (hits[m] + 1) / (rounds + 1) if observed[m] == observed[m] else float("nan"))
            for m in TESTED}


def cluster_counts(dataset: str, dates: bool) -> dict[str, tuple[np.ndarray, int]]:
    """Per-cluster micro counts (clusters, versions, runs, 4) of each comparison of this date version."""
    suffix = "_dates" if dates else ""
    clean = read_jsonl(TEST_DIR / f"{dataset}_clean{suffix}.jsonl")
    ids = [ex.id for ex in clean]
    clusters = cluster_ids(clean)
    labels = json.loads((TEST_DIR / f"{dataset}_clean{suffix}.meta.json").read_text())["labels"]
    counts = np.zeros((len(clean), len(VERSIONS), len(RUNS), len(labels), 4), dtype=np.int64)
    for v, version in enumerate(VERSIONS):
        examples = clean if version == "clean" else read_jsonl(TEST_DIR / f"{dataset}_{version}{suffix}.jsonl")
        assert [ex.id for ex in examples] == ids, f"{dataset} {version}{suffix}: not the examples of the clean test set"
        for r, run in enumerate(RUNS):
            predictions = read_predictions(PRED_DIR / run / dataset / f"{version}{suffix}.jsonl")
            counts[:, v, r] = example_counts(examples, predictions, labels)
    out = {}
    for comparison, (base, keep) in COMPARISONS.items():
        if base != f"clean{suffix}":
            continue
        kept = [j for j, label in enumerate(labels) if keep is None or label in keep]
        micro = counts[:, :, :, kept].sum(axis=3)  # (examples, versions, runs, 4)
        out[comparison] = (per_cluster(micro, clusters), int(clusters.max() + 1))
    return out


def analyse(job: tuple[str, bool]) -> tuple[dict, list[dict]]:
    dataset, dates = job
    results, tests = {}, []
    for comparison, (by_cluster, n_clusters) in cluster_counts(dataset, dates).items():
        print(f"{comparison} {dataset}: counted, testing ...", flush=True)
        draws = bootstrap_totals(by_cluster, DRAWS, seed=0)  # (draws, versions, runs, 4)
        point = by_cluster.sum(axis=0)  # (versions, runs, 4)
        pt = {m: model_scores(point, m) for m in MODELS}  # model -> metric -> (versions,)
        dr = {m: model_scores(draws, m) for m in MODELS}  # model -> metric -> (draws, versions)

        result = {"clusters": n_clusters, "models": {}}
        for model in MODELS:
            result["models"][model] = {
                version: {metric: [float(pt[model][metric][v]), *interval(dr[model][metric][:, v])] for metric in METRICS}
                for v, version in enumerate(VERSIONS)
            }
        # BERT's change minus GLiNER's change (interval only)
        result["change_gap"] = {}
        for v, version in enumerate(VERSIONS[1:], start=1):
            result["change_gap"][version] = {}
            for metric in TESTED:
                change = {d: (pt[d][metric][v] - pt[d][metric][0], dr[d][metric][:, v] - dr[d][metric][:, 0]) for d in ("bert", "gliner")}
                result["change_gap"][version][metric] = [float(change["bert"][0] - change["gliner"][0]),
                                                         *interval(change["bert"][1] - change["gliner"][1])]
        results[f"{comparison}/{dataset}"] = result

        def runs(model):
            return [RUNS.index(run) for run in MODELS[model]]

        def add(family, model, version, x, y, point_diff, draw_diff):
            for metric, (diff, p) in randomization(x, y).items():
                assert abs(diff - point_diff[metric]) < 1e-9
                tests.append({"family": family, "comparison": comparison, "dataset": dataset, "model": model,
                              "version": version, "metric": metric, "difference": diff,
                              "ci": list(interval(draw_diff[metric])), "p": p})

        for model in DETECTORS + FINE_TUNED:
            family = "q2_change" if model in DETECTORS else "ft_change"
            for v, version in enumerate(VERSIONS[1:], start=1):
                x, y = by_cluster[:, v][:, runs(model)], by_cluster[:, 0][:, runs(model)]
                add(family, model, version, x, y,
                    {m: pt[model][m][v] - pt[model][m][0] for m in TESTED},
                    {m: dr[model][m][:, v] - dr[model][m][:, 0] for m in TESTED})
        for v, version in enumerate(VERSIONS):
            x, y = by_cluster[:, v][:, runs("bert-noisy")], by_cluster[:, v][:, runs("bert-clean")]
            add("q3_difference", "bert-noisy - bert-clean", version, x, y,
                {m: pt["bert-noisy"][m][v] - pt["bert-clean"][m][v] for m in TESTED},
                {m: dr["bert-noisy"][m][:, v] - dr["bert-clean"][m][:, v] for m in TESTED})
    return results, tests


def verdict(t: dict) -> str:
    low, high = t["ci"]
    if t["p_holm"] < ALPHA and (low > 0 or high < 0) and abs(t["difference"]) >= SMALLEST:
        better = (t["difference"] > 0) != (t["metric"] == "leak_rate")  # a lower leak rate is better
        if t["family"] == "q3_difference":
            return "noisy-trained better" if better else "clean-trained better"
        return "better on noisy" if better else "worse on noisy"
    if t["p_holm"] < ALPHA:
        return "detectable but negligible"
    return "not established"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    jobs = [(dataset, dates) for dataset in SOURCES for dates in (False, True)]
    results, tests = {}, []
    with ProcessPoolExecutor(len(jobs)) as pool:
        for new_results, new_tests in pool.map(analyse, jobs):
            results |= new_results
            tests += new_tests
    order = {c: i for i, c in enumerate(COMPARISONS)}
    results = dict(sorted(results.items(), key=lambda kv: (order[kv[0].split("/")[0]], list(SOURCES).index(kv[0].split("/")[1]))))
    tests.sort(key=lambda t: (list(FAMILIES).index(t["family"]), order[t["comparison"]], list(SOURCES).index(t["dataset"])))
    for family in FAMILIES:
        for metric in TESTED:
            group = [t for t in tests if (t["family"], t["metric"]) == (family, metric)]
            for t, adjusted in zip(group, holm([t["p"] for t in group])):
                t["p_holm"] = adjusted
                t["verdict"] = verdict(t)
    sizes = {f: sum(t["family"] == f for t in tests) // len(TESTED) for f in FAMILIES}

    lines = ["# Questions 2 and 3: scores with 95% intervals and paired tests", "",
             f"Paired cluster bootstrap ({DRAWS} draws) and approximate randomization ({ROUNDS} rounds), seed 0, "
             "clusters of the clean test set. Fine-tuned BERT: mean of the three training seeds. "
             "Holm's correction within each family: " + ", ".join(f"{FAMILIES[f].split(':')[0]} {sizes[f]} tests" for f in FAMILIES)
             + " (for each of F1, F2 and leak rate).", "",
             "Established: corrected p below 0.05, interval excluding 0, and a difference of at least 0.01.", ""]
    for comparison in COMPARISONS:
        lines += [f"## Comparison {comparison}: scores [95% CI]", "",
                  "| Dataset | Model | Test set | P | R | F1 | F2 | Leak rate |", "|---|---|---|---|---|---|---|---|"]
        for dataset in SOURCES:
            for model, by_version in results[f"{comparison}/{dataset}"]["models"].items():
                for version, m in by_version.items():
                    lines.append(f"| {dataset} | {model} | {version} | " + " | ".join(fmt(m[k][0], m[k][1:]) for k in METRICS) + " |")
        lines.append("")
    for family, title in FAMILIES.items():
        for metric, name in (("f2", "micro F2"), ("f1", "micro F1"), ("leak_rate", "leak rate")):
            lines += [f"## {title}, {name}", "",
                      "| Comparison | Dataset | Model | Test set | Difference [95% CI] | p | p (Holm) | Verdict |",
                      "|---|---|---|---|---|---|---|---|"]
            for t in (t for t in tests if (t["family"], t["metric"]) == (family, metric)):
                lines.append(f"| {t['comparison']} | {t['dataset']} | {t['model']} | {t['version']} | "
                             f"{fmt(t['difference'], t['ci'])} | {t['p']:.5f} | {t['p_holm']:.5f} | {t['verdict']} |")
            lines.append("")
    lines += ["## BERT's change minus GLiNER's change (interval only, not tested)", "",
              "Negative F2 or F1: BERT loses more than GLiNER. Positive leak rate: BERT's leak rate rises more.", "",
              "| Comparison | Dataset | Test set | F2 | F1 | Leak rate |", "|---|---|---|---|---|---|"]
    for key, r in results.items():
        comparison, dataset = key.split("/")
        for version, gap in r["change_gap"].items():
            lines.append(f"| {comparison} | {dataset} | {version} | " + " | ".join(fmt(gap[m][0], gap[m][1:]) for m in ("f2", "f1", "leak_rate")) + " |")
    lines.append("")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "significance_noise.json").write_text(json.dumps({"scores": results, "tests": tests}, indent=1) + "\n")
    (ANALYSIS_DIR / "significance_noise.md").write_text("\n".join(lines) + "\n")
    print(f"Saved to {ANALYSIS_DIR / 'significance_noise.md'} and .json")


if __name__ == "__main__":
    main()
