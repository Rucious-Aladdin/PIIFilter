"""Statistics for the analysis of the benchmark scores (confidence intervals and tests).

Every score here is computed from per-example counts made with the rules of
benchmark/scoring.py, so the resampled scores use exactly the same definitions as the
main scores. The steps:

1. example_counts(): TP, FP, FN and leaked spans of each example, label by label;
2. cluster_ids(): examples that are not independent are grouped (the two locales of a
   Nemotron uid, identical texts), and the group is the unit that gets resampled;
3. bootstrap_totals(): paired cluster bootstrap. The same draws are used for every
   detector, so differences between detectors can be read on the same samples;
4. randomization_test(): paired approximate randomization test of a difference;
5. holm(): Holm's correction for multiple tests.

match_counts() gives the looser matching rules of the sensitivity analysis.
"""

import json

import numpy as np

import baselines
from PIIFilter.masking import Span

from .data import PRED_DIR, TEST_DIR, Example, read_jsonl, read_predictions
from .scoring import Counts, count_example
from .spans import normalize_spans

TP, FP, FN, LEAKED = range(4)
DETECTORS = list(baselines.DETECTORS)
PAIRS = [("presidio", "bert"), ("presidio", "gliner"), ("bert", "gliner")]

# The four comparisons of question 1: test set version, and the labels kept (None: all).
COMPARISONS = {
    "A": ("clean", None),
    "B": ("clean", ["EMAIL", "PHONE", "DATE_OF_BIRTH", "CARD_NUMBER"]),
    "A-dates": ("clean_dates", None),
    "B-dates": ("clean_dates", ["EMAIL", "PHONE", "DATE", "CARD_NUMBER"]),
}


def load_comparison(dataset: str, comparison: str) -> tuple[list[Example], list[str], dict[str, dict]]:
    """Test examples, scored labels and each detector's predictions for one comparison."""
    version, keep = COMPARISONS[comparison]
    path = TEST_DIR / f"{dataset}_{version}.jsonl"
    labels = json.loads(path.with_suffix(".meta.json").read_text())["labels"]
    labels = [label for label in labels if keep is None or label in keep]
    predictions = {d: read_predictions(PRED_DIR / d / dataset / f"{version}.jsonl") for d in DETECTORS}
    return read_jsonl(path), labels, predictions


def example_counts(examples: list[Example], predictions: dict[str, list[Span]], labels: list[str]) -> np.ndarray:
    """Array (examples, labels, 4): TP, FP, FN and leaked gold spans of each example and label."""
    out = np.zeros((len(examples), len(labels), 4), dtype=np.int64)
    for i, ex in enumerate(examples):
        counts = {label: Counts() for label in labels}
        count_example(ex.text, ex.spans, predictions[ex.id], counts)
        for j, label in enumerate(labels):
            c = counts[label]
            out[i, j] = (c.tp, c.fp, c.fn, c.leaked)
    return out


def cluster_ids(examples: list[Example]) -> np.ndarray:
    """One integer per example; examples that share a Nemotron uid or the same text share it."""
    keys = {}
    ids = []
    for ex in examples:
        if ex.id.startswith("nemotron-"):
            key = ("uid", ex.id.split("-", 2)[2])  # nemotron-<locale>-<uid>
        else:
            key = ("text", ex.text)
        ids.append(keys.setdefault(key, len(keys)))
    return np.array(ids)


def per_cluster(counts: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Sum the per-example counts of each cluster (first axis)."""
    out = np.zeros((clusters.max() + 1, *counts.shape[1:]), dtype=np.int64)
    np.add.at(out, clusters, counts)
    return out


def scores(totals: np.ndarray) -> dict[str, np.ndarray]:
    """P, R, F1, F2 and leak rate from summed counts (last axis: TP, FP, FN, leaked).

    Same conventions as benchmark/scoring.py: a score that does not exist is NaN (None
    there); F1 and F2 are 0 when there was something to find and nothing correct was found.
    """
    tp, fp, fn, leaked = (totals[..., k].astype(float) for k in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        r = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        leak = np.where(tp + fn > 0, leaked / (tp + fn), np.nan)

        def f(beta: float) -> np.ndarray:
            value = (1 + beta**2) * p * r / (beta**2 * p + r)
            zero = np.isnan(p) | (p == 0) | (r == 0)
            return np.where(np.isnan(r), np.nan, np.where(zero, 0.0, value))

        return {"precision": p, "recall": r, "f1": f(1), "f2": f(2), "leak_rate": leak}


def bootstrap_totals(cluster_counts: np.ndarray, draws: int = 10_000, seed: int = 0, chunk: int = 200) -> np.ndarray:
    """Paired cluster bootstrap: (draws, *rest) totals, each one the sum over a resample of clusters.

    cluster_counts has one row per cluster; its other axes (detectors, labels, counts) are
    kept, so every detector is summed over the same resampled clusters.
    """
    n = cluster_counts.shape[0]
    flat = cluster_counts.reshape(n, -1).astype(float)  # float products are exact for counts, and faster
    rng = np.random.default_rng(seed)
    out = np.empty((draws, flat.shape[1]), dtype=np.int64)
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=size)  # how many times each cluster is drawn
        out[start : start + size] = np.rint(weights.astype(float) @ flat)
    return out.reshape(draws, *cluster_counts.shape[1:])


def interval(values: np.ndarray) -> tuple[float, float]:
    """95% percentile interval, ignoring the draws where the score does not exist."""
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def randomization_test(x: np.ndarray, y: np.ndarray, metric: str = "f2", rounds: int = 10_000, seed: int = 0,
                       chunk: int = 500) -> tuple[float, float]:
    """Paired approximate randomization test of score(x) - score(y). Returns (difference, p-value).

    x and y are the per-cluster counts (clusters, 4) of two detectors on the same test set,
    already summed over the labels being scored. In each round, the outputs of the two
    detectors are swapped in each cluster with probability 1/2. The p-value is the share of
    rounds whose difference is at least as large, in absolute value, as the observed one
    (two-sided), counting the observed difference itself: (c + 1) / (rounds + 1).
    """
    observed = float(scores(x.sum(0))[metric] - scores(y.sum(0))[metric])
    gap = (y - x).astype(float)
    total_x, total_y = x.sum(0), y.sum(0)
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, rounds, chunk):
        size = min(chunk, rounds - start)
        swap = rng.integers(0, 2, size=(size, x.shape[0])).astype(float)
        moved = np.rint(swap @ gap)  # what moves from y to x in each round
        diff = scores(total_x + moved)[metric] - scores(total_y - moved)[metric]
        hits += int(np.sum(np.abs(diff) >= abs(observed) - 1e-12))
    return observed, (hits + 1) / (rounds + 1)


def holm(p_values: list[float]) -> list[float]:
    """Holm's adjusted p-values, in the input order. A test is significant if its value is <= alpha."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def match_counts(text: str, gold: list[Span], detected: list[Span], labels: list[str], rule: str) -> tuple[int, int, int, int]:
    """(gold found, gold, detections that count, detections) under a matching rule.

    Detections are cleaned like in scoring.py, and only the scored labels count.
    - "strict": same start, end and label (the main rule);
    - "same_span": same start and end, any label;
    - "overlap_same_label": the two spans share at least one character and the label;
    - "overlap_any_label": the two spans share at least one character.
    Recall = gold found / gold, precision = detections that count / detections.
    """
    gold = [g for g in gold if g.label in labels]
    found = [s for s in normalize_spans(text, detected) if s.label in labels]

    def hit(a: Span, b: Span) -> bool:
        if rule == "strict":
            return (a.start, a.end, a.label) == (b.start, b.end, b.label)
        if rule == "same_span":
            return (a.start, a.end) == (b.start, b.end)
        overlap = a.start < b.end and b.start < a.end
        return overlap and (rule == "overlap_any_label" or a.label == b.label)

    gold_found = sum(any(hit(g, s) for s in found) for g in gold)
    found_ok = sum(any(hit(g, s) for g in gold) for s in found)
    return gold_found, len(gold), found_ok, len(found)


# Kinds of strict errors (experiments/paper_results.py).
FN_TYPES = ("missed", "boundary", "wrong_label")
FP_TYPES = ("spurious", "boundary", "wrong_label")


def error_type(span: Span, others: list[Span], no_overlap: str) -> str:
    """"boundary" if a span of `others` with the same label overlaps `span`, "wrong_label" if
    only spans with other labels overlap it, else `no_overlap`."""
    near = [o for o in others if o.start < span.end and span.start < o.end]
    if any(o.label == span.label for o in near):
        return "boundary"
    return "wrong_label" if near else no_overlap


def classify_errors(text: str, gold: list[Span], detected: list[Span], labels: list[str]) -> list[tuple[str, str, Span]]:
    """Every strict error of one example, as (side, type, span) with side "FN" or "FP".

    The errors are exactly the FN and FP of benchmark/scoring.py. A gold span is compared
    with all the detections, a detection with all the gold spans of the test set.
    """
    found = normalize_spans(text, detected)
    gold_keys = {(s.start, s.end, s.label) for s in gold if s.label in labels}
    found_keys = {(s.start, s.end, s.label) for s in found if s.label in labels}
    errors = [("FN", error_type(Span(*key), found, "missed"), Span(*key)) for key in sorted(gold_keys - found_keys)]
    errors += [("FP", error_type(Span(*key), gold, "spurious"), Span(*key)) for key in sorted(found_keys - gold_keys)]
    return errors


def overmasked_chars(text: str, raw_gold: list[tuple[int, int, str]], detected: list[Span]) -> tuple[int, int]:
    """(masked, total) non-space characters outside every gold span of any dataset label.

    raw_gold holds the dataset's own spans, including the labels we do not score (city, time...).
    Every detection counts, because the filter masks all of them.
    """
    in_gold = {i for start, end, _ in raw_gold for i in range(start, end)}
    masked = {i for s in normalize_spans(text, detected) for i in range(s.start, s.end)}
    outside = [i for i, char in enumerate(text) if not char.isspace() and i not in in_gold]
    return sum(i in masked for i in outside), len(outside)
