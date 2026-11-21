"""Strict-matching scores (proposal, Section 4.3) and the leak rate.

Strict matching: a detected span is a true positive (TP) only if its start, end and label
are exactly those of a gold span. Every other detection is a false positive (FP), and every
gold span without an exact match is a false negative (FN). A partial span is one FP plus
one FN: finding "Minji" in "Minji Park" still sends "Park" to the LLM.

    P = TP / (TP + FP)    R = TP / (TP + FN)    F1 = 2PR / (P + R)    F2 = 5PR / (4P + R)

Leak rate: the share of gold spans that the detected spans, whatever their label, do not
fully cover. Those values, or part of them, would reach the LLM in clear.

Detected spans are cleaned with the same rules as the gold spans (benchmark/spans.py).
Only the labels the dataset has are scored; detections with other labels are dropped.

A score that does not exist is None (null in JSON), not 0: precision when nothing was
detected (0 / 0), and recall, F1, F2 and the leak rate when there was nothing to find.
F1 and F2 are 0 when there was something to find and none of it was found.
"""

from dataclasses import dataclass

from PIIFilter.masking import Span

from .data import Example
from .spans import normalize_spans


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    leaked: int = 0  # gold spans not fully covered by the detections

    @property
    def support(self) -> int:
        """Number of gold spans."""
        return self.tp + self.fn


def ratio(a: int, b: int) -> float | None:
    """a / b, or None when b is 0: the score does not exist (e.g. precision with no detection)."""
    return a / b if b else None


def f_score(p: float | None, r: float | None, beta: float) -> float | None:
    """F-beta: beta = 1 weighs precision and recall equally, beta = 2 counts recall twice."""
    if r is None:  # nothing to find
        return None
    if not p or not r:  # nothing correct found (p is None when nothing was detected)
        return 0.0
    b2 = beta**2
    return (1 + b2) * p * r / (b2 * p + r)


def fmt(value: float | None, interval: tuple[float, float] | None = None) -> str:
    """Three decimals, with an optional interval; "n/a" for a score that does not exist."""
    if value is None or value != value:  # None here, NaN in benchmark/analysis.py
        return "n/a"
    return f"{value:.3f}" + (f" [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "")


def metrics(c: Counts) -> dict:
    p = ratio(c.tp, c.tp + c.fp)
    r = ratio(c.tp, c.tp + c.fn)
    return {
        "support": c.support, "tp": c.tp, "fp": c.fp, "fn": c.fn,
        "precision": p, "recall": r, "f1": f_score(p, r, 1), "f2": f_score(p, r, 2),
        "leak_rate": ratio(c.leaked, c.support),
    }


def count_example(text: str, gold: list[Span], detected: list[Span], counts: dict[str, Counts]) -> None:
    """Add one example's TP, FP, FN and leaks to counts, which has one entry per scored label."""
    detected = normalize_spans(text, detected)
    masked = {i for s in detected for i in range(s.start, s.end)}
    gold_keys = {(s.start, s.end, s.label) for s in gold if s.label in counts}
    found_keys = {(s.start, s.end, s.label) for s in detected if s.label in counts}

    for start, end, label in found_keys:
        if (start, end, label) in gold_keys:
            counts[label].tp += 1
        else:
            counts[label].fp += 1
    for start, end, label in gold_keys - found_keys:
        counts[label].fn += 1
    for start, end, label in gold_keys:
        if any(i not in masked for i in range(start, end)):
            counts[label].leaked += 1


def score(examples: list[Example], predictions: dict[str, list[Span]], labels: list[str]) -> dict:
    """Per-label and micro-averaged scores of one detector on one test set.

    predictions maps each example id to the spans the detector found in its text.
    Micro average: TP, FP, FN and leaks are summed over the labels before computing scores.
    """
    missing = [ex.id for ex in examples if ex.id not in predictions]
    if missing:
        raise ValueError(f"{len(missing)} of {len(examples)} examples have no prediction (first: {missing[0]})")

    counts = {label: Counts() for label in labels}
    for ex in examples:
        count_example(ex.text, ex.spans, predictions[ex.id], counts)

    total = Counts(
        tp=sum(c.tp for c in counts.values()),
        fp=sum(c.fp for c in counts.values()),
        fn=sum(c.fn for c in counts.values()),
        leaked=sum(c.leaked for c in counts.values()),
    )
    return {"per_label": {label: metrics(c) for label, c in counts.items()}, "micro": metrics(total)}
