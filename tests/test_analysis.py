"""Tests for the statistics of benchmark/analysis.py."""

import numpy as np

from benchmark.analysis import (
    bootstrap_totals,
    cluster_ids,
    example_counts,
    holm,
    match_counts,
    per_cluster,
    randomization_test,
    scores,
)
from benchmark.data import Example
from benchmark.scoring import score
from PIIFilter.masking import Span

TEXT = "Hi, I'm Minji Park, email minji@x.com, phone 010-1234-5678."
GOLD = [Span(8, 18, "NAME"), Span(26, 37, "EMAIL"), Span(45, 58, "PHONE")]


def test_scores_follow_the_worked_example_of_the_proposal():
    s = scores(np.array([1, 1, 2, 0]))  # TP, FP, FN, leaked
    assert round(float(s["precision"]), 3) == 0.5
    assert round(float(s["recall"]), 3) == 0.333
    assert round(float(s["f2"]), 3) == 0.357


def test_scores_that_do_not_exist_are_nan_and_nothing_found_is_zero():
    s = scores(np.array([0, 0, 3, 3]))  # nothing detected
    assert np.isnan(s["precision"]) and s["f2"] == 0 and s["leak_rate"] == 1
    assert np.isnan(scores(np.array([0, 2, 0, 0]))["recall"])  # nothing to find


def test_example_counts_add_up_to_the_main_scores():
    examples = [Example("a", TEXT, GOLD), Example("b", TEXT, GOLD[:1])]
    predictions = {"a": [Span(8, 13, "NAME"), Span(26, 37, "EMAIL")], "b": [Span(8, 18, "NAME")]}
    labels = ["NAME", "EMAIL", "PHONE"]
    totals = example_counts(examples, predictions, labels).sum(axis=(0, 1))
    main = score(examples, predictions, labels)["micro"]
    assert (totals[0], totals[1], totals[2]) == (main["tp"], main["fp"], main["fn"])
    assert float(scores(totals)["f2"]) == main["f2"]


def test_linked_examples_share_a_cluster():
    examples = [
        Example("nemotron-us-7", "a", []),
        Example("nemotron-intl-7", "b", []),
        Example("gretel-1", "same text", []),
        Example("gretel-2", "same text", []),
        Example("gretel-3", "other", []),
    ]
    ids = cluster_ids(examples)
    assert ids[0] == ids[1] and ids[2] == ids[3] and len(set(ids)) == 3


def test_bootstrap_is_paired_and_reproducible():
    counts = np.arange(5 * 2 * 4).reshape(5, 2, 4)  # 5 clusters, 2 detectors
    a = bootstrap_totals(counts, draws=50, seed=0)
    b = bootstrap_totals(counts, draws=50, seed=0)
    assert a.shape == (50, 2, 4) and (a == b).all()
    clusters = np.array([0, 0, 1])
    assert per_cluster(np.ones((3, 4), dtype=int), clusters).tolist() == [[2] * 4, [1] * 4]


def test_identical_detectors_are_never_different():
    x = np.array([[3, 1, 1, 1], [2, 0, 2, 2], [5, 2, 0, 0]])
    difference, p = randomization_test(x, x.copy(), rounds=200)
    assert difference == 0 and p == 1


def test_holm_on_a_worked_example():
    # sorted: 0.005, 0.01, 0.03, 0.04 -> 4 x 0.005, 3 x 0.01, 2 x 0.03, max(1 x 0.04, 0.06)
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert [round(a, 3) for a in adjusted] == [0.03, 0.06, 0.06, 0.02]


def test_matching_rules():
    detected = [Span(8, 13, "NAME"), Span(26, 37, "PHONE")]  # "Minji" only; the email with a wrong label
    labels = ["NAME", "EMAIL", "PHONE"]
    assert match_counts(TEXT, GOLD, detected, labels, "strict") == (0, 3, 0, 2)
    assert match_counts(TEXT, GOLD, detected, labels, "same_span") == (1, 3, 1, 2)
    assert match_counts(TEXT, GOLD, detected, labels, "overlap_same_label") == (1, 3, 1, 2)
    assert match_counts(TEXT, GOLD, detected, labels, "overlap_any_label") == (2, 3, 2, 2)
