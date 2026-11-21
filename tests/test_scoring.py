"""Tests for strict matching and the leak rate in benchmark/scoring.py."""

import pytest

from benchmark.data import Example
from benchmark.scoring import score
from PIIFilter.masking import Span

LABELS = ["NAME", "EMAIL", "PHONE"]

# Worked example of the proposal (Table 6).
TEXT = "Hi, I'm Minji Park. Email minji.park@example.com, phone 010-1234-5678."
GOLD = [Span(8, 18, "NAME"), Span(26, 48, "EMAIL"), Span(56, 69, "PHONE")]


def run(detected: list[Span], gold: list[Span] = GOLD, labels: list[str] = LABELS) -> dict:
    return score([Example("x", TEXT, gold)], {"x": detected}, labels)


def test_worked_example_of_table_6():
    s = run([Span(8, 13, "NAME"), Span(26, 48, "EMAIL")])  # "Minji" only; phone missed
    micro = s["micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (1, 1, 2)
    assert micro["precision"] == pytest.approx(0.500, abs=1e-3)
    assert micro["recall"] == pytest.approx(0.333, abs=1e-3)
    assert micro["f1"] == pytest.approx(0.400, abs=1e-3)
    assert micro["f2"] == pytest.approx(0.357, abs=1e-3)
    assert micro["leak_rate"] == pytest.approx(2 / 3)  # part of the name, and the phone
    assert s["per_label"]["EMAIL"]["f2"] == 1.0


def test_perfect_detector():
    micro = run(GOLD)["micro"]
    assert (micro["f1"], micro["f2"], micro["leak_rate"]) == (1.0, 1.0, 0.0)


def test_nothing_detected():
    # Precision is 0 / 0, so it has no value; F2 is a real 0, since everything was missed.
    micro = run([])["micro"]
    assert (micro["precision"], micro["recall"], micro["f2"], micro["leak_rate"]) == (None, 0.0, 0.0, 1.0)


def test_nothing_to_find():
    # No gold PHONE here: recall, F2 and the leak rate have no value; the one detection is an FP.
    phone = run([Span(0, 2, "PHONE")], gold=GOLD[:2])["per_label"]["PHONE"]
    assert (phone["fp"], phone["precision"], phone["recall"], phone["f2"], phone["leak_rate"]) == (1, 0.0, None, None, None)


def test_detections_are_cleaned_like_the_gold():
    split_name = [Span(8, 13, "NAME"), Span(14, 18, "NAME")]  # "Minji" + "Park" -> "Minji Park"
    assert run(split_name)["per_label"]["NAME"]["tp"] == 1


def test_wrong_label_is_an_error_but_not_a_leak():
    s = run([Span(8, 18, "PHONE"), Span(26, 48, "EMAIL"), Span(56, 69, "PHONE")])
    assert (s["per_label"]["NAME"]["fn"], s["per_label"]["PHONE"]["fp"]) == (1, 1)
    assert s["micro"]["leak_rate"] == 0.0  # the name was masked, under the wrong label


def test_labels_the_dataset_lacks_are_not_scored():
    s = run(GOLD + [Span(0, 2, "DATE_OF_BIRTH")])  # DATE_OF_BIRTH is not in LABELS
    assert "DATE_OF_BIRTH" not in s["per_label"]
    assert s["micro"]["fp"] == 0


def test_missing_prediction_is_an_error():
    with pytest.raises(ValueError):
        score([Example("x", TEXT, GOLD)], {}, LABELS)
