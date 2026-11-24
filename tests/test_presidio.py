"""Test for the Presidio detector (rules only). BERT and GLiNER need large model downloads,
so they are checked by experiments/time_detectors.py instead."""

from baselines.presidio_detector import PresidioDetector


def test_presidio_finds_patterns_but_no_names():
    text = "I'm Minji Park. Email minji.park@example.com, card 4111-1111-1111-1111."
    found = {(text[s.start : s.end], s.label) for s in PresidioDetector().detect(text)}
    assert ("minji.park@example.com", "EMAIL") in found
    assert ("4111-1111-1111-1111", "CARD_NUMBER") in found
    assert not any(label == "NAME" for _, label in found)


def test_presidio_dates_are_birth_dates_or_any_date():
    text = "Invoice due 2024-08-15."
    assert [s.label for s in PresidioDetector().detect(text)] == ["DATE_OF_BIRTH"]
    assert [s.label for s in PresidioDetector(dates=True).detect(text)] == ["DATE"]
