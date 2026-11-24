"""Tests for the noisy copies in benchmark/noise.py."""

import random

import pytest

from benchmark.noise import NOISE_KINDS, choose_edits, example_rng, move_spans, noisy_copy
from PIIFilter.masking import Span

TEXT = "Write to my landlord. I'm Minji Park from 41 Maple Street, call me at 010-2345-6789 or minji.park@example.com."
SPANS = [Span(26, 36, "NAME"), Span(42, 57, "ADDRESS"), Span(70, 83, "PHONE"), Span(87, 109, "EMAIL")]
KINDS = NOISE_KINDS + ("mixed",)


def values(text, spans):
    return [(text[s.start : s.end], s.label) for s in spans]


def test_spans_are_the_values():
    assert values(TEXT, SPANS) == [
        ("Minji Park", "NAME"), ("41 Maple Street", "ADDRESS"), ("010-2345-6789", "PHONE"),
        ("minji.park@example.com", "EMAIL"),
    ]


def test_rate_zero_changes_nothing():
    text, (spans,), edits = noisy_copy(TEXT, [SPANS], "mixed", random.Random(0), rate=0)
    assert (text, spans, edits) == (TEXT, SPANS, [])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(200))
def test_spans_follow_the_text(kind, seed):
    text, (spans,), edits = noisy_copy(TEXT, [SPANS], kind, random.Random(seed), rate=0.3)
    assert len(spans) == len(SPANS)
    for old, new in zip(SPANS, spans):
        assert old.label == new.label
        inside = [e for e in edits if old.start <= e["start"] and e["end"] <= old.end]
        if not inside:  # untouched values move but stay the same
            assert text[new.start : new.end] == TEXT[old.start : old.end]
        else:  # changed values are still there, with the same label
            expected = TEXT[old.start : old.end]
            for e in sorted(inside, key=lambda e: e["start"], reverse=True):
                expected = expected[: e["start"] - old.start] + e["new"] + expected[e["end"] - old.start :]
            assert text[new.start : new.end] == expected.strip()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(200))
def test_no_edit_crosses_a_span_edge_or_another_edit(kind, seed):
    edits = choose_edits(TEXT, [SPANS], kind, random.Random(seed), rate=0.5)
    for e in edits:
        assert e.start < e.end
        for s in SPANS:
            assert e.end <= s.start or e.start >= s.end or (s.start <= e.start and e.end <= s.end)
    for a, b in zip(edits, edits[1:]):
        assert a.end <= b.start


def test_number_of_edits_follows_the_rate():
    _, _, edits = noisy_copy(TEXT, [SPANS], "typo", random.Random(0), rate=0.1)
    assert len(edits) == round(0.1 * len(TEXT.split()))  # 18 words -> 2 edits
    _, _, edits = noisy_copy("Hello there", [[]], "typo", random.Random(0), rate=0.1)
    assert len(edits) == 1  # at least one


def test_each_kind_does_what_it_says():
    rng = random.Random(0)
    text, _, _ = noisy_copy(TEXT, [SPANS], "case", rng, rate=1)
    assert text == TEXT.lower()
    text, _, _ = noisy_copy(TEXT, [SPANS], "punct", rng, rate=1)
    assert text == TEXT.replace("landlord.", "landlord").replace("Street,", "Street").removesuffix(".")
    text, (spans,), _ = noisy_copy(TEXT, [SPANS], "number", rng, rate=1)
    phone = text[spans[2].start : spans[2].end]
    assert phone != "010-2345-6789" and phone.replace(" ", "").replace("-", "") == "01023456789"
    assert "41 Maple Street" in text  # "41" alone is not a number with separators


def test_clean_and_dates_spans_move_together():
    dates_text = "Born 1990-05-12, Minji Park lives here."
    clean = [Span(17, 27, "NAME")]
    dates = [Span(5, 15, "DATE"), Span(17, 27, "NAME")]
    for seed in range(100):
        text, (c, d), _ = noisy_copy(dates_text, [clean, dates], "mixed", random.Random(seed), rate=0.5)
        assert (c[0].start, c[0].end) == (d[1].start, d[1].end)


def test_same_seed_same_copy():
    a = noisy_copy(TEXT, [SPANS], "mixed", example_rng(1, "mixed", "x"))
    b = noisy_copy(TEXT, [SPANS], "mixed", example_rng(1, "mixed", "x"))
    c = noisy_copy(TEXT, [SPANS], "mixed", example_rng(2, "mixed", "x"))
    assert a == b and a[0] != c[0]


def test_mixed_uses_every_kind():
    kinds = set()
    for seed in range(100):
        kinds |= {e["kind"] for e in noisy_copy(TEXT, [SPANS], "mixed", random.Random(seed), rate=0.3)[2]}
    assert kinds == set(NOISE_KINDS)


def test_edit_inside_a_span_is_marked():
    _, _, edits = noisy_copy(TEXT, [SPANS], "case", random.Random(0), rate=1)
    inside = {e["old"]: e["inside"] for e in edits}
    assert inside["Minji"] and inside["Maple"] and not inside["Write"]


def test_move_spans_shifts_by_edits_before_and_inside():
    from benchmark.noise import Edit

    spans = [Span(4, 8, "NAME")]
    assert move_spans(spans, [Edit(0, 1, "", "typo")]) == [Span(3, 7, "NAME")]  # before
    assert move_spans(spans, [Edit(5, 6, "ab", "typo")]) == [Span(4, 9, "NAME")]  # inside
    assert move_spans(spans, [Edit(8, 9, "", "punct")]) == spans  # after


@pytest.mark.parametrize("kind", ["punct", "mixed"])
def test_noise_never_joins_two_values(kind):
    text = "Owner: Roger Leblanc, Roger Leblanc. Emails: a@x.com, b@y.com."
    spans = [Span(7, 20, "NAME"), Span(22, 35, "NAME"), Span(45, 52, "EMAIL"), Span(54, 61, "EMAIL")]
    for seed in range(300):
        new, (moved,), _ = noisy_copy(text, [spans], kind, random.Random(seed), rate=1)
        assert len(moved) == 4
