"""Tests for mask() and unmask() in PIIFilter/masking.py."""

from PIIFilter.masking import Span, mask, unmask

TEXT = "I am Minji Park, call me at 010-2345-6789."
SPANS = [Span(5, 15, "NAME"), Span(28, 41, "PHONE")]


def test_mask_replaces_each_span_with_a_placeholder():
    masked, mapping = mask(TEXT, SPANS)
    assert masked == "I am <NAME_1>, call me at <PHONE_1>."
    assert mapping == {"<NAME_1>": "Minji Park", "<PHONE_1>": "010-2345-6789"}


def test_unmask_restores_the_original_text():
    masked, mapping = mask(TEXT, SPANS)
    assert unmask(masked, mapping) == TEXT


def test_same_value_gets_the_same_placeholder():
    text = "Minji Park and Minji Park"
    masked, mapping = mask(text, [Span(0, 10, "NAME"), Span(15, 25, "NAME")])
    assert masked == "<NAME_1> and <NAME_1>"
    assert mapping == {"<NAME_1>": "Minji Park"}


def test_overlapping_span_is_dropped():
    spans = [Span(5, 15, "NAME"), Span(11, 15, "NAME"), Span(28, 41, "PHONE")]
    masked, _ = mask(TEXT, spans)
    assert masked == "I am <NAME_1>, call me at <PHONE_1>."


def test_date_placeholder_is_restored():
    masked, mapping = mask("Born 3 May", [Span(5, 10, "DATE")])
    assert masked == "Born <DATE_1>"
    assert unmask("You were born on <DATE_1>.", mapping) == "You were born on 3 May."
