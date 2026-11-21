"""Tests for the span clean-up rules in benchmark/spans.py."""

from benchmark.spans import normalize_spans
from PIIFilter.masking import Span


def words(text: str, spans: list[Span]) -> list[tuple[str, str]]:
    return [(text[s.start : s.end], s.label) for s in spans]


def test_first_name_and_surname_are_joined():
    text = "I am Minji Park."
    spans = [Span(5, 10, "NAME"), Span(11, 15, "NAME")]
    assert words(text, normalize_spans(text, spans)) == [("Minji Park", "NAME")]


def test_comma_or_line_break_keeps_spans_apart():
    text = "Doe, Jane\nJohn"
    spans = [Span(0, 3, "NAME"), Span(5, 9, "NAME"), Span(10, 14, "NAME")]
    assert words(text, normalize_spans(text, spans)) == [("Doe", "NAME"), ("Jane", "NAME"), ("John", "NAME")]


def test_different_labels_are_not_joined():
    text = "Park 010-2345-6789"
    spans = [Span(0, 4, "NAME"), Span(5, 18, "PHONE")]
    assert words(text, normalize_spans(text, spans)) == [("Park", "NAME"), ("010-2345-6789", "PHONE")]


def test_title_is_removed_from_names_only():
    text = "Dr. Kim lives at Dr. Lee Street"
    spans = [Span(0, 7, "NAME"), Span(17, 31, "ADDRESS")]
    assert words(text, normalize_spans(text, spans)) == [("Kim", "NAME"), ("Dr. Lee Street", "ADDRESS")]


def test_ai4privacy_titles_are_removed_but_not_inside_a_word():
    text = "Mstr Zelenay and Masterson"
    spans = [Span(0, 12, "NAME"), Span(17, 26, "NAME")]
    assert words(text, normalize_spans(text, spans)) == [("Zelenay", "NAME"), ("Masterson", "NAME")]


def test_span_that_is_only_a_title_is_dropped():
    text = "Dr. A. Smith"
    spans = [Span(0, 3, "NAME"), Span(4, 12, "NAME")]
    assert words(text, normalize_spans(text, spans)) == [("A. Smith", "NAME")]


def test_whitespace_is_trimmed_and_bad_spans_dropped():
    text = "call 074.0010446 now"
    spans = [Span(5, 17, "PHONE"), Span(15, 99, "PHONE")]  # trailing space; second is out of range
    assert words(text, normalize_spans(text, spans)) == [("074.0010446", "PHONE")]


def test_enclosing_brackets_and_quotes_are_removed():
    text = 'mail [erik@email.com] or "Erik Olsson"'
    spans = [Span(5, 21, "EMAIL"), Span(25, 38, "NAME")]
    assert words(text, normalize_spans(text, spans)) == [("erik@email.com", "EMAIL"), ("Erik Olsson", "NAME")]


def test_bracket_that_does_not_enclose_the_span_is_kept():
    text = "call (555) 123-4567"
    assert words(text, normalize_spans(text, [Span(5, 19, "PHONE")])) == [("(555) 123-4567", "PHONE")]


def test_overlap_keeps_the_span_that_starts_first():
    text = "Mr. Booth"
    spans = [Span(4, 9, "NAME"), Span(0, 9, "NAME")]  # after the title is removed, both are "Booth"
    assert words(text, normalize_spans(text, spans)) == [("Booth", "NAME")]


def test_joined_span_keeps_the_lower_score():
    text = "minji park"
    spans = [Span(0, 5, "NAME", 0.9), Span(6, 10, "NAME", 0.6)]
    assert normalize_spans(text, spans) == [Span(0, 10, "NAME", 0.6)]
