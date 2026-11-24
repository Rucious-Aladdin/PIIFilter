"""Tests for the token tags of experiments/train_bert.py."""

from PIIFilter.masking import Span

LABEL2ID = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-PHONE_NUMBER": 3, "I-PHONE_NUMBER": 4}


def test_bio_labels(load_script):
    train_bert = load_script("train_bert")
    # "[CLS] i ' m min ##ji park , 010 [SEP]" for "I'm Minji Park, 010"
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (4, 7), (7, 9), (10, 14), (14, 15), (16, 19), (0, 0)]
    spans = [Span(4, 14, "NAME"), Span(16, 19, "PHONE")]
    assert train_bert.bio_labels(offsets, spans, LABEL2ID) == [-100, 0, 0, 0, 1, 2, 2, 0, 3, -100]


def test_two_spans_in_a_row_both_start_with_b(load_script):
    train_bert = load_script("train_bert")
    offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]
    spans = [Span(0, 3, "NAME"), Span(4, 7, "NAME")]
    assert train_bert.bio_labels(offsets, spans, LABEL2ID) == [-100, 1, 1, -100]
