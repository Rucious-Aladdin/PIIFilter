"""Tests for reading datasets and the JSONL format in benchmark/data.py."""

import json

from benchmark.data import Example, label_map, predicted_ids, read_ai4privacy, read_jsonl, scored_labels, write_jsonl
from PIIFilter.masking import Span


def test_dates_version_scores_every_date_but_not_times():
    labels = label_map("nemotron", dates=True)
    assert labels["date_of_birth"] == labels["date"] == labels["date_time"] == "DATE"
    assert "time" not in labels and "DATE_OF_BIRTH" not in labels.values()
    assert scored_labels("nemotron") == ["NAME", "EMAIL", "PHONE", "ADDRESS", "DATE_OF_BIRTH", "CARD_NUMBER"]
    # ai4privacy has no birth dates, but it has dates
    assert scored_labels("ai4privacy", dates=True) == ["NAME", "EMAIL", "PHONE", "ADDRESS", "DATE", "CARD_NUMBER"]


def test_ai4privacy_reader_keeps_english_records_only(tmp_path):
    records = [
        {"uid": 1, "language": "en", "region": "US", "source_text": "I am Minji Park",
         "privacy_mask": [{"start": 5, "end": 10, "label": "GIVENNAME"}, {"start": 11, "end": 15, "label": "SURNAME"}]},
        {"uid": 2, "language": "fr", "region": "FR", "source_text": "Je suis Minji", "privacy_mask": []},
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert list(read_ai4privacy(path)) == [
        ("ai4privacy-1", "I am Minji Park", [(5, 10, "GIVENNAME"), (11, 15, "SURNAME")], {"region": "US"})
    ]


def test_jsonl_round_trip(tmp_path):
    examples = [Example("x-1", "I am Minji Park", [Span(5, 15, "NAME")], {"locale": "us"})]
    path = tmp_path / "test.jsonl"
    write_jsonl(path, examples)
    assert read_jsonl(path) == examples


def test_predicted_ids_drops_a_line_cut_by_an_interruption(tmp_path):
    path = tmp_path / "clean.jsonl"
    path.write_text('{"id": "a", "spans": []}\n{"id": "b", "spans": []}\n{"id": "c", "sp')
    assert predicted_ids(path) == {"a", "b"}
    assert path.read_text() == '{"id": "a", "spans": []}\n{"id": "b", "spans": []}\n'
    assert predicted_ids(tmp_path / "missing.jsonl") == set()
