"""Tests for the safety checks of the experiment scripts (no model is loaded)."""

import json
import sys

import pytest

from benchmark.data import Example, write_jsonl
from PIIFilter.masking import Span


class FakeDetector:
    model_id = "fake-model"
    revision = "abc"

    def __init__(self, threshold=0.35):
        self.threshold = threshold

    def detect(self, text):
        return [Span(0, 2, "NAME", 0.9)]


@pytest.fixture
def benchmark(load_script, monkeypatch, tmp_path):
    """run_benchmark.py on a two-example test set in a temporary folder."""
    module = load_script("run_benchmark")
    write_jsonl(tmp_path / "gretel_clean.jsonl", [Example("a", "Jo one", []), Example("b", "Al two", [])])
    monkeypatch.setattr(module, "TEST_DIR", tmp_path)
    monkeypatch.setattr(module, "computer_info", lambda: {})

    def run(threshold=0.35):
        monkeypatch.setattr(module, "make_detector", lambda *_: FakeDetector(threshold))
        monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "gliner", "--datasets", "gretel", "--out", str(tmp_path / "out")])
        module.main()

    return run, tmp_path / "out" / "gliner" / "gretel" / "clean.jsonl"


def test_complete_run_rewrites_nothing(benchmark):
    run, out = benchmark
    run()
    meta = out.with_suffix(".meta.json")
    before = (out.read_bytes(), meta.read_bytes(), out.stat().st_mtime_ns, meta.stat().st_mtime_ns)
    run()
    assert (out.read_bytes(), meta.read_bytes(), out.stat().st_mtime_ns, meta.stat().st_mtime_ns) == before


def test_resume_with_other_settings_stops(benchmark):
    run, out = benchmark
    run()
    out.write_text(out.read_text().splitlines(keepends=True)[0])  # as if interrupted after one example
    kept = out.read_bytes()
    with pytest.raises(SystemExit, match="threshold: 0.35 -> 0.5"):
        run(threshold=0.5)
    assert out.read_bytes() == kept


def test_resume_with_same_settings_finishes(benchmark):
    run, out = benchmark
    run()
    out.write_text(out.read_text().splitlines(keepends=True)[0])
    run()
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["a", "b"]


def test_predictions_without_meta_are_refused(benchmark):
    run, out = benchmark
    out.parent.mkdir(parents=True)
    out.write_text('{"id": "a", "spans": []}\n')
    with pytest.raises(SystemExit, match="no clean.meta.json"):
        run()


def test_config_is_saved_before_the_first_prediction(benchmark, monkeypatch):
    run, out = benchmark

    def crash(self, text):
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeDetector, "detect", crash)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert json.loads(out.with_suffix(".meta.json").read_text())["threshold"] == 0.35


def test_sensitivity_stops_without_writing_when_the_low_cutoff_run_is_missing(load_script, monkeypatch, tmp_path):
    module = load_script("sensitivity")
    for dataset in module.SOURCES:
        write_jsonl(tmp_path / "test" / f"{dataset}_clean.jsonl", [Example("a", "text", [])])
    monkeypatch.setattr(module, "TEST_DIR", tmp_path / "test")
    monkeypatch.setattr(module, "LOW_CUTOFF_RUN", tmp_path / "missing")
    monkeypatch.setattr(module, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(sys, "argv", ["sensitivity.py"])
    with pytest.raises(SystemExit, match="--threshold 0.05"):
        module.main()
    assert not (tmp_path / "analysis").exists()


def test_leftover_ignores_restored_values_and_ordinary_dates(load_script):
    toy = load_script("toy_experiment")
    mapping = {"<PHONE_1>": "010-2345-6789"}
    assert toy.leftover_placeholders("Call my phone 010-2345-6789 on date 15 March.", mapping) == []
    assert toy.leftover_placeholders("Born on DATE_1, call PHONE_1 or <DATE_2>.", mapping) == ["DATE_1", "PHONE_1", "<DATE_2>"]
