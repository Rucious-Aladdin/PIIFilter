"""The demo scenarios, and the whole demo chain with a fake detector and a fake LLM (no network)."""

import json
import sys
from pathlib import Path

import pytest

from PIIFilter.masking import LABELS, PLACEHOLDER_RE, Span

SAMPLES = Path(__file__).resolve().parent.parent / "experiments" / "demo_samples"
DEMO = [json.loads(path.read_text()) for path in sorted(SAMPLES.glob("*.json"))]


def test_twenty_scenarios_each_clean_and_noisy():
    ids = [sample["id"] for sample in DEMO]
    assert len(ids) == len(set(ids)) == 40
    assert {i.removesuffix("_clean") for i in ids if i.endswith("_clean")} == {i.removesuffix("_noisy") for i in ids if i.endswith("_noisy")}


@pytest.mark.parametrize("sample", DEMO, ids=lambda s: s["id"])
def test_each_gold_value_is_once_in_its_prompt(sample):
    assert sample["gold"]
    for gold in sample["gold"]:
        assert gold["label"] in LABELS
        assert sample["prompt"].count(gold["value"]) == 1
        assert sample["prompt"].lower().count(gold["value"].lower()) == 1  # the privacy check ignores case


def test_demo_covers_the_six_labels_with_fictional_emails():
    golds = [gold for sample in DEMO for gold in sample["gold"]]
    assert {gold["label"] for gold in golds} == set(LABELS) - {"DATE"}
    assert all(gold["value"].endswith("@example.com") for gold in golds if gold["label"] == "EMAIL")


class GoldDetector:
    """Finds exactly the gold values of the demo prompts."""

    model_id = "fake-detector"

    def __init__(self, threshold=None, device=None):
        self.gold = {sample["prompt"]: sample["gold"] for sample in DEMO}

    def detect(self, text):
        spans = []
        for gold in self.gold[text]:
            start = text.index(gold["value"])
            spans.append(Span(start, start + len(gold["value"]), gold["label"], 1.0))
        return spans


class FakeLLM:
    """Answers with every placeholder it received."""

    model_id = "fake-llm"

    def chat(self, prompt):
        return "Done: " + ", ".join(PLACEHOLDER_RE.findall(prompt)) + "."


def test_whole_chain_with_fake_llm(load_script, monkeypatch, tmp_path):
    toy, summary = load_script("toy_experiment"), load_script("summarize_demo")
    monkeypatch.setattr(toy, "GlinerDetector", GoldDetector)
    monkeypatch.setattr(toy, "make_llm", lambda backend, model: FakeLLM())
    monkeypatch.setattr(sys, "argv", ["toy_experiment.py", "--samples", str(SAMPLES), "--out", str(tmp_path), "--threshold", "0.35"])
    toy.main()

    data = json.loads((tmp_path / "openai" / "toy_results.json").read_text())
    assert len(data["examples"]) == 40
    assert data["meta"]["detector"]["threshold"] == 0.35
    for example in data["examples"]:
        assert not any(gold["value"] in example["masked_prompt"] for gold in next(s for s in DEMO if s["id"] == example["id"])["gold"])
        assert all(gold["value"] in example["answer"] for gold in next(s for s in DEMO if s["id"] == example["id"])["gold"])
    text = summary.summarize(data["examples"])
    for check in ("Private?", "Enough context?", "Usable answer?"):
        assert f"| {check} | 20 / 20 | 20 / 20 |" in text
    assert "None." in text


def test_summary_counts_failures_by_version(load_script):
    summary = load_script("summarize_demo")
    good = {"private": True, "enough_context": True, "usable_answer": True}
    examples = [
        {"id": "a_clean", "masked_prompt": "x", "answer": "y", "checks": good},
        {"id": "a_noisy", "masked_prompt": "x", "answer": "y", "checks": {**good, "private": False}},
    ]
    text = summary.summarize(examples)
    assert "| Private? | 1 / 1 | 0 / 1 |" in text
    assert "`a_noisy`: Private?" in text
