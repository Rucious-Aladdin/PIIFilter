"""The three benchmark datasets: download sources, readers, and the common format.

Each source is a Hugging Face dataset repository pinned to one revision (commit), so every
computer downloads the same files (experiments/download_data.py). load_examples() reads
the English records of one split and turns them into Examples: the text and its gold
spans, with the dataset's labels mapped to our six (proposal, Table 5) and cleaned with
benchmark/spans.py. Labels we do not score are dropped.

Comparison C (dates=True): Presidio and BERT only have a label for any date, so we also
score every date under one label, DATE, in place of DATE_OF_BIRTH. Times alone are not
counted.

On disk, an Example is one line of JSON:
    {"id": ..., "text": ..., "spans": [{"start": ..., "end": ..., "label": ...}], "meta": {...}}
"""

import ast
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from PIIFilter.masking import Span

from .spans import normalize_spans

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TEST_DIR = ROOT / "data" / "test"
PRED_DIR = ROOT / "results" / "predictions"
ANALYSIS_DIR = ROOT / "results" / "analysis"
THRESHOLD_FILE = ROOT / "results" / "gliner_threshold.json"  # GLiNER's cut-off (tune_gliner.py)

# Our six labels, plus DATE for comparison C (every date, in place of DATE_OF_BIRTH).
ALL_LABELS = ("NAME", "EMAIL", "PHONE", "ADDRESS", "DATE_OF_BIRTH", "DATE", "CARD_NUMBER")


@dataclass
class Example:
    """One text with its gold PII spans (our labels)."""

    id: str
    text: str
    spans: list[Span]
    meta: dict = field(default_factory=dict)  # dataset details kept for analysis (e.g. locale)


# A reader yields (id, text, [(start, end, dataset label)], meta) for each English record.
RawRecord = tuple[str, str, list[tuple[int, int, str]], dict]


def read_ai4privacy(path: Path) -> Iterator[RawRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["language"] == "en":
                spans = [(s["start"], s["end"], s["label"]) for s in r["privacy_mask"]]
                yield f"ai4privacy-{r['uid']}", r["source_text"], spans, {"region": r["region"]}


def read_nemotron(path: Path) -> Iterator[RawRecord]:
    df = pd.read_parquet(path, columns=["uid", "text", "spans", "locale", "document_format", "document_type"])
    for uid, text, spans, locale, doc_format, doc_type in df.itertuples(index=False):
        # The spans column holds the text of a Python list, e.g. "[{'start': 0, ...}]".
        spans = [(s["start"], s["end"], s["label"]) for s in ast.literal_eval(spans)]
        meta = {"locale": locale, "document_format": doc_format, "document_type": doc_type}
        # Each uid has two documents, one per locale (us, intl), so the id needs both.
        yield f"nemotron-{locale}-{uid}", text, spans, meta


def read_gretel(path: Path) -> Iterator[RawRecord]:
    df = pd.read_parquet(path, columns=["index", "generated_text", "pii_spans", "language", "document_type"])
    df = df[df["language"] == "English"]
    for index, text, spans, _, doc_type in df.itertuples(index=False):
        # The pii_spans column holds a JSON list.
        spans = [(s["start"], s["end"], s["label"]) for s in json.loads(spans)]
        yield f"gretel-{index}", text, spans, {"document_type": doc_type}


SOURCES = {
    # Short generated sentences, many languages. The validation split is our test set
    # (there is no test split); the train split is used for fine-tuning (question 3).
    # Made with Llama models, so the Llama license files come with the data.
    # No birth-date label: DATE_OF_BIRTH is not scored on this dataset.
    "ai4privacy": {
        "repo": "ai4privacy/open-pii-masking-500k-ai4privacy",
        "revision": "506996d625ed970a0063432daf6007cf4a3a48e3",
        "files": ["data/train/*", "data/validation/*", "README.md", "license", "llama-*-license.txt"],
        "splits": {"test": "data/validation/test.jsonl", "train": "data/train/train.jsonl"},
        "test_size": 10_000,  # examples drawn for our test set (chosen after timing the detectors)
        "read": read_ai4privacy,
        "labels": {
            "GIVENNAME": "NAME", "SURNAME": "NAME",
            "EMAIL": "EMAIL",
            "TELEPHONENUM": "PHONE",
            "BUILDINGNUM": "ADDRESS", "STREET": "ADDRESS",
            "CREDITCARDNUMBER": "CARD_NUMBER",
        },
        # Comparison C. DATE includes dates with a time ("1975-08-08T00:00:00"); TIME is left out.
        "date_labels": ["DATE"],
    },
    # Generated English documents (forms, invoices, emails, notes). Test split only.
    "nemotron": {
        "repo": "nvidia/Nemotron-PII",
        "revision": "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed",
        "files": ["data/test-*", "README.md"],
        "splits": {"test": "data/test-00000-of-00001.parquet"},
        "test_size": 2_000,
        "read": read_nemotron,
        "labels": {
            "first_name": "NAME", "last_name": "NAME",
            "email": "EMAIL",
            "phone_number": "PHONE",
            "street_address": "ADDRESS",
            "date_of_birth": "DATE_OF_BIRTH",
            "credit_debit_card": "CARD_NUMBER",
        },
        # Comparison C: date_time is a date with a time ("2025-07-13T17:43:03"); time is left out.
        "date_labels": ["date", "date_of_birth", "date_time"],
    },
    # Generated financial documents in 7 languages. Test split only, English records.
    "gretel": {
        "repo": "gretelai/synthetic_pii_finance_multilingual",
        "revision": "7b844d16738527a04264f50214cb426a4cea0897",
        "files": ["data/test-*", "README.md"],
        "splits": {"test": "data/test-00000-of-00001.parquet"},
        "test_size": 2_962,  # all English test documents
        "read": read_gretel,
        "labels": {
            "name": "NAME", "first_name": "NAME", "last_name": "NAME",
            "email": "EMAIL",
            "phone_number": "PHONE",
            "street_address": "ADDRESS",
            "date_of_birth": "DATE_OF_BIRTH",
            "credit_card_number": "CARD_NUMBER",
        },
        # Comparison C, as for Nemotron-PII: time is left out.
        "date_labels": ["date", "date_of_birth", "date_time"],
    },
}


def label_map(name: str, dates: bool = False) -> dict[str, str]:
    """Dataset label -> our label. With dates=True (comparison C), every date becomes DATE
    and nothing is DATE_OF_BIRTH."""
    labels = SOURCES[name]["labels"]
    if not dates:
        return labels
    labels = {raw: ours for raw, ours in labels.items() if ours != "DATE_OF_BIRTH"}
    return labels | {raw: "DATE" for raw in SOURCES[name]["date_labels"]}


def scored_labels(name: str, dates: bool = False) -> list[str]:
    """Our labels that the dataset has; the others are not scored on it (no DOB in ai4privacy)."""
    return [label for label in ALL_LABELS if label in label_map(name, dates).values()]


def load_examples(name: str, split: str = "test", dates: bool = False) -> list[Example]:
    """All English records of one split, with gold spans in our labels (dates: comparison C)."""
    source = SOURCES[name]
    path = RAW_DIR / name / source["splits"][split]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run: uv run python experiments/download_data.py {name}")
    labels = label_map(name, dates)
    examples = []
    for id_, text, raw_spans, meta in source["read"](path):
        spans = [Span(start, end, labels[label]) for start, end, label in raw_spans if label in labels]
        examples.append(Example(id_, text, normalize_spans(text, spans), meta))
    return examples


def spans_to_json(spans: list[Span]) -> list[dict]:
    """Spans as JSON objects; "score" is written only when the span has one (detector outputs)."""
    items = []
    for s in spans:
        item = {"start": s.start, "end": s.end, "label": s.label}
        if s.score is not None:
            item["score"] = s.score
        items.append(item)
    return items


def spans_from_json(items: list[dict]) -> list[Span]:
    return [Span(i["start"], i["end"], i["label"], i.get("score")) for i in items]


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            record = {"id": ex.id, "text": ex.text, "spans": spans_to_json(ex.spans), "meta": ex.meta}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return [Example(r["id"], r["text"], spans_from_json(r["spans"]), r["meta"]) for r in records]


# Detector outputs: results/predictions/<detector>/<dataset>/<version>.jsonl, one line per
# example of data/test/<dataset>_<version>.jsonl: {"id": ..., "spans": [{..., "score": ...}]}.
# The text is not repeated; the id links each line to its example.


def read_predictions(path: Path) -> dict[str, list[Span]]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return {r["id"]: spans_from_json(r["spans"]) for r in records}


def predicted_ids(path: Path) -> set[str]:
    """Ids already in a predictions file, so an interrupted run can resume.

    Lines are written one at a time, so an interruption can only cut the last line: it is
    removed from the file.
    """
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
        path.write_text("".join(lines), encoding="utf-8")
    return {json.loads(line)["id"] for line in lines}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
