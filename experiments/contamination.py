"""Did BERT see our Gretel test documents in training? (S3 of the Q1 report)

    uv run python experiments/contamination.py

Gravitee's BERT was trained on gravitee-io/pii-detection-dataset, which has 25,948 rows from
Gretel PII finance (the dataset card does not say which split). This script downloads that
corpus at a pinned revision into data/raw/gravitee/ (git-ignored) and measures how much of
each of our 2,962 Gretel test documents it contains:

- identical texts (after joining runs of whitespace);
- the share of the document's word 13-grams (lowercased) that also occur in its Gretel rows.

Then it scores BERT on Gretel again without the documents that look seen in training.
Writes results/analysis/contamination.md.
"""

import argparse
import re

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from benchmark.analysis import example_counts, load_comparison, scores
from benchmark.data import ANALYSIS_DIR, RAW_DIR

REPO, REVISION = "gravitee-io/pii-detection-dataset", "c90469d2e6033ae0d94eccb17f9da15ad5d67b0f"
N = 13
WORD = re.compile(r"\w+")


def grams(text: str) -> set[int]:
    words = WORD.findall(text.lower())
    return {hash(" ".join(words[i : i + N])) for i in range(len(words) - N + 1)}


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    folder = RAW_DIR / "gravitee"
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=["data.parquet", "README.md"],
                      local_dir=folder)
    corpus = pd.read_parquet(folder / "data.parquet", columns=["text", "source"])
    gretel_rows = corpus[corpus["source"].str.contains("synthetic_pii_finance")]["text"].tolist()
    print(f"{len(corpus)} training rows, {len(gretel_rows)} from Gretel PII finance")

    norm = lambda t: " ".join(t.split())  # noqa: E731
    seen_texts = {norm(t) for t in gretel_rows}
    seen_grams = set().union(*(grams(t) for t in gretel_rows))

    examples, labels, predictions = load_comparison("gretel", "A")
    identical = np.array([norm(ex.text) in seen_texts for ex in examples])
    shares = []
    for ex in examples:
        g = grams(ex.text)
        shares.append(len(g & seen_grams) / len(g) if g else 0.0)
    shares = np.array(shares)

    lines = ["# Did BERT see the Gretel test documents?", "",
             f"Training corpus: {REPO} at {REVISION[:12]}, {len(corpus)} rows, {len(gretel_rows)} from Gretel PII finance.",
             f"Gretel test documents: {len(examples)}.", "",
             "| Measure | Documents |", "|---|---|",
             f"| identical text in the training rows | {int(identical.sum())} |"]
    for cut in (0.5, 0.8, 1.0):
        lines.append(f"| at least {cut:.0%} of their 13-grams in the training rows | {int((shares >= cut).sum())} |")
    lines.append(f"| median share of 13-grams in the training rows | {np.median(shares):.3f} |")

    # BERT and GLiNER on Gretel, on all documents and on the documents not seen (comparison A)
    seen = identical | (shares >= 0.5)
    lines += ["", "Micro F2 on Gretel (comparison A), all documents and documents not seen in training "
              "(no identical text, under 50% of 13-grams):", "", "| Detector | All | Not seen | Documents not seen |",
              "|---|---|---|---|"]
    for detector in ("bert", "gliner"):
        counts = example_counts(examples, predictions[detector], labels)
        all_f2 = float(scores(counts.sum(axis=(0, 1)))["f2"])
        unseen_f2 = float(scores(counts[~seen].sum(axis=(0, 1)))["f2"])
        lines.append(f"| {detector} | {all_f2:.3f} | {unseen_f2:.3f} | {int((~seen).sum())} |")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "contamination.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
