"""Make the noisy test sets (question 2): data/test/<dataset>_noisy_<kind>[_dates].jsonl.

Each noisy test set is a copy of a clean test set, with the same examples and ids, where
every text gets the mistakes of one kind of noise (benchmark/noise.py): typo, space, case,
punct, number, or mixed. The gold spans move with the text and keep their labels. The
_dates version gets exactly the same text, with the spans of the clean _dates version
(comparison C), so both versions of a noisy test set match like the clean ones do.

    uv run python experiments/make_noisy.py                        # every dataset and kind
    uv run python experiments/make_noisy.py gretel --kinds typo    # one dataset, one kind

The amount of noise (--rate, edits per word) and the seed are fixed before any scoring.
The seed of the test sets (1) differs from the one used for the training copies (question 3).
Each example's meta gets a "noise" entry listing its edits (kind, position in the clean
text, old and new string, inside a gold span or not), and next to each test set a
.meta.json records the clean test set it comes from, the noise settings, counts and a
SHA-256 checksum, so two computers can check they made the same files.
"""

import argparse
import json
from collections import Counter

from benchmark.data import SOURCES, TEST_DIR, Example, file_sha256, read_jsonl, write_jsonl
from benchmark.noise import NOISE_KINDS, RATE, example_rng, noisy_copy

TEST_SEED = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"datasets: {', '.join(SOURCES)} (default: all)")
    parser.add_argument("--kinds", nargs="+", choices=NOISE_KINDS + ("mixed",), default=NOISE_KINDS + ("mixed",))
    parser.add_argument("--rate", type=float, default=RATE, help=f"edits per word (default: {RATE})")
    parser.add_argument("--seed", type=int, default=TEST_SEED, help=f"noise seed (default: {TEST_SEED})")
    args = parser.parse_args()
    unknown = set(args.names) - set(SOURCES)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    for name in args.names or SOURCES:
        sources = {v: TEST_DIR / f"{name}_{v}.jsonl" for v in ("clean", "clean_dates")}
        for path in sources.values():
            if not path.exists():
                raise SystemExit(f"{path} is missing; run experiments/prepare_data.py first")
        clean, dates = read_jsonl(sources["clean"]), read_jsonl(sources["clean_dates"])
        if [(e.id, e.text) for e in clean] != [(e.id, e.text) for e in dates]:
            raise SystemExit(f"{name}: the clean and dates test sets do not have the same texts")
        source_meta = {v: json.loads(p.with_suffix(".meta.json").read_text()) for v, p in sources.items()}

        for kind in args.kinds:
            noisy = {"clean": [], "clean_dates": []}
            edits_per_kind, inside, changed = Counter(), 0, 0
            for c, d in zip(clean, dates):
                text, (c_spans, d_spans), edits = noisy_copy(
                    c.text, [c.spans, d.spans], kind, example_rng(args.seed, kind, c.id), args.rate
                )
                noise = {"kind": kind, "edits": edits}
                noisy["clean"].append(Example(c.id, text, c_spans, {**c.meta, "noise": noise}))
                noisy["clean_dates"].append(Example(d.id, text, d_spans, {**d.meta, "noise": noise}))
                edits_per_kind.update(e["kind"] for e in edits)
                inside += sum(e["inside"] for e in edits)
                changed += bool(edits)

            for source, examples in noisy.items():
                version = f"noisy_{kind}" + ("_dates" if source == "clean_dates" else "")
                path = TEST_DIR / f"{name}_{version}.jsonl"
                write_jsonl(path, examples)
                labels = source_meta[source]["labels"]
                counts = Counter(s.label for ex in examples for s in ex.spans)
                meta = {
                    **{k: source_meta[source][k] for k in ("dataset", "repo", "revision", "split", "english_records", "seed")},
                    "version": version,
                    "source": f"{name}_{source}.jsonl",
                    "source_sha256": source_meta[source]["sha256"],
                    "noise": {"kind": kind, "rate": args.rate, "seed": args.seed},
                    "n": len(examples),
                    "labels": labels,
                    "spans_per_label": {label: counts[label] for label in labels},
                    "spans_per_label_clean": source_meta[source]["spans_per_label"],
                    "examples_without_spans": sum(not ex.spans for ex in examples),
                    "examples_changed": changed,
                    "edits": sum(edits_per_kind.values()),
                    "edits_per_kind": dict(sorted(edits_per_kind.items())),
                    "edits_inside_spans": inside,  # inside a span of the clean version
                    "sha256": file_sha256(path),
                }
                path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            total = sum(edits_per_kind.values())
            print(f"{name} noisy_{kind}: {changed}/{len(clean)} examples changed, {total} edits "
                  f"({inside} inside a value) -> {name}_noisy_{kind}[_dates].jsonl")


if __name__ == "__main__":
    main()
