"""Download the benchmark datasets into data/raw/<name>/ (sources in benchmark/data.py).

    uv run python experiments/download_data.py              # all three datasets
    uv run python experiments/download_data.py nemotron     # only some of them

Files already downloaded are not downloaded again.
"""

import argparse

from huggingface_hub import snapshot_download

from benchmark.data import RAW_DIR, SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"datasets to download: {', '.join(SOURCES)} (default: all)")
    args = parser.parse_args()
    unknown = set(args.names) - set(SOURCES)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    for name in args.names or SOURCES:
        source = SOURCES[name]
        print(f"{name}: {source['repo']} @ {source['revision'][:7]}")
        snapshot_download(
            source["repo"],
            repo_type="dataset",
            revision=source["revision"],
            allow_patterns=source["files"],
            local_dir=RAW_DIR / name,
        )
        print(f"  -> {RAW_DIR / name}")


if __name__ == "__main__":
    main()
