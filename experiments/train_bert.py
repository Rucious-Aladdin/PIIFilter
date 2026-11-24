"""Fine-tune the BERT baseline for question 3: models/bert-<variant>-s<seed>/.

Both versions start from the benchmark's BERT (gravitee-io/bert-small-pii-detection, same
revision) and keep its 51 BIO tags, so baselines/bert_detector.py reads them like the
original model. They learn from the ai4privacy train split (experiments/prepare_train.py):

    clean   train_clean
    noisy   train_clean + train_noisy (one mixed-noise copy of each record)

Everything else is the same for both: the settings below and the number of steps (three
passes over train_clean), so the only difference is the noisy copies (proposal, Appendix B).
The model is saved after the last step; the dev sets are only logged, never used to choose
a checkpoint, since the noisy dev set would favour the noisy version.

Our labels become the model's classes (DATE -> DATE_TIME, as any date); every other token
is O, so the model also learns that cities, titles and the like are not our labels. Each
token is tagged by the span its first character falls in: B- for the first token of a span,
I- for the next ones, -100 for [CLS] and [SEP].

    CUDA_VISIBLE_DEVICES=0 uv run python experiments/train_bert.py clean --seed 0
    CUDA_VISIBLE_DEVICES=1 uv run python experiments/train_bert.py noisy --seed 0
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from baselines.bert_detector import BertDetector
from benchmark.data import ROOT, file_sha256, read_jsonl
from PIIFilter.masking import Span

TRAIN_DIR = ROOT / "data" / "train"
MODELS_DIR = ROOT / "models"

# Our label -> the model's class
LABEL_TO_CLASS = {
    "NAME": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "ADDRESS": "LOCATION",
    "DATE": "DATE_TIME",
    "CARD_NUMBER": "CREDIT_CARD",
}

SETTINGS = {
    "learning_rate": 5e-5,
    "batch_size": 32,
    "epochs_of_clean_data": 3,
    "warmup": 0.06,  # share of the steps
    "weight_decay": 0.01,
    "max_length": 256,  # the window BertDetector reads
}


def bio_labels(offsets: list[tuple[int, int]], spans: list[Span], label2id: dict[str, int]) -> list[int]:
    """One tag id per token, from the character offsets of the tokens."""
    labels, previous = [], None
    for start, end in offsets:
        if start == end:  # [CLS], [SEP]
            labels.append(-100)
            previous = None
            continue
        span = next((s for s in spans if s.start <= start < s.end), None)
        if span is None:
            labels.append(label2id["O"])
        else:
            tag = "I" if span is previous else "B"
            labels.append(label2id[f"{tag}-{LABEL_TO_CLASS[span.label]}"])
        previous = span
    return labels


def encode(path: Path, tokenizer, label2id: dict[str, int], max_length: int) -> list[dict]:
    examples = read_jsonl(path)
    enc = tokenizer(
        [ex.text for ex in examples], truncation=True, max_length=max_length, return_offsets_mapping=True
    )
    return [
        {
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": bio_labels(enc["offset_mapping"][i], ex.spans, label2id),
        }
        for i, ex in enumerate(examples)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=["clean", "noisy"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="model folder (default: models/bert-<variant>-s<seed>)")
    parser.add_argument("--max-steps", type=int, help="stop earlier, for a quick test (default: the full run)")
    args = parser.parse_args()
    out = args.out or MODELS_DIR / f"bert-{args.variant}-s{args.seed}"

    import torch
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer,
        TrainingArguments,
    )

    base, revision = BertDetector.default_model, BertDetector.default_revision
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    model = AutoModelForTokenClassification.from_pretrained(base, revision=revision)
    label2id = model.config.label2id

    files = {"train_clean": TRAIN_DIR / "ai4privacy_train_clean.jsonl"}
    if args.variant == "noisy":
        files["train_noisy"] = TRAIN_DIR / "ai4privacy_train_noisy.jsonl"
    files |= {"dev_clean": TRAIN_DIR / "ai4privacy_dev_clean.jsonl", "dev_noisy": TRAIN_DIR / "ai4privacy_dev_noisy.jsonl"}
    for path in files.values():
        if not path.exists():
            raise SystemExit(f"{path} is missing; run experiments/prepare_train.py first")
    data = {name: encode(path, tokenizer, label2id, SETTINGS["max_length"]) for name, path in files.items()}
    train = data["train_clean"] + data.get("train_noisy", [])

    # The same number of steps for both versions: three passes over the clean training set.
    steps = math.ceil(SETTINGS["epochs_of_clean_data"] * len(data["train_clean"]) / SETTINGS["batch_size"])
    if args.max_steps:
        steps = min(steps, args.max_steps)

    training_args = TrainingArguments(
        output_dir=str(out / "trainer"),
        max_steps=steps,
        per_device_train_batch_size=SETTINGS["batch_size"],
        per_device_eval_batch_size=128,
        learning_rate=SETTINGS["learning_rate"],
        warmup_steps=round(SETTINGS["warmup"] * steps),
        weight_decay=SETTINGS["weight_decay"],
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=max(1, steps // 6),
        logging_steps=max(1, steps // 30),
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset={"clean": data["dev_clean"], "noisy": data["dev_noisy"]},
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )
    start = time.perf_counter()
    trainer.train()
    final_eval = trainer.evaluate()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    meta = {
        "variant": args.variant, "seed": args.seed,
        "base_model": base, "base_revision": revision,
        "settings": SETTINGS, "steps": steps, "train_examples": len(train),
        "data": {name: {"file": path.name, "sha256": file_sha256(path)} for name, path in files.items()},
        "final_dev_loss": {k: round(v, 5) for k, v in final_eval.items() if k.endswith("_loss")},
        "log": trainer.state.log_history,
        "seconds": round(time.perf_counter() - start, 1),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "packages": {p: version(p) for p in ("transformers", "torch")},
    }
    (out / "training.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {out} ({steps} steps, {meta['seconds']} s); dev loss {meta['final_dev_loss']}")


if __name__ == "__main__":
    main()
