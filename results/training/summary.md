# Fine-tuning for question 3: data and runs

## Data

From `data/train/*.meta.json` (experiments/prepare_train.py): English records of the ai4privacy train split, a dev sample held out (seed 0), and one mixed-noise copy of each record (seed 2). Every date is under DATE.

| File | Records | Noise | NAME | EMAIL | PHONE | ADDRESS | DATE | CARD_NUMBER |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ai4privacy_train_clean | 115,533 | none | 101,783 | 13,282 | 17,984 | 17,487 | 16,391 | 2,148 |
| ai4privacy_train_noisy | 115,533 | mixed, rate 0.1, seed 2 | 101,783 | 13,282 | 17,984 | 17,487 | 16,391 | 2,148 |
| ai4privacy_dev_clean | 5,000 | none | 4,443 | 592 | 760 | 765 | 692 | 104 |
| ai4privacy_dev_noisy | 5,000 | mixed, rate 0.1, seed 2 | 4,443 | 592 | 760 | 765 | 692 | 104 |

English records in the train split: 120,533.

## Runs

Base model: gravitee-io/bert-small-pii-detection (revision f8c27a85c5). Settings: learning_rate 5e-05, batch_size 32, epochs_of_clean_data 3, warmup 0.06, weight_decay 0.01, max_length 256.

| Model | Training records | Steps | Minutes | GPU | Dev loss, clean | Dev loss, noisy |
|---|---:|---:|---:|---|---:|---:|
| bert-clean-s0 | 115,533 | 10,832 | 1.9 | NVIDIA GeForce RTX 4090 | 0.0124 | 0.1314 |
| bert-clean-s1 | 115,533 | 10,832 | 2.0 | NVIDIA L40S | 0.0119 | 0.1330 |
| bert-clean-s2 | 115,533 | 10,832 | 2.3 | NVIDIA GeForce RTX 4090 | 0.0124 | 0.1361 |
| bert-noisy-s0 | 231,066 | 10,832 | 1.9 | NVIDIA GeForce RTX 4090 | 0.0138 | 0.0267 |
| bert-noisy-s1 | 231,066 | 10,832 | 1.8 | NVIDIA GeForce RTX 4090 | 0.0126 | 0.0253 |
| bert-noisy-s2 | 231,066 | 10,832 | 2.5 | NVIDIA GeForce RTX 4090 | 0.0126 | 0.0254 |

Packages: torch 2.14.1, transformers 5.16.1.

## Dev loss during training (mean over the seeds)

| Model | Dev set | step 1,805 | step 3,610 | step 5,415 | step 7,220 | step 9,025 | step 10,830 |
|---|---|---:|---:|---:|---:|---:|---:|
| bert-clean | clean | 0.0173 | 0.0135 | 0.0127 | 0.0128 | 0.0127 | 0.0122 |
| bert-clean | noisy | 0.0945 | 0.1048 | 0.1143 | 0.1250 | 0.1352 | 0.1335 |
| bert-noisy | clean | 0.0210 | 0.0162 | 0.0146 | 0.0129 | 0.0134 | 0.0130 |
| bert-noisy | noisy | 0.0411 | 0.0319 | 0.0285 | 0.0264 | 0.0269 | 0.0258 |

The dev sets are only logged: the model of the last step is kept.
