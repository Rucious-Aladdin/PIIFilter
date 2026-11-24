# PII Filter

CSED490E term project (POSTECH, Fall 2026). An on-device filter that finds personal
information (PII) in a chatbot prompt, replaces each value with a placeholder such as
`<NAME_1>` before the prompt leaves the device, and puts the real values back into the
LLM's answer.

```
user prompt → detector (NER) → mask → cloud LLM → unmask → answer shown to the user
                                 └── placeholder → value mapping stays on the device
```

The full plan is in [docs/PIIGuard_Proposal_Merged.docx](docs/PIIGuard_Proposal_Merged.docx)
and the slides in [docs/PII_Filter_long.pdf](docs/PII_Filter_long.pdf).

## Repository layout

```
PIIFilter/                 the filter itself
  masking.py               Span, mask() and unmask() with numbered placeholders
  llm.py                   LLM backends: local (transformers), openai, anthropic
  pipeline.py              run_filter(): detect → mask → LLM → unmask, returns every step
  device.py                pick_device(): the GPU when there is one, the CPU otherwise
baselines/                 PII detectors compared in the benchmark (DETECTORS in __init__.py)
  presidio_detector.py     Presidio, rules only (no NLP model)
  bert_detector.py         BERT (gravitee-io/bert-small-pii-detection) or a fine-tuned copy
  gliner_detector.py       GLiNER (urchade/gliner_multi_pii-v1), given our label names
  windows.py               overlapping windows for texts longer than a model can read
benchmark/                 code shared by the benchmark scripts
  data.py                  the three datasets: sources, readers, label mapping, JSONL format
  spans.py                 span clean-up shared by gold spans and detector outputs
  scoring.py               strict matching: P, R, F1, F2 and the leak rate
  analysis.py              confidence intervals, paired tests, Holm's correction (Q1 analysis)
  noise.py                 noisy copies of a text whose gold spans move with it (Q2, Q3)
experiments/
  download_data.py         download the benchmark datasets into data/raw/
  prepare_data.py          draw the clean test sets (and their dates versions) into data/test/
  run_benchmark.py         run one detector on the test sets into results/predictions/
  score.py                 score every predictions file into results/scores.md
  make_noisy.py            make the noisy test sets (Q2) into data/test/
  prepare_train.py         draw the fine-tuning data (Q3) into data/train/
  train_bert.py            fine-tune BERT on clean or clean + noisy data (Q3) into models/
  summarize_noise.py       tables of questions 2 and 3 (results/noise/)
  significance_noise.py    intervals and paired tests of questions 2 and 3 (results/analysis/)
  results_tables.py        LaTeX tables of the key results of Q1 to Q3 (docs/results_tables/)
  tune_gliner.py           choose GLiNER's cut-off once (results/gliner_threshold.json)
  time_detectors.py        time the detectors on this computer (results/timing.json)
  significance.py          scores with 95% intervals and paired tests (results/analysis/)
  sensitivity.py           looser matching, GLiNER cut-off, document length
  contamination.py         overlap of BERT's training data with the Gretel test set
  paper_results.py         main table, over-masking and error types (results/paper/)
  measure_resources.py     time and memory of each detector on CPU (results/paper/)
  toy_experiment.py        toy experiment and demo (proposal Sections 5.3 and 5.4)
  summarize_demo.py        how many prompts pass each check of the demo
  scenario_samples/        toy experiment inputs, one JSON file per prompt
  demo_samples/            demo inputs: 20 scenarios, clean and noisy (fictional values)
data/raw/                  downloaded datasets (git-ignored)
data/test/                 test sets drawn from them, clean and noisy (git-ignored)
data/train/                fine-tuning data for question 3 (git-ignored)
models/                    fine-tuned BERT versions (git-ignored)
results/predictions/       detector outputs: <detector>/<dataset>/<version>.jsonl
results/scores.md, .json   benchmark scores
results/without_name_address/   the same without NAME and ADDRESS (see Question 1)
results/gliner_threshold.json, timing.json   GLiNER cut-off, detector timings
results/analysis/          intervals and tests of questions 1 to 3, sensitivity, contamination
results/noise/             tables of questions 2 and 3 (summary.md, summary_dates.md, ...)
data/analysis/             GLiNER run at a low cut-off, for the sensitivity analysis (git-ignored)
results/paper/             tables of the paper
results/toy/<backend>/     toy experiment outputs (toy_results.json, toy_results.md)
slides/toy_experiment/     Beamer frames for the Toy Experiment slides + compiled preview
docs/                      proposal, presentation, the Q1 report (q1_report/: en, fr, ko),
                           key result tables of Q1 to Q3 (results_tables/results_tables.pdf)
tests/                     unit tests (uv run pytest)
pyproject.toml, uv.lock    environment (uv)
```

Labels used everywhere: `NAME`, `EMAIL`, `PHONE`, `ADDRESS`, `DATE_OF_BIRTH`, `CARD_NUMBER`.
In comparison C (below), `DATE` takes the place of `DATE_OF_BIRTH`.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned
in `.python-version`; uv downloads it if needed.

```bash
# install uv once (skip if `uv --version` works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create .venv and install the exact versions in uv.lock
uv sync

# run the tests
uv run pytest
```

`uv sync` installs the same PyTorch build on every computer, with its CUDA libraries (about
5 GB in `.venv`). The same environment works on computers without a GPU.

- API keys go in a `.env` file at the repository root (it is git-ignored and loaded by the
  experiment script). The default `openai` backend needs `OPENAI_API_KEY`
  (https://platform.openai.com/api-keys; the API is billed separately from ChatGPT).

  ```
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...
  ```

- The first run downloads GLiNER from Hugging Face (about 1 GB). GLiNER runs on the GPU when
  there is one and on the CPU otherwise; `--device cpu` or `--device cuda` forces the choice.
- Only the optional `local` backend needs a large GPU: Qwen2.5-7B-Instruct is about 15 GB
  to download and needs about 16 GB of GPU memory in bf16.

## The benchmark

| Question | What it measures | Status |
| --- | --- | --- |
| Q1 | Presidio (rules only), BERT and GLiNER on clean text | done, see the [report](docs/q1_report/q1_report_en.md) ([fr](docs/q1_report/q1_report_fr.md), [ko](docs/q1_report/q1_report_ko.md)) |
| Q2 | the same detectors on noisy copies of the test sets | done, see [Question 2](#question-2-noisy-text) |
| Q3 | BERT: original, fine-tuned on clean text, fine-tuned on clean + noisy text | done, see [Question 3](#question-3-fine-tuning-bert-on-noisy-text) |

The key tables of all three questions, with intervals and tests, are in one three-page PDF:
[docs/results_tables/results_tables.pdf](docs/results_tables/results_tables.pdf).

Question 1 uses four comparisons, because Presidio's rules cannot find names or addresses,
and Presidio and BERT only have a class for any date. A-dates and B-dates together are
called comparison C.

| Comparison | Labels scored |
| --- | --- |
| A | the six labels |
| B | EMAIL, PHONE, DATE_OF_BIRTH, CARD_NUMBER |
| A-dates | as A, with every date under DATE in place of DATE_OF_BIRTH |
| B-dates | as B, with every date under DATE in place of DATE_OF_BIRTH |

The scripts run in this order. Predictions and scores are in the repository, so steps 3 and
4 (the slow ones) can be skipped; the datasets are not, so steps 1 and 2 are needed. Step 7
is not in the repository either: step 8 stops and writes nothing without it.

```bash
uv run python experiments/download_data.py            # 1. datasets
uv run python experiments/prepare_data.py             # 2. test sets
uv run python experiments/tune_gliner.py              # 3. GLiNER's cut-off
uv run python experiments/run_benchmark.py presidio   # 4. predictions (also bert, gliner; --dates for C)
uv run python experiments/score.py                    # 5. scores
uv run python experiments/significance.py             # 6. intervals and paired tests
uv run python experiments/run_benchmark.py gliner --threshold 0.05 \
    --out data/analysis/predictions --name gliner_cutoff_0.05   # 7. for step 8 (about 1 h 45 on CPU)
uv run python experiments/sensitivity.py              # 8. sensitivity analyses
uv run python experiments/contamination.py            # 9. did BERT see our Gretel documents?
```

## Benchmark datasets

```bash
uv run python experiments/download_data.py   # about 830 MB, into data/raw/
```

The script downloads only the files we use, at the revision pinned in
`benchmark/data.py`, so every computer gets the same data.

| Dataset | Files | English records |
| --- | --- | --- |
| [ai4privacy open-pii-masking-500k](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) | train + validation (our test set) | 120,533 train, 30,160 validation |
| [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | test | 100,000 (50,000 `us` + 50,000 `intl` locale) |
| [Gretel PII finance](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | test | 2,962 of 5,594 (7 languages) |

Each Nemotron `uid` has two documents, one per locale (`us` and `intl`); we keep both.

### Test sets

```bash
uv run python experiments/prepare_data.py                    # all three, at our sizes
uv run python experiments/prepare_data.py gretel --n 500     # one dataset, another size
```

Our sizes (`test_size` in `benchmark/data.py`), chosen after timing the detectors: 10,000
ai4privacy sentences, 2,000 Nemotron documents, and all 2,962 English Gretel documents.

| Our label | ai4privacy | Nemotron-PII | Gretel |
| --- | --- | --- | --- |
| NAME | 8,748 | 2,053 | 4,475 |
| EMAIL | 1,185 | 1,091 | 837 |
| PHONE | 1,567 | 464 | 531 |
| ADDRESS | 1,533 | 324 | 2,322 |
| DATE_OF_BIRTH | - | 363 | 143 |
| CARD_NUMBER | 185 | 257 | 76 |
| Examples without any of our labels | 1,493 | 415 | 711 |

Each test set is a random sample of the English test records, drawn with a fixed seed
(`--seed`, default 0), so every computer gets the same examples. It is written to
`data/test/<dataset>_clean.jsonl`, one example per line:

```json
{"id": "ai4privacy-5989045", "text": "Hi, this is Medeni Malyote and I need your QH@aol.com address.", "spans": [{"start": 12, "end": 26, "label": "NAME"}, {"start": 43, "end": 53, "label": "EMAIL"}], "meta": {"region": "CA"}}
```

`meta` keeps a few dataset details for analysis (Nemotron: locale, document format and
type). Records without any of our labels are kept, since they show whether a detector
masks text that is not PII. `<dataset>_clean.meta.json` gives the size, the seed, the
labels scored on the test set, the gold spans per label and a SHA-256 checksum of the
file, to check that two computers have the same test set.

**Dates version (comparison C).** Presidio and BERT have no birth-date class, only one for
any date, so the same script also writes `data/test/<dataset>_clean_dates.jsonl`: the same
examples and spans, except that every date is gold under `DATE` and nothing is
`DATE_OF_BIRTH`. Times alone are not counted.

| Dataset labels that become `DATE` | Left out | `DATE` spans |
| --- | --- | --- |
| ai4privacy: `DATE` (it includes dates with a time, `1975-08-08T00:00:00`) | `TIME` | 1,387 |
| Nemotron-PII: `date`, `date_of_birth`, `date_time` (`2025-07-13T17:43:03`) | `time` | 2,126 |
| Gretel: `date`, `date_of_birth`, `date_time` | `time` | 5,768 |

Gold labels are mapped to ours as in Table 5 of the proposal (`benchmark/data.py`), then
cleaned by `benchmark/spans.py`, the same rules later used on detector outputs:

1. spans outside the text are dropped, and span edges move inward past whitespace and past
   a pair of brackets or quotes around the whole span (`[x@y.com]` → `x@y.com`; the `(` of
   `(555) 123-4567` stays);
2. a leading title (Mr, Mrs, Ms, Dr, ..., and Mister, Madame, Master, Mayoress, Mstr, which
   ai4privacy labels TITLE as often as Mr) is removed from NAME spans;
3. overlapping spans: the one that starts first is kept;
4. spans with the same label that only spaces separate are joined ("Minji" + "Park"). A
   comma or a line break keeps them apart, so "Doe, Jane" stays two NAME spans, and so do
   190 ai4privacy addresses written as "Marine Drive, 1382".

### Running the detectors

```bash
uv run python experiments/run_benchmark.py presidio     # every dataset, clean version
uv run python experiments/run_benchmark.py bert --datasets nemotron gretel
uv run python experiments/run_benchmark.py gliner       # uses results/gliner_threshold.json
uv run python experiments/run_benchmark.py bert --model models/bert-noisy --name bert-noisy
uv run python experiments/run_benchmark.py gliner --dates   # dates versions (comparison C)
```

`--dates` reads the `_dates` test sets. Presidio and BERT then call their date class
(`DATE_TIME`) `DATE` instead of `DATE_OF_BIRTH`, and GLiNER is asked for "date" instead of
"date of birth", with the same cut-off.

Each run writes `results/predictions/<name>/<dataset>/<version>.jsonl` one line at a time.
After an interruption, run the same command again: examples already done are skipped, and
the run stops if the model, revision or cut-off differ from the file's. A
`.meta.json` next to each file records the model, revision, cut-off, computer, packages,
the time taken and the checksum of the test set; `score.py` skips predictions made on
another version of a test set. Run one detector at a time on a computer, since each one
already uses every core. The clean versions take about 20 s with Presidio, 6 min with BERT
and 1 h 20 with GLiNER on a laptop CPU, so different computers can take different runs.

### Scoring

A detector's output on one test set goes to
`results/predictions/<detector>/<dataset>/<version>.jsonl` (`<version>` is `clean` or a
noisy copy), one line per example, linked to it by its id:

```json
{"id": "ai4privacy-5989045", "spans": [{"start": 12, "end": 26, "label": "NAME", "score": 0.97}]}
```

```bash
uv run python experiments/score.py   # scores every predictions file found
```

The detected spans are first cleaned with the same rules as the gold spans. Then, with
strict matching (proposal, Section 4.3), a detection is a true positive (TP) only if its
start, end and label are exactly those of a gold span; other detections are false
positives (FP), and gold spans without an exact match are false negatives (FN). Labels a
dataset does not have are not scored on it, and detections with those labels are dropped.

- P = TP / (TP + FP), R = TP / (TP + FN), F1 = 2PR / (P + R), F2 = 5PR / (4P + R)
- Leak rate: the share of gold spans that the detections, whatever their label, do not
  fully cover. These values would reach the LLM in clear, at least in part.
- Per-label scores, and micro averages (TP, FP, FN summed over the labels first).
- A score that does not exist is `null` in JSON and `n/a` in the tables, not 0: precision
  when nothing was detected (0 / 0), and recall, F1, F2 and the leak rate when there was
  nothing to find. F1 and F2 are 0 when there was something to find and none was found.

`results/scores.md` has one table per detector and version, like Table 7 of the proposal:
label cells give F2 (support), and P, R, F1, F2 and the leak rate are micro averages.
`results/scores.json` has every count and score.

`--labels` scores only some labels, and then needs `--out` so the full scores are kept.
Presidio's rules cannot find names or addresses, so we also compare every detector without
them; the micro averages then cover these labels only:

```bash
uv run python experiments/score.py --labels EMAIL PHONE DATE_OF_BIRTH DATE CARD_NUMBER \
    --out results/without_name_address
```

Each test set is scored on the labels it has among these: DATE_OF_BIRTH on the clean
versions, DATE on the `_dates` versions.

### Detectors

Each detector in `baselines/` has `detect(text) -> list[Span]`, with character positions
and our labels. The models are used as published, at a pinned revision, with their
default settings; the wrapper only maps labels and splits long texts.

| Detector | Model | Labels | Long texts |
| --- | --- | --- | --- |
| `presidio` | Presidio's pattern recognizers with `NoOpNlpEngine` (no spaCy model) | EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, DATE_TIME → EMAIL, PHONE, CARD_NUMBER, DATE_OF_BIRTH | no limit |
| `bert` | `gravitee-io/bert-small-pii-detection`, pipeline with `aggregation_strategy="simple"` | PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME, CREDIT_CARD → our six; 19 other classes dropped | windows of 256 tokens, 64 shared |
| `gliner` | `urchade/gliner_multi_pii-v1`, given "person", "email", "phone number", "street address", "date of birth", "credit card number" | already ours | windows of 384 words, 64 shared |

- Presidio finds no names or addresses, by design. With no model there are no tokens, so
  context words cannot raise its scores; it keeps every match anyway.
- BERT's files set no maximum length, so without windows it would fail beyond 512 tokens
  (about 6% of Nemotron and Gretel documents). 256 is the `max_length` in its tokenizer
  config. Its tokenizer lowercases every text: this model never sees case.
- GLiNER reads at most 384 words (`max_len`) and silently drops the rest: without windows,
  it would miss the end of 11% of Nemotron and 25% of Gretel documents. Where two windows
  find overlapping spans, the longer one is kept (the rule the BERT pipeline uses).

GLiNER's cut-off is chosen once, before any test (proposal, Section 5.2):

```bash
uv run python experiments/tune_gliner.py     # about 1 minute on a CPU
```

It runs GLiNER on 300 sentences of the ai4privacy *train* split and keeps the cut-off with
the best micro F2: **0.35** (`results/gliner_threshold.json`). F2 hardly moves between 0.25
and 0.60 (0.859 to 0.861) on clean text; the default 0.5 gives 0.860.

```bash
uv run python experiments/time_detectors.py  # about 3 minutes on a CPU
```

On a laptop CPU with 8 cores, per example: Presidio 0.001 to 0.004 s, BERT 0.009 to
0.056 s, GLiNER 0.11 s (ai4privacy) to 0.99 s (Gretel). The whole plan (Presidio, GLiNER
and three BERT versions on 6 versions of each test set) takes about 0.2 h per 1,000
ai4privacy examples, 0.9 h per 1,000 Nemotron and 1.9 h per 1,000 Gretel examples.

### Question 1: clean text

Micro F2 / leak rate on the clean test sets, all six labels (per-label scores in
`results/scores.md`):

| Detector | ai4privacy (10,000) | Nemotron-PII (2,000) | Gretel (2,962) |
| --- | --- | --- | --- |
| Presidio, rules only | 0.157 / 0.868 | 0.431 / 0.572 | 0.171 / 0.839 |
| BERT (Gravitee) | 0.796 / 0.113 | 0.762 / **0.088** | **0.718** / **0.083** |
| GLiNER (urchade) | **0.867** / **0.061** | **0.823** / 0.101 | 0.713 / 0.195 |

Without NAME and ADDRESS, the labels Presidio's rules can find (EMAIL, PHONE,
DATE_OF_BIRTH, CARD_NUMBER; ai4privacy has no DATE_OF_BIRTH), in
`results/without_name_address/scores.md`:

| Detector | ai4privacy | Nemotron-PII | Gretel |
| --- | --- | --- | --- |
| Presidio, rules only | 0.616 / 0.406 | 0.757 / 0.103 | 0.569 / 0.157 |
| BERT (Gravitee) | 0.891 / **0.054** | 0.757 / 0.024 | 0.459 / **0.030** |
| GLiNER (urchade) | **0.898** / 0.071 | **0.960** / **0.015** | **0.737** / 0.205 |

The same two comparisons with every date gold under DATE, in place of DATE_OF_BIRTH
(`_dates` test sets, comparison C; ai4privacy now has a date label too):

| Detector | All six labels: ai4privacy | Nemotron-PII | Gretel | Without NAME and ADDRESS: ai4privacy | Nemotron-PII | Gretel |
| --- | --- | --- | --- | --- | --- | --- |
| Presidio, rules only | 0.178 / 0.850 | 0.532 / 0.501 | 0.227 / 0.796 | 0.544 / 0.493 | 0.791 / 0.200 | 0.408 / 0.606 |
| BERT (Gravitee) | 0.790 / 0.106 | 0.778 / 0.141 | **0.787** / **0.090** | 0.835 / **0.049** | 0.784 / 0.139 | **0.744** / **0.084** |
| GLiNER (urchade) | **0.872** / **0.057** | **0.801** / **0.137** | 0.595 / 0.355 | **0.909** / 0.057 | **0.868** / **0.110** | 0.467 / 0.519 |

Bold marks the best value in each column. 32 of the 36 differences of F2 and all 36
differences of F1 are established. The intervals, the tests and the reading of these results
are in the [report](docs/q1_report/q1_report_en.md).

### Question 2: noisy text

```bash
uv run python experiments/make_noisy.py                      # 36 noisy test sets, about 10 s
uv run python experiments/run_benchmark.py bert --versions noisy_typo noisy_space \
    noisy_case noisy_punct noisy_number noisy_mixed           # also presidio, gliner; --dates
uv run python experiments/score.py
uv run python experiments/summarize_noise.py                 # results/noise/summary.md (--dates)
```

`benchmark/noise.py` makes the mistakes of Section 4.2 of the proposal, one kind per copy
plus a mix: `typo` (a letter added, dropped, swapped or replaced by a neighbouring key),
`space` (a space dropped between words or added inside one), `case` (capitals typed in
lowercase), `punct` (end-of-sentence or clause punctuation dropped), `number` (separators in
numbers dropped or mixed) and `mixed`. Each copy makes round(0.10 × words) edits, at least
one, fixed before any scoring (seed 1). Every edit is inside a gold span or outside all of
them, never across an edge, so the gold spans move with the text and keep their labels; an
edit that would leave only spaces between two values with the same label is not made, since
the span clean-up would join them. Every noisy test set has the same gold spans per label as
its clean version. Each example's `meta.noise` lists its edits and whether each one falls
inside a value. The test sets are `data/test/<dataset>_noisy_<kind>[_dates].jsonl`, with the
same ids as the clean ones.

Micro F2 on the clean test set and change on each noisy copy, all six labels:

| Dataset | Detector | clean | typo | space | case | punct | number | mixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ai4privacy | Presidio | 0.157 | -0.008 | -0.019 | 0.000 | 0.000 | 0.000 | -0.007 |
| | BERT | 0.796 | -0.089 | -0.116 | 0.000 | -0.016 | -0.002 | -0.058 |
| | GLiNER | 0.867 | -0.038 | -0.076 | -0.028 | -0.011 | -0.003 | -0.040 |
| Nemotron-PII | Presidio | 0.431 | -0.014 | -0.057 | 0.000 | 0.000 | -0.059 | -0.055 |
| | BERT | 0.762 | -0.049 | -0.077 | 0.000 | -0.007 | 0.000 | -0.038 |
| | GLiNER | 0.823 | -0.010 | -0.037 | -0.008 | -0.005 | -0.005 | -0.017 |
| Gretel | Presidio | 0.171 | -0.008 | -0.017 | 0.000 | 0.000 | -0.007 | -0.014 |
| | BERT | 0.718 | -0.064 | -0.069 | 0.000 | -0.011 | +0.003 | -0.033 |
| | GLiNER | 0.713 | -0.009 | -0.014 | +0.004 | -0.012 | +0.001 | -0.007 |

Missing and extra spaces cost the most, then typos. GLiNER loses less than BERT on the typo,
space and mixed copies of all three datasets (the 95% intervals of the difference exclude 0),
but lowercase text raises its leak rate (ai4privacy: 6.1% to 9.0%). BERT is not affected by
lowercase text at all, as expected: its tokenizer lowercases every text. Number formats matter
most for dates (comparison A-dates: Presidio -0.241 on Nemotron-PII). Every change of 0.01 or
more in the table is established by the paired tests below, and none of the smaller ones is.
Leak rates and the other comparisons are in `results/noise/`.

### Question 3: fine-tuning BERT on noisy text

```bash
uv run python experiments/prepare_train.py                         # data/train/, about 15 s
CUDA_VISIBLE_DEVICES=0 uv run python experiments/train_bert.py clean --seed 0   # about 2 min on a 4090
CUDA_VISIBLE_DEVICES=1 uv run python experiments/train_bert.py noisy --seed 0
uv run python experiments/run_benchmark.py bert --model models/bert-noisy-s0 --name bert-noisy-s0 \
    --versions clean noisy_typo noisy_space noisy_case noisy_punct noisy_number noisy_mixed   # and --dates
```

Both versions start from the benchmark's BERT and keep its 51 BIO tags, so `BertDetector`
reads them unchanged. They learn from the English ai4privacy *train* split (our ai4privacy
test set comes from its validation split) minus 5,000 dev records: 115,533 records for the
clean version, the same plus one mixed-noise copy of each (seed 2, not the test sets' seed)
for the noisy version. Same settings and the same 10,832 steps for both (learning rate 5e-5,
batch 32, three passes over the clean data); the last step is kept, the dev sets are only
logged. Every date is learned as BERT's date class, which ai4privacy would otherwise teach
it to forget. Three seeds of each version; the models are not in git.

Micro F2, mean over the three seeds (standard deviations 0.001 on ai4privacy, at most 0.02
elsewhere), all six labels:

| Dataset | Model | clean | typo | space | mixed |
| --- | --- | ---: | ---: | ---: | ---: |
| ai4privacy (in-domain) | fine-tuned on clean | 0.986 | 0.873 | 0.848 | 0.915 |
| | fine-tuned on clean + noisy | 0.983 | 0.951 | 0.940 | 0.962 |
| Nemotron-PII | fine-tuned on clean | 0.713 | 0.596 | 0.601 | 0.630 |
| | fine-tuned on clean + noisy | 0.781 | 0.742 | 0.730 | 0.756 |
| Gretel | fine-tuned on clean | 0.508 | 0.417 | 0.425 | 0.461 |
| | fine-tuned on clean + noisy | 0.534 | 0.499 | 0.496 | 0.521 |

On ai4privacy the noisy copies win back two thirds of the loss (mixed: 66%) for 0.003 of
F2 on clean text. On the other two datasets the noisy version is better on clean text too.
Both versions saw ai4privacy-style sentences, so they are compared with each other, not with
the original BERT or the other detectors (proposal, Appendix B): fine-tuning on ai4privacy
alone lowers the scores on Gretel, whose documents were in the original BERT's training
data, mostly through addresses (F2 0.80 to 0.12), because ai4privacy marks only the building
number and street. All columns and comparisons are in `results/noise/`. In F2 on the six
labels, every difference between the two versions is established on Nemotron-PII and Gretel,
on clean text too; on ai4privacy those on the typo, space, number and mixed copies are, and the
clean cost (0.003) is detectable but below 0.01.

### Intervals and tests of questions 2 and 3

```bash
uv run python experiments/significance_noise.py   # results/analysis/significance_noise.md (about 3 min, 6 processes)
uv run python experiments/results_tables.py       # docs/results_tables/tables/
cd docs/results_tables && latexmk -pdf results_tables.tex
```

The method of question 1 (below), with the clusters of the clean test set, so a noisy copy is
always drawn with its clean example. A fine-tuned version's score is the mean of its three
seeds, in each bootstrap draw and randomization round too; the test swaps the two versions'
outputs for the three seed pairs at once. Three families of tests, each for F1, F2 and leak
rate, with Holm's correction: change on a noisy copy of Presidio, BERT and GLiNER (216 tests),
noisy-trained minus clean-trained BERT on every test set (84), and change on a noisy copy of
each fine-tuned version (144). The randomization uses 20,000 rounds because these families
are larger than those of question 1.

### Analysis of question 1

```bash
uv run python experiments/significance.py      # 95% intervals, paired tests, Holm (about 2 min)
uv run python experiments/sensitivity.py       # S1 looser matching, S2 GLiNER's cut-off, S4 length
uv run python experiments/contamination.py     # S3: downloads BERT's training corpus (43 MB)
```

- Intervals: paired cluster bootstrap (10,000 draws, seed 0). Examples that are not
  independent are drawn together: the two locales of a Nemotron `uid`, identical texts.
- Tests: two-sided paired approximate randomization (10,000 rounds) of the difference of
  micro F1, of micro F2 and of leak rate, with Holm's correction over the 36 tests of each
  metric. A difference counts if the corrected p-value is below 0.05, its interval excludes
  0, and it is at least 0.01.
- S2 needs the GLiNER run at a low cut-off (step 7 above), kept out of `results/predictions`.

### Tables of the paper

```bash
uv run python experiments/paper_results.py                   # comparison A (about 20 s)
uv run python experiments/paper_results.py --comparison A-dates
uv run python experiments/measure_resources.py               # about 4 minutes on a CPU
```

| Output in `results/paper/` | Content |
| --- | --- |
| `main_table.md`, `.tex` | P, R, F1, F2, leak rate; bold only where the tests establish the best detector |
| `overmasking.md` | share of the text outside every gold span that the filter masks |
| `error_types.md` | strict errors by type; the totals are checked against `results/scores.json` |
| `resources.md`, `.json` | time per example, peak RAM, parameters and threads, on CPU |

Other comparisons get a suffix (`main_table_A-dates.md`). One example per error type goes to
`data/analysis/error_examples.md`, which stays out of git because it quotes dataset text.

## Toy experiment

The toy experiment checks that the whole filter works end to end on two prompts from the
proposal: a message to a landlord typed carefully (clean) and carelessly (noisy). It gives
no scores. For each prompt it records the output of every step and answers three questions:

- **Private?** No PII value (or part of one) appears in the text sent to the LLM.
  Checked automatically against the gold spans in the scenario file.
- **Enough context?** The LLM answer does the task and uses every placeholder.
- **Usable answer?** After unmasking, no placeholder is left in the answer.

The last two are automatic proxies; confirm them by reading the answers in `toy_results.md`.

### Run

```bash
# OpenAI (default; model gpt-5.5) - the setting used for the slides
uv run python experiments/toy_experiment.py

# Anthropic (model claude-opus-5-5)
uv run python experiments/toy_experiment.py --backend anthropic

# local open-weight LLM (Qwen/Qwen2.5-7B-Instruct, greedy decoding)
uv run python experiments/toy_experiment.py --backend local
```

Hosted models do not decode deterministically, so the wording of the answers can change
from run to run.

| Option          | Default                          | Meaning                                                  |
| --------------- | -------------------------------- | -------------------------------------------------------- |
| `--backend`   | `openai`                       | `openai`, `anthropic` or `local`                   |
| `--model`     | per backend                      | LLM model id, e.g. `--model gpt-5-mini`                |
| `--threshold` | `0.5`                          | GLiNER confidence cut-off                                |
| `--device`    | `cuda` if available, else `cpu` | device for GLiNER (the local LLM uses the GPU if any)   |
| `--samples`   | `experiments/scenario_samples` | folder of scenario JSON files                            |
| `--out`       | `results/toy`                  | outputs go to `<out>/<backend>/`                       |

To pick a GPU, prefix the command with e.g. `CUDA_VISIBLE_DEVICES=0`.

The script writes `results/toy/<backend>/toy_results.json` (every step, spans with
confidences, check details, model ids and package versions) and `toy_results.md` (the
same as tables), and prints the Markdown.

The `anthropic` backend uses the Messages API with `effort: medium` and opts into the
server-side refusal fallback (`fallbacks: "default"`) for models that support it.

### Scenario samples

Each file in `experiments/scenario_samples/` is one prompt. Files are run in name order.

```json
{
  "id": "landlord_clean",
  "title": "Example 1 (clean)",
  "prompt": "Write a short message to my landlord. I'm Minji Park from 41 Maple Street, ...",
  "gold": [
    {"label": "NAME", "value": "Minji Park"},
    {"label": "ADDRESS", "value": "41 Maple Street"},
    {"label": "PHONE", "value": "010-2345-6789"}
  ]
}
```

To add a scenario, add a file such as `03_email_card.json`. Each gold `value` must appear
in `prompt` exactly as written.

### Results (gpt-5.5)

Full outputs: [results/toy/openai/toy_results.md](results/toy/openai/toy_results.md).

|                                            | Example 1 (clean)     | Example 2 (noisy)     |
| ------------------------------------------ | --------------------- | --------------------- |
| GLiNER confidence (NAME / ADDRESS / PHONE) | 0.999 / 0.969 / 1.000 | 0.686 / 0.921 / 0.985 |
| Private?                                   | Yes                   | Yes                   |
| Enough context?                            | Yes                   | Yes                   |
| Usable answer?                             | Yes                   | Yes                   |

Answer shown to the user (clean): *Hi, this is Minji Park from 41 Maple Street. My heater is
broken and needs repair. Please call me at 010-2345-6789 when you can. Thank you.*

- GLiNER masked all three values in both prompts, so nothing leaked. On the noisy prompt
  the lowercase name's confidence fell from 0.999 to 0.686, closer to the 0.5 cut-off. One
  prompt cannot show how often scores fall below the cut-off; the RQ2 benchmark measures it.
- GPT-5.5 used every placeholder and all of them were restored. Two more runs (not saved)
  gave the same checks with slightly different wording.
- Unmasking restores what the user typed, so the noisy answer reads "this is minji park
  from 41 maple stret".
- An earlier run with the local Qwen2.5-7B-Instruct backend is kept in
  [results/toy/local/](results/toy/local/toy_results.md). There the LLM ended both answers
  with `<NAME_1` (no closing `>`), which simple replacement does not restore.

## Demo (20 prompts)

`experiments/demo_samples/` holds 20 everyday scenarios, each clean and noisy, with fictional
values. The demo uses the toy experiment script at the benchmark's cut-off. It calls the LLM
40 times, so it needs `OPENAI_API_KEY` and is billed.

```bash
uv run python experiments/toy_experiment.py --samples experiments/demo_samples --out results/demo --threshold 0.35
uv run python experiments/summarize_demo.py results/demo/openai/toy_results.json --out results/demo/openai/summary.md
```

The summary counts the prompts that pass each of the three checks, apart for clean and noisy
prompts, and lists the answers to read.

## Slides

`slides/toy_experiment/toy_frames.tex` holds three Beamer frames (setup, the clean example
end to end, results) for the Toy Experiment section of the deck. `preview.pdf` shows them in
a look close to the deck.

```bash
cd slides/toy_experiment
latexmk -pdf preview.tex
```

To use them in the deck, add `\input{toy_frames}` where the empty Toy Experiment slide is.
The deck's preamble needs `\usepackage[T1]{fontenc}`, `tikz`, `tcolorbox`, `booktabs` and
`colortbl`.
With TinyTeX, the packages for the preview are:
`tlmgr install beamer translator fira fontaxes mweights tcolorbox pgf booktabs colortbl cm-super`.
