# PIIGuard, questions 2 and 3: how much does noise cost, and does training on noise win it back?

*Analysis report. PIIGuard team (Louis Lenoble, Sunho Cha, Suseong Kim), CSED490E, November 24, 2026.*

## Summary

Question 2 runs the three detectors of question 1 (Presidio with its rules only, Gravitee's
BERT, GLiNER) on noisy copies of the same test sets. Question 3 fine-tunes that BERT on clean
ai4privacy text, or on the same text plus one noisy copy of each record, and compares the two.
Scoring, matching and statistics are those of question 1
([report](../q1/q1_report_en.md)): micro F2 under strict matching, leak rate, 95% intervals
from a paired cluster bootstrap, paired randomization tests with Holm's correction.

1. **Missing or extra spaces cost the most**, then typos. On the six labels, BERT loses 0.069
   to 0.116 F2 when about one word in ten has a space dropped or added, GLiNER 0.014 to 0.076.
2. **GLiNER is more robust than BERT** to typos, spaces and mixed noise on all three datasets:
   on the mixed copies BERT loses 0.018 to 0.026 F2 more (95% intervals below 0). Lowercase
   text is the exception: it does not change BERT at all (its tokenizer lowercases) but costs
   GLiNER 0.028 F2 and 3.0 points of leak rate on ai4privacy.
3. **Number formats break dates.** When every date is scored (comparison A-dates), dropping
   or mixing the separators costs Presidio 0.241 F2 on Nemotron-PII and BERT 0.066.
4. **In its own domain, training on noisy copies wins back two thirds of the loss.** On
   ai4privacy, BERT fine-tuned on clean text loses 0.071 F2 on the mixed copy; the version
   trained on clean + noisy text is 0.047 [0.044, 0.050] better there, and costs 0.003 on clean
   text (detectable, below the 0.01 that counts).
5. **On the other datasets, the noisy version is better everywhere, clean text included**:
   +0.068 F2 on clean Nemotron-PII, +0.026 on clean Gretel, +0.126 and +0.060 on the mixed
   copies, and lower leak rates. Every one of these differences is established, and every seed
   of the noisy version scores above every seed of the clean one.
6. **Fine-tuning on ai4privacy alone does not make a better detector for other data.** Both
   fine-tuned versions fall below the original BERT on Gretel (0.51 to 0.53 against 0.72),
   mostly on addresses, because ai4privacy marks only the building number and street. As the
   proposal requires (Appendix B), the fine-tuned versions are compared with each other only.

The key tables of all three questions are also in a three-page PDF:
[results_tables.pdf](../results_tables/results_tables.pdf).

---

## 1. The questions

> **Q2.** How much do Presidio, BERT and GLiNER lose when the text has the mistakes people make
> when typing quickly (proposal, Section 4.2)?
>
> **Q3.** Does fine-tuning BERT on noisy text win that loss back, and what does it cost on clean
> text (proposal, Section 5.2 and Appendix B)?

Both questions reuse the test sets, labels, span clean-up, detectors, scoring and comparisons
of question 1. Only the text changes (question 2) or the BERT weights (question 3).

## 2. Noisy test sets

### 2.1 The kinds of noise

`benchmark/noise.py` makes one noisy copy of a text for each kind of mistake of Section 4.2,
plus a mix:

| Kind | Edit | Example (made up) |
| --- | --- | --- |
| `typo` | in a word of 3+ letters: a letter added, dropped, swapped with the next, or replaced by a neighbouring QWERTY key | example → exmaple |
| `space` | a space dropped between two words, or added inside a word of 4+ letters | is minji.park@... → isminji.park@... |
| `case` | a word with capitals typed in lowercase | Minji Park → minji park |
| `punct` | punctuation at the end of a sentence or clause dropped (not inside `x@y.com` or `1,234`) | sentences run together |
| `number` | the separators of a number dropped, or mixed (space, hyphen, nothing) | 010-2345-6789 → 01023456789 |
| `mixed` | each edit of one of the five kinds above, chosen at random | |

### 2.2 How the gold spans follow the text

Every edit replaces a non-empty range of the original text, and edits never overlap. An edit
is either inside a gold span or outside all of them: an edit that would cross a span edge is
not made. The spans then move with the text (a start moves by the length change of the edits
before it, an end also by the edits inside the span), keep their label (a misspelled email is
still an email), and are cleaned with the same rules as every gold span.

One more rule came from a check of the first version: an edit that would leave only spaces
between two values with the same label is not made. Dropping the comma of "Roger Leblanc,
Roger Leblanc" would let the span clean-up join the two names into one gold span, so the noise
would change the gold spans themselves. With this rule, **every noisy test set has exactly the
same number of gold spans per label as its clean version** (checked for all 36 sets).

### 2.3 Rate and seeds

A copy makes round(0.10 × words) edits, at least one: about one edit per ten words. A text with
fewer places for a kind of noise gets fewer edits (a `number` copy of a text without numbers is
left as it is). The rate and the seed (1) were fixed before any scoring. Each example draws
from its own generator, `Random("<seed>:<kind>:<example id>")`, so a copy does not depend on the
order of the examples. The A-dates versions get the same edits as the six-label versions,
because both lists of gold spans are passed together and no edit may cross either.

The noisy test sets are `data/test/<dataset>_noisy_<kind>[_dates].jsonl` (3 datasets × 6 kinds
× 2 label sets = 36 files), with the same ids as the clean ones. Each example's `meta.noise`
lists its edits and whether each one falls inside a value.

### 2.4 What the noisy test sets look like

From `results/noise/test_sets.md`:

| Dataset | Examples | Edits per example (mixed) | Edits inside a PII value (mixed) | Examples changed by `number` |
| --- | ---: | ---: | ---: | ---: |
| ai4privacy | 10,000 | 1.7 | 18.1% | 2,487 |
| Nemotron-PII | 2,000 | 13.2 | 5.2% | 1,711 |
| Gretel | 2,962 | 19.1 | 5.8% | 1,992 |

ai4privacy sentences are short and dense in PII, so about one edit in five touches a value;
in the long Nemotron-PII and Gretel documents, most edits fall in the words around the values.
Inside values, `case` (38% on ai4privacy) and `number` (59%) hit them most, `punct` almost
never (values rarely end with a sentence mark).

## 3. Fine-tuning BERT

### 3.1 Data

`experiments/prepare_train.py` reads the English records of the ai4privacy **train** split
(120,533; our ai4privacy test set comes from the validation split), holds out 5,000 of them as
a dev set (seed 0), and keeps the other 115,533 for training. It also writes one mixed-noise copy
of every record, made like the noisy test sets but with **seed 2**, so no test copy is ever
seen in training.

| File | Records | NAME | EMAIL | PHONE | ADDRESS | DATE | CARD_NUMBER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train, clean | 115,533 | 101,783 | 13,282 | 17,984 | 17,487 | 16,391 | 2,148 |
| train, noisy | 115,533 | same spans, 201,528 edits (1.7 per record, 21% inside a value) | | | | | |
| dev, clean and noisy | 5,000 each | 4,443 | 592 | 760 | 765 | 692 | 104 |

Version **(a)** learns from *train, clean*; version **(b)** from *train, clean* + *train, noisy*.

### 3.2 Labels and tags

Both versions start from the benchmark's BERT (`gravitee-io/bert-small-pii-detection`, revision
`f8c27a85c5`) and keep its 51 BIO tags, so `baselines/bert_detector.py` reads them exactly like
the original model. Our labels are taught as the model's own classes:

| Our label | BERT class |
| --- | --- |
| NAME | PERSON |
| EMAIL | EMAIL_ADDRESS |
| PHONE | PHONE_NUMBER |
| ADDRESS | LOCATION |
| DATE (every date) | DATE_TIME |
| CARD_NUMBER | CREDIT_CARD |

Every date is taught as DATE_TIME, as in comparison C: ai4privacy has no birth-date label, and
leaving its dates untagged would teach the model to forget its date class. On the six-label
comparisons the detector still maps DATE_TIME to DATE_OF_BIRTH, as for the original model.
All other tokens are `O`, so the model also learns that cities, titles and the like are not
our labels. A token takes the span its first character falls in: `B-` for the first token of a
span, `I-` for the next ones, ignored (-100) for `[CLS]` and `[SEP]`.

### 3.3 Recipe

| Setting | Value |
| --- | --- |
| Trainer | Hugging Face `Trainer`, AdamW (its default), linear schedule |
| Learning rate | 5e-5, warm-up over the first 6% of the steps |
| Weight decay | 0.01 |
| Batch | 32 |
| Steps | 10,832 for **both** versions: three passes over the clean training set. Version (b) sees each of its records 1.5 times |
| Length | 256 tokens, the window `BertDetector` reads (records: median 29 tokens; 1 of 115,533 is cut) |
| Precision | bf16 |
| Checkpoint | the last step; the dev sets are only logged, never used to choose a model (the noisy dev set would favour version b) |
| Seeds | 0, 1 and 2 for each version (6 models) |

The only difference between (a) and (b) is the noisy copies: same base, settings, number of
steps and seed list. No setting was tuned. The settings are fixed in
`experiments/train_bert.py`, and each run writes its record (settings, data checksums, every
logged loss, time, GPU, package versions) to `results/training/<model>.json`.

### 3.4 Runs

Each model trained in about 2 minutes on one GPU (five on RTX 4090, one on an L40S), with
torch 2.14.1 and transformers 5.16.1 (`results/training/summary.md`). Dev loss during
training, mean over the three seeds:

| Model | Dev set | step 1,805 | 3,610 | 5,415 | 7,220 | 9,025 | 10,830 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) clean | clean | 0.0173 | 0.0135 | 0.0127 | 0.0128 | 0.0127 | 0.0122 |
| (a) clean | noisy | 0.0945 | 0.1048 | 0.1143 | 0.1250 | 0.1352 | 0.1335 |
| (b) clean + noisy | clean | 0.0210 | 0.0162 | 0.0146 | 0.0129 | 0.0134 | 0.0130 |
| (b) clean + noisy | noisy | 0.0411 | 0.0319 | 0.0285 | 0.0264 | 0.0269 | 0.0258 |

The clean-only version gets *worse* on the noisy dev set as training goes on (0.095 to 0.134),
in all three seeds: it fits the clean spelling of its training data more and more. Version (b)
improves on both dev sets.

## 4. Evaluation and statistics

- **Runs.** Presidio, BERT and GLiNER on the 36 noisy test sets, with the settings of question
  1 (GLiNER cut-off 0.35, fixed revisions): Presidio 3 minutes on CPU, BERT 16 minutes on one
  GPU, GLiNER 48 minutes on three. The six fine-tuned models on the clean and noisy test sets
  (42 sets each), read by the same `BertDetector`. All predictions are in `results/predictions/`.
- **Scores.** `experiments/score.py`, unchanged: micro P, R, F1, F2 and leak rate, strict
  matching, comparisons A, B, A-dates and B-dates. A fine-tuned version's score is the mean of
  its three seeds.
- **Intervals and tests** (`experiments/significance_noise.py`). The method of question 1: a
  paired cluster bootstrap (10,000 draws) for the 95% intervals, and paired approximate
  randomization tests. The clusters are those of the clean test set, so the noisy copies of an
  example are always drawn with it. For a fine-tuned version, the mean over the seeds is taken
  in each draw and each round, and the test swaps the outputs of the two versions for the three
  seed pairs at once. Three families of tests, each for F1, F2 and leak rate, with Holm's
  correction: the change of Presidio, BERT and GLiNER on each noisy copy (216 tests), noisy
  minus clean version on every test set (84), and the change of each fine-tuned version on
  each noisy copy (144). These families are larger than those of question 1, so the
  randomization uses 20,000 rounds.
- **Established**, as in question 1: corrected p below 0.05, interval excluding 0, and a
  difference of at least 0.01.

Below, *italic* values are not established; in the question 3 tables, **bold** differences are.
Every interval is in `results/analysis/significance_noise.md`.

## 5. Question 2: the off-the-shelf detectors on noisy text

#### F2 on the clean test set and change on each noisy copy, comparison A (six labels)

| Dataset | Detector | clean | typo | space | case | punct | number | mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ai4privacy | Presidio | 0.157 | *−0.008* | −0.019 | *0.000* | *0.000* | *0.000* | *−0.007* |
|  | BERT | 0.796 | −0.089 | −0.116 | *0.000* | −0.016 | *−0.002* | −0.058 |
|  | GLiNER | 0.867 | −0.038 | −0.076 | −0.028 | −0.011 | *−0.003* | −0.040 |
| Nemotron-PII | Presidio | 0.431 | −0.014 | −0.057 | *0.000* | *0.000* | −0.059 | −0.055 |
|  | BERT | 0.762 | −0.049 | −0.077 | *0.000* | *−0.007* | *0.000* | −0.038 |
|  | GLiNER | 0.823 | −0.010 | −0.037 | *−0.008* | *−0.005* | *−0.005* | −0.017 |
| Gretel | Presidio | 0.171 | *−0.008* | −0.017 | *0.000* | *0.000* | *−0.007* | −0.014 |
|  | BERT | 0.718 | −0.064 | −0.069 | *0.000* | −0.011 | *+0.003* | −0.033 |
|  | GLiNER | 0.713 | *−0.009* | −0.014 | *+0.004* | −0.012 | *+0.001* | *−0.007* |

Every change of 0.01 or more is established, and none of the smaller ones.

**BERT's change minus GLiNER's** (95% interval; negative: BERT loses more):

| Dataset | typo | space | mixed |
| --- | --- | --- | --- |
| ai4privacy | −0.050 [−0.056, −0.045] | −0.040 [−0.046, −0.034] | −0.018 [−0.023, −0.013] |
| Nemotron-PII | −0.038 [−0.046, −0.030] | −0.040 [−0.049, −0.032] | −0.021 [−0.028, −0.014] |
| Gretel | −0.054 [−0.062, −0.047] | −0.055 [−0.062, −0.048] | −0.026 [−0.033, −0.020] |

This difference of differences has intervals only, no test. On clean Gretel text the two are
level in F2 (question 1); on the typo, space and mixed copies GLiNER's F2 is the higher one
(0.706 against 0.685 on mixed; the two detectors were not tested against each other on the
noisy copies).

#### Leak rate (%) on the clean test set and change in points, comparison A (higher is worse)

| Dataset | Detector | clean | typo | space | case | punct | number | mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ai4privacy | Presidio | 86.8 | *+0.7* | +1.3 | *0.0* | *0.0* | *0.0* | *+0.5* |
|  | BERT | 11.3 | +1.8 | +3.4 | *0.0* | *0.0* | *+0.2* | +1.4 |
|  | GLiNER | 6.1 | +1.1 | +4.3 | +3.0 | *+1.0* | *+0.3* | +2.4 |
| Nemotron-PII | Presidio | 57.2 | +1.4 | +5.0 | *0.0* | *0.0* | +7.9 | +6.5 |
|  | BERT | 8.8 | *+0.5* | +2.5 | *0.0* | *−0.2* | +1.2 | +2.1 |
|  | GLiNER | 10.1 | *+0.9* | +2.6 | *+0.5* | *−0.2* | *+0.5* | +1.6 |
| Gretel | Presidio | 83.9 | *+0.8* | +1.6 | *0.0* | *0.0* | +1.1 | +1.5 |
|  | BERT | 8.3 | +1.4 | +3.0 | *0.0* | +1.2 | *+0.1* | +1.3 |
|  | GLiNER | 19.5 | +2.0 | +2.6 | *+0.1* | +2.2 | *+0.1* | +1.9 |

#### F2, comparison A-dates (every date scored)

| Dataset | Detector | clean | typo | space | case | punct | number | mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ai4privacy | Presidio | 0.178 | *−0.007* | −0.021 | *0.000* | *0.000* | −0.034 | −0.017 |
|  | BERT | 0.790 | −0.082 | −0.110 | *0.000* | −0.016 | −0.022 | −0.059 |
|  | GLiNER | 0.872 | −0.035 | −0.075 | −0.026 | −0.011 | *−0.010* | −0.040 |
| Nemotron-PII | Presidio | 0.532 | −0.010 | −0.050 | *0.000* | *0.000* | −0.241 | −0.168 |
|  | BERT | 0.778 | −0.042 | −0.067 | *0.000* | *−0.001* | −0.066 | −0.072 |
|  | GLiNER | 0.801 | *−0.004* | −0.033 | *−0.008* | *−0.004* | −0.035 | −0.033 |
| Gretel | Presidio | 0.227 | *−0.006* | −0.013 | *0.000* | *0.000* | −0.081 | −0.050 |
|  | BERT | 0.787 | −0.075 | −0.077 | *0.000* | −0.016 | −0.022 | −0.051 |
|  | GLiNER | 0.595 | −0.010 | −0.015 | *+0.002* | −0.012 | −0.013 | −0.016 |

### Reading

1. **Spaces, then typos.** A word glued to its neighbour or split in two changes the tokens the
   models see, so the predicted span edges move and strict matching counts a miss. Of the five
   kinds, the space copy is the largest loss of BERT and GLiNER on every dataset in comparison
   A; in A-dates, number noise comes close to it, and passes it for GLiNER on Nemotron-PII.
2. **Case is a test of the pipeline as much as of the models.** BERT's tokenizer is uncased:
   its outputs on the case copies are identical to those on the clean text, as they must be.
   GLiNER uses capitals as a cue for names, and lowercase names are missed more often.
3. **Presidio's weak point is format.** Its rules lose little on words but much on numbers,
   and most on dates once they are scored (A-dates).
4. **Noise in the words around a value matters too.** In Nemotron-PII and Gretel only 2 to 4%
   of the space edits fall inside a value, yet BERT still loses about 0.07 F2 on those copies.

## 6. Question 3: BERT fine-tuned on noisy text

#### F2 of the two fine-tuned versions, comparison A (mean of three seeds)

| Dataset | Model | clean | typo | space | case | punct | number | mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ai4privacy | (a) clean | 0.986 | 0.873 | 0.848 | 0.986 | 0.978 | 0.971 | 0.915 |
|  | (b) clean + noisy | 0.983 | 0.951 | 0.940 | 0.983 | 0.976 | 0.981 | 0.962 |
| | difference | *−0.003* | **+0.079** | **+0.092** | *−0.003* | *−0.003* | **+0.011** | **+0.047** |
| Nemotron-PII | (a) clean | 0.713 | 0.596 | 0.601 | 0.713 | 0.684 | 0.690 | 0.630 |
|  | (b) clean + noisy | 0.781 | 0.742 | 0.730 | 0.781 | 0.750 | 0.786 | 0.756 |
| | difference | **+0.068** | **+0.145** | **+0.130** | **+0.068** | **+0.067** | **+0.096** | **+0.126** |
| Gretel | (a) clean | 0.508 | 0.417 | 0.425 | 0.508 | 0.509 | 0.509 | 0.461 |
|  | (b) clean + noisy | 0.534 | 0.499 | 0.496 | 0.534 | 0.548 | 0.536 | 0.521 |
| | difference | **+0.026** | **+0.082** | **+0.071** | **+0.026** | **+0.039** | **+0.027** | **+0.060** |

Seed standard deviations are at most 0.002 on ai4privacy and 0.025 elsewhere. Wherever a
difference is established, the lowest seed of (b) is above the highest seed of (a); the
narrowest margin is Gretel `punct` (0.002).

#### Loss, gain and cost, comparison A, F2 [95% interval]

| Dataset | Loss of (a) on mixed | Won back by (b) on mixed | Share | Cost on clean, (a) − (b) | Leak on mixed, (a) → (b) |
| --- | --- | --- | ---: | --- | --- |
| ai4privacy | 0.071 [0.067, 0.074] | +0.047 [0.044, 0.050] | 66% | *+0.003* [0.001, 0.004] | *2.3% → 2.0%* |
| Nemotron-PII | 0.082 [0.074, 0.091] | +0.126 [0.117, 0.135] | 153% | −0.068 [−0.075, −0.061] | 24.6% → 12.9% |
| Gretel | 0.047 [0.042, 0.051] | +0.060 [0.054, 0.065] | 128% | −0.026 [−0.032, −0.021] | 43.1% → 37.8% |

The share is above 100% out of domain because (b) is better on clean text too; there, the F2
of each version says more than the share. On ai4privacy the share is 70% for typos, 67% for
spaces and 69% for numbers. Version (b) still loses on noisy text, only less: on ai4privacy
mixed, 0.021 instead of 0.071.

#### Comparison A-dates

| Dataset | Model | clean | typo | space | case | punct | number | mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ai4privacy | (a) clean | 0.987 | 0.881 | 0.857 | 0.987 | 0.980 | 0.950 | 0.914 |
|  | (b) clean + noisy | 0.985 | 0.956 | 0.945 | 0.985 | 0.978 | 0.982 | 0.965 |
| | difference | *−0.002* | **+0.074** | **+0.088** | *−0.002* | *−0.003* | **+0.032** | **+0.051** |
| Nemotron-PII | (a) clean | 0.743 | 0.644 | 0.645 | 0.743 | 0.718 | 0.604 | 0.597 |
|  | (b) clean + noisy | 0.817 | 0.782 | 0.773 | 0.817 | 0.792 | 0.802 | 0.785 |
| | difference | **+0.074** | **+0.139** | **+0.128** | **+0.074** | **+0.073** | **+0.199** | **+0.187** |
| Gretel | (a) clean | 0.492 | 0.423 | 0.427 | 0.492 | 0.491 | 0.440 | 0.430 |
|  | (b) clean + noisy | 0.519 | 0.489 | 0.489 | 0.519 | 0.527 | 0.513 | 0.506 |
| | difference | **+0.027** | **+0.066** | **+0.062** | **+0.027** | **+0.035** | **+0.073** | **+0.076** |

With every date scored, the number copies show the largest gain: the clean-only version
misses dates whose separators were changed (Nemotron-PII 0.604), the noisy version much less
(0.802).

### 6.1 Why both fine-tuned versions are below the original BERT out of domain

F2 per label on the clean test sets, comparison A (fine-tuned: mean of three seeds):

| Dataset | Label | Gold values | Original BERT | (a) clean | (b) clean + noisy |
| --- | --- | ---: | ---: | ---: | ---: |
| Gretel | ADDRESS | 2,322 | 0.80 | 0.13 | 0.13 |
| Gretel | NAME | 4,475 | 0.85 | 0.73 | 0.77 |
| Gretel | EMAIL | 837 | 0.85 | 0.70 | 0.81 |
| Gretel | CARD_NUMBER | 76 | 0.71 | 0.36 | 0.43 |
| Nemotron-PII | EMAIL | 1,091 | 0.95 | 0.63 | 0.82 |
| Nemotron-PII | CARD_NUMBER | 257 | 0.95 | 0.42 | 0.65 |
| Nemotron-PII | ADDRESS | 324 | 0.45 | 0.60 | 0.65 |

- **Addresses.** ai4privacy marks only the building number and the street as an address; Gretel
  marks the whole address. Trained on ai4privacy, both versions learn the short edges, and
  strict matching fails on Gretel (0.80 to 0.13). This is a difference between the datasets'
  conventions, not an effect of noise, and the noisy copies do not change it.
- **Training data.** The original BERT was trained on data that includes Gretel's PII finance
  documents (question 1, contamination analysis S3); retrained on ai4privacy only, it loses that
  advantage.
- **Emails and card numbers** fall out of domain for both versions, and the noisy copies halve
  the fall (Nemotron-PII emails: 0.63 to 0.82).
- Without names and addresses (comparison B), both fine-tuned versions are *above* the original
  BERT on Gretel (0.507 and 0.524 against 0.459), which agrees with the above.

This is why the proposal compares the two fine-tuned versions with each other only: they share
everything but the noisy copies, while the original BERT was trained on other data.

## 7. Conclusions

1. Noise costs every detector something, and spacing mistakes cost the most. For a filter in
   front of a language model, BERT is the more fragile of the two neural detectors; GLiNER is
   more robust, except to lowercase names.
2. Training on noisy copies works in its own domain: about two thirds of the loss is won back,
   at no meaningful cost on clean text.
3. Out of domain, the noisy copies seem to act as a regularizer as well: the clean-only version
   appears to over-fit the clean spelling of ai4privacy (its noisy dev loss rises during
   training), and the noisy version is better on clean text too.
4. Fine-tuning on a single dataset with its own span conventions is not enough to beat the
   original BERT on other data; a better detector would need training data whose label edges
   match those of the target text.

## 8. Limitations

- **Synthetic noise.** The edits come from a script, at one rate (0.10). Real typing mistakes
  have another distribution, and other rates were not tried.
- **One training domain.** Both versions learn from ai4privacy only, so out-of-domain results
  mix the effect of noise with that of the span conventions (Section 6.1).
- **Seeds.** The tests cover the variation of the test examples, with the three training seeds
  taken as fixed; they say nothing about other training runs. The seed-level check (every seed
  of (b) above every seed of (a) wherever a difference is established) covers part of that.
- **Overlap.** 89 of the 10,000 ai4privacy test sentences (0.9%) also appear, word for word, in
  the ai4privacy train split. Both versions saw them, so the comparison between versions is
  fair, but the in-domain scores of both are slightly optimistic.
- **No tuning.** One set of training settings, chosen before the runs, the same for both
  versions. Better settings may exist for either.
- **Strict matching only.** As in question 1, a value found with one wrong edge counts as missed;
  the looser rules of question 1's sensitivity analysis were not rerun on the noisy copies.

## 9. Reproducing the results

The datasets and the models are not in git (see the README); predictions, scores, training
records and intervals are.

```bash
uv run python experiments/make_noisy.py                 # 36 noisy test sets (about 10 s)
uv run python experiments/run_benchmark.py bert --versions noisy_typo noisy_space noisy_case \
    noisy_punct noisy_number noisy_mixed                 # also presidio, gliner; and --dates
uv run python experiments/prepare_train.py              # fine-tuning data (about 15 s)
CUDA_VISIBLE_DEVICES=0 uv run python experiments/train_bert.py clean --seed 0   # about 2 min; seeds 0-2, clean and noisy
uv run python experiments/run_benchmark.py bert --model models/bert-clean-s0 --name bert-clean-s0 \
    --versions clean noisy_typo noisy_space noisy_case noisy_punct noisy_number noisy_mixed   # and --dates
uv run python experiments/score.py
uv run python experiments/summarize_noise.py            # results/noise/summary.md (--dates)
uv run python experiments/summarize_data.py             # results/noise/test_sets.md, results/training/summary.md
uv run python experiments/significance_noise.py         # results/analysis/significance_noise.md (about 3 min)
uv run python experiments/results_tables.py             # the tables of docs/reports/results_tables/
```

| File | Content |
| --- | --- |
| `benchmark/noise.py`, `tests/test_noise.py` | the noise and its tests |
| `results/noise/test_sets.md` | edits made in each noisy test set |
| `results/training/` | the record of each training run, and `summary.md` |
| `results/predictions/<model>/<dataset>/` | every output |
| `results/scores.md`, `results/noise/summary*.md` | the scores |
| `results/analysis/significance_noise.md` | every interval and test |
