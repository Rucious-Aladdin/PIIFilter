# PIIGuard, question 1: how well do off-the-shelf detectors find PII in clean text?

*Analysis report. PIIGuard team (Louis Lenoble, Sunho Cha, Suseong Kim), CSED490E, November 20, 2026.*

## Summary

We compared three off-the-shelf PII detectors on three English test sets with gold annotations:
Presidio with its rules only, a small BERT model fine-tuned for PII (Gravitee), and GLiNER
(urchade, PII version). As in the proposal, the main scores are the micro F1 and F2 under strict
matching; we also report
the share of gold PII values that would reach the language model unmasked (leak rate). Every
score comes with a 95% confidence interval, and every difference between two detectors was
tested with a paired test and corrected for multiple comparisons.

1. **On the six labels of the proposal**, GLiNER has the best F2 on ai4privacy (0.867) and
   Nemotron-PII (0.823). On Gretel, BERT (0.718) and GLiNER (0.713) cannot be told apart in F2; in F1, GLiNER is
   ahead (0.637 against 0.572).
   Presidio scores 0.16 to 0.43: its rules cannot find names or addresses.
2. **On the labels Presidio can find** (email, phone, date of birth, card number), Presidio is
   level in F2 with BERT on Nemotron-PII and ahead of it on Gretel. This comes from the date label:
   neither Presidio nor BERT has a "date of birth" class, only a class for any date, which we
   had to count as date of birth.
3. **Counting every date as PII** (comparison C, decided by the team) removes this artefact. The
   precision of Presidio's dates rises from 0.22 to 0.96 on Nemotron-PII, and BERT's from 0.02
   to 0.58 on Gretel. GLiNER then leads on ai4privacy and Nemotron-PII, and BERT leads clearly on
   Gretel (0.787 against 0.595).
4. **Leak rate**: GLiNER leaks least on ai4privacy (6.1%), BERT least on Gretel (8.3%). On
   Nemotron-PII the difference between them is not established.
5. **F1 confirms the F2 orders.** It reverses none of them and separates the four pairs that F2
   leaves level, each time in favour of the more precise detector: GLiNER over BERT on Gretel (A)
   and ai4privacy (B), Presidio over BERT on Nemotron-PII (B and B-dates).
6. The pipeline passed every validity check (perfect score of the gold, exact accounting,
   identical outputs when run again). No established ranking changes when GLiNER's cut-off
   moves between 0.25 and 0.60.

---

## 1. The question

The proposal asks three questions. This report answers the first one only:

> **Q1.** On clean text, how do a rule-based system (Presidio, without any neural model), a
> fine-tuned BERT (Gravitee) and GLiNER compare at finding the six PII labels of the proposal?

Question 2 (the same detectors on noisy copies of the test sets) and question 3 (BERT after
fine-tuning on noisy text) reuse the data, code and scoring described here.

## 2. Data

### 2.1 Datasets and test sets

| Dataset | Split used | English records | Our test set | Median length | Kind of text |
|---|---|---|---|---|---|
| ai4privacy open-pii-masking-500k | validation (there is no test split) | 30,160 | 10,000 sentences | 102 characters | short generated sentences and chat lines |
| Nemotron-PII | test | 100,000 (50,000 `us` + 50,000 `intl` locale) | 2,000 documents | 745 characters | generated forms, invoices, e-mails, notes |
| Gretel synthetic PII finance | test, English records | 2,962 (of 5,594) | all 2,962 documents | 1,414 characters | generated financial and legal documents |

Each dataset is downloaded at a fixed revision. Each test set is a random sample drawn with
seed 0, so that everyone can run exactly the same experiments. The sizes were fixed after timing the detectors, so that the
whole plan (questions 1 to 3) fits on laptop CPUs.

Records without any of our labels are kept (1,493, 415 and 711 of them): they show whether a
detector masks text that is not PII.

### 2.2 Labels

Gold labels are mapped to the six labels of the proposal:

| Our label | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| NAME | GIVENNAME, SURNAME | first_name, last_name | name, first_name, last_name |
| EMAIL | EMAIL | email | email |
| PHONE | TELEPHONENUM | phone_number | phone_number |
| ADDRESS | BUILDINGNUM, STREET | street_address | street_address |
| DATE_OF_BIRTH | (none) | date_of_birth | date_of_birth |
| CARD_NUMBER | CREDITCARDNUMBER | credit_debit_card | credit_card_number |
| DATE (comparison C) | DATE | date, date_of_birth, date_time | date, date_of_birth, date_time |

All other labels (city, time, age, account numbers...) are not scored. ai4privacy has no
birth-date label, so DATE_OF_BIRTH is not scored on it. DATE counts dates only: times alone
(`TIME`, `time`) are left out, while a date followed by a time ("2025-07-13T17:43:03") counts,
as ai4privacy itself labels such values as dates.

Gold spans per label in our test sets:

| Label | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| NAME | 8,748 | 2,053 | 4,475 |
| EMAIL | 1,185 | 1,091 | 837 |
| PHONE | 1,567 | 464 | 531 |
| ADDRESS | 1,533 | 324 | 2,322 |
| DATE_OF_BIRTH | - | 363 | 143 |
| CARD_NUMBER | 185 | 257 | 76 |
| DATE (comparison C) | 1,387 | 2,126 | 5,768 |

### 2.3 Span clean-up

The same four rules are applied to the gold spans and to every detector's output, so both
sides are compared in the same way:

1. span edges move inward past whitespace, and past a pair of brackets or quotes that encloses
   the whole span (`[x@y.com]` becomes `x@y.com`, but `(555) 123-4567` keeps its parenthesis);
2. a leading title is removed from NAME spans (Mr, Mrs, Ms, Miss, Mx, Dr, Prof, Sir, Madam, and
   Mister, Madame, Master, Mayoress, Mstr, which ai4privacy labels as titles as often as Mr);
3. overlapping spans: the one that starts first is kept;
4. spans with the same label separated only by spaces are joined ("Minji" + "Park"); a comma or
   a line break keeps them apart.

The bracket rule and the five extra titles were added after looking at errors, because they
were annotation conventions rather than detection errors: in Gretel, e-mails are often
annotated with their Markdown brackets, and in ai4privacy, "Mister Maffongelli" against a gold
"Maffongelli" caused about 430 GLiNER and 370 BERT errors. Neither rule changes the test sets;
they only change how detections are compared.

## 3. Detectors

| | Presidio | BERT | GLiNER |
|---|---|---|---|
| Model | presidio-analyzer 2.2.366, pattern recognizers only | `gravitee-io/bert-small-pii-detection`, 28.5 M parameters | `urchade/gliner_multi_pii-v1`, about 289 M parameters |
| How it finds PII | regular expressions, a phone-number library, a checksum for cards | one tag per token (25 classes) | scores every span for label names given in plain words |
| Our labels | EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, DATE_TIME | PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME, CREDIT_CARD (19 other classes dropped) | "person", "email", "phone number", "street address", "date of birth" (or "date" in comparison C), "credit card number" |
| Settings | defaults, no NLP model loaded | defaults; reads lowercased text | cut-off 0.35 |
| Long texts | no limit | windows of 256 tokens overlapping by 64 | windows of 384 words overlapping by 64 |

- **Presidio, the baseline, without a neural model.** Presidio is our baseline: a purely
  rule-based and very fast system (see the computing time below), which shows how the two deep
  learning models differ. Presidio normally runs a spaCy model to find names and
  places. We give it an NLP engine that loads no model, so only self-contained rules run. It
  therefore cannot find names or addresses, by design.
- **Label correspondences.** Presidio's and BERT's DATE_TIME class covers any date or time; it
  is counted as DATE_OF_BIRTH in comparisons A and B, and as DATE in comparison C. BERT's
  LOCATION class also covers cities and countries, and is counted as ADDRESS.
- **Long texts.** Without windows, BERT would fail on texts longer than 512 tokens (about 6% of
  Nemotron-PII and Gretel documents), and GLiNER would silently ignore everything after its
  384th word (11% of Nemotron-PII and 25% of Gretel documents).
- **GLiNER's cut-off** was chosen once, before any test, as the proposal plans: the value with
  the best micro F2 on 300 sentences of the ai4privacy *train* split. F2 hardly moves between
  0.25 and 0.60 (0.859 to 0.861); 0.35 was the best.
- **Computing time** on a laptop CPU (8 cores), for the three test sets: Presidio 20 s, BERT
  6 min, GLiNER 1 h 20 (1 h 33 with the "date" label).

## 4. Scoring

- **Strict matching** (the proposal's choice). A detection is correct (true positive, TP) only if its start, end and
  label are exactly those of a gold span. Every other detection is a false positive (FP); every
  gold span without an exact match is a false negative (FN). A partial span therefore counts as
  one FP and one FN: finding "Minji" in "Minji Park" still sends "Park" to the language model.
  It is the usual rule in NER, which makes our scores comparable with other work. It is harsh
  on boundaries, however: masking "[x@y.com]" instead of "x@y.com" counts as an error although
  nothing leaks. The leak rate below and analysis S1 measure what this choice changes.
- **Scores**, computed per label and micro-averaged (TP, FP and FN summed over the labels first):

  ```
  P  = TP / (TP + FP)        R  = TP / (TP + FN)
  F1 = 2·P·R / (P + R)       F2 = 5·P·R / (4·P + R)
  ```

  As in the proposal, F1 and F2 are both main scores: F1 is the usual NER score, and F2 weighs
  recall twice as much as precision, because for a privacy filter a leak costs more than an
  extra mask.
- **Leak rate**: the share of gold spans that the detections, whatever their label, do not fully
  cover. These values would reach the language model in clear, at least in part.
- A score that does not exist (for example the precision of a detector that found nothing) is
  reported as "n/a", not as 0.

## 5. Choices made by the team

| When | Choice | Why |
|---|---|---|
| Nov 3 | Keep the three datasets of the proposal | different kinds of text (sentences, forms, financial documents) |
| Nov 3 | All benchmark runs on CPU-only laptops | the computers available to the team |
| Nov 3 | Presidio with rules only, without spaCy | it is the baseline: a purely rule-based and very fast system, which shows how the deep learning models differ; Presidio can run with an NLP engine that loads no model |
| Nov 3 | GLiNER's cut-off chosen on training data, as in the proposal | no tuning on the test sets |
| Nov 3 | Windows for long texts, sizes taken from each model's limits | otherwise texts are truncated, silently for GLiNER |
| Nov 3 | Undefined scores reported as n/a, not 0 | a missing score is not a bad score |
| Nov 3 | Bracket rule in the span clean-up | e-mails annotated with their Markdown brackets in Gretel |
| Nov 3 | Test set sizes 10,000 / 2,000 / 2,962, after timing the detectors | the whole plan must fit on laptop CPUs |
| Nov 4 | Five more titles in the span clean-up | ai4privacy's own title list |
| Nov 4 | Comparison B, without NAME and ADDRESS | Presidio's rules cannot find them |
| Nov 4 to 20 | Comparison C: every date as DATE, dates only, not times | Presidio and BERT have no birth-date class (agreed at a team meeting) |
| Nov 20 | Keep the leak rate next to the strict scores | it measures what reaches the language model |
| Nov 20 | Analysis plan written before any score of comparison C, hypotheses and smallest relevant difference (0.01 of F2) validated | so that comparison C is a test, not an observation after the fact |

The four comparisons used in the rest of the report:

| Comparison | Labels scored |
|---|---|
| **A** | the six labels of the proposal |
| **B** | EMAIL, PHONE, DATE_OF_BIRTH, CARD_NUMBER (without NAME and ADDRESS) |
| **A-dates** | as A, with DATE in place of DATE_OF_BIRTH |
| **B-dates** | as B, with DATE in place of DATE_OF_BIRTH |

---

## 6. The analysis plan, point by point

### 6.0 General rules

*This first part of the plan sets the rules shared by the whole analysis. It separates two kinds
of analyses: confirmatory analyses, whose question and decision rule are fixed before the
results are seen, and exploratory analyses, which look for what stands out in the results and
are labelled as such. It then sets three requirements: no score is given without its 95%
interval; no detector is called better than another without a paired test; every result can be recomputed with a command.*

The paragraphs below define these terms, in the order in which they appear in the rest of the
report.

**Confirmatory or exploratory.** An analysis is confirmatory when the question and the rule that
answers it are fixed before the results are seen: it supports conclusions. An exploratory
analysis looks in the results for what stands out: it gives explanations and leads, not proof.
The confirmatory part is the hypotheses of section 6.2 (five on A and B, from the proposal; four
on C) and the 72 paired tests of section 6.5: 4 comparisons × 3 datasets × 3 pairs of
detectors, on F2 and on the leak rate. The 36 F1 tests were added afterwards, to follow the proposal. The sensitivity analyses (6.6) are exploratory.

**Why statistics.** Each test set is a sample: with other documents from the same dataset, the
scores would be slightly different. The following tools measure this uncertainty.

- **95% interval (bootstrap).** The range of plausible values for the score the detector would
  have on a very large number of documents of the same kind. It is computed by simulating other
  test sets from ours: draw at random, with replacement, as many documents as the test set holds
  (some come out twice, others not at all), recompute the score, and repeat 10,000 times. The
  250th and the 9,750th of the 10,000 sorted scores bound the interval. A narrow interval means a
  stable score (ai4privacy, 10,000 documents); a wide one, a less certain score (Gretel). The
  same draws are used for the three detectors, which also gives the interval of the difference
  between two detectors.
- **Paired test (approximate randomization).** It tells whether the difference between two
  detectors could be due to chance. "Paired": both detectors are compared on the same
  documents, so the difficulty of each document cancels out. The test first assumes that the two
  detectors are interchangeable; then swapping their outputs on a document changes nothing on
  average. So their outputs are swapped document by document, on a coin toss, the difference is
  recomputed, and this is repeated 10,000 times. The p-value is the share of rounds where the
  simulated difference is at least as large as the observed one: the smaller it is, the less
  chance alone can explain the difference.
- **Holm's correction.** With 36 tests at the 5% level, about two would come out "significant"
  by chance even with no real difference. The correction raises the p-values accordingly: the
  smallest is multiplied by 36, the next by 35, and so on. With 10,000 rounds, the smallest
  possible corrected p-value is 0.0036.
- **Linked documents.** The two locales of a Nemotron-PII document and texts that appear twice
  are not independent: they are drawn and swapped together.

**Decision rule.** A detector is called better than another only if three conditions hold: the
corrected p-value is below 0.05, the 95% interval of the difference excludes 0, and the
difference is at least 0.01 (below that, it has no practical interest). Otherwise the
difference is "not established": the two detectors are not necessarily equal, but our data
cannot separate them.

All commands are in section 9.

### 6.1 What is fixed before the analysis

*Code, data versions, model versions, settings, metrics and statistical parameters.*

| Item | Value |
|---|---|
| Models | BERT revision f8c27a85, GLiNER revision 1fcf13e8, presidio-analyzer 2.2.366 |
| Main score | micro F2, strict matching (F1 added afterwards) |
| Statistics | 95% intervals; 10,000 bootstrap draws and 10,000 permutation rounds; seed 0; α = 0.05; Holm's correction; smallest relevant difference 0.01 |
| Code | committed at the end of question 1, with this report |

### 6.2 Hypotheses

*On A and B: taken from the expectations written in the proposal, before any score. On C:
written and validated before any score of comparison C.*

**On comparisons A and B**

| Hypothesis | Expectation in the proposal | Result |
|---|---|---|
| **H-AB1.** In A, Presidio has the lowest F2 and the highest leak rate of the three detectors on all three datasets. | Rules cannot find names and addresses, which have no fixed format. | **Holds.** F2 0.157 / 0.431 / 0.171 and leak 0.868 / 0.572 / 0.839 (ai4privacy / Nemotron-PII / Gretel); all its differences with the two models are established (6.5). |
| **H-AB2.** On fixed-format labels (comparison B), Presidio comes much closer to the models. | Rules cover fixed formats (e-mails, phones, cards, dates); what they miss is names and addresses. | **Holds.** The F2 gap to the best model falls from 0.71 / 0.39 / 0.55 in A to 0.28 / 0.20 / 0.17 in B. Formats differ: Presidio has the best F2 of the three on e-mails (0.990 / 0.988 / 0.960) and the lowest on phones (0.371 / 0.727 / 0.542). |
| **H-AB3.** BERT's precision is low on ADDRESS and DATE_OF_BIRTH. | Its LOCATION and DATE_TIME classes cover any place and any date, which is broader than our labels and lowers precision. | **Holds, except ADDRESS on Gretel.** BERT's precision on ADDRESS: 0.281 / 0.206 / 0.667; on DATE_OF_BIRTH: 0.111 on Nemotron-PII, 0.018 on Gretel (ai4privacy has no such label). These are its two least precise labels, except ADDRESS on Gretel, where its card numbers are less precise (0.460). GLiNER, asked for exactly our labels, is more precise on both labels everywhere (ADDRESS 0.541 / 0.655 / 0.755; DATE_OF_BIRTH 0.935 / 0.617). |
| **H-AB4.** Against GLiNER, BERT does better on Gretel than on the two other datasets. | BERT was trained partly on Gretel PII finance, so its Gretel score is not an out-of-domain result. | **Holds in A, not in B.** F2 difference BERT − GLiNER in A: −0.071 / −0.061 / +0.005; in B: −0.006 / −0.203 / −0.278. In B the date label dominates: with DATE (B-dates), BERT is ahead on Gretel (+0.277). S3 shows that this advantage does not come from test documents seen in training. |
| **H-AB5.** No expected direction between BERT and GLiNER on ai4privacy and Nemotron-PII. | The proposal does not choose between the two models. | GLiNER is better on both in A (F2 +0.071 and +0.061, established) and on Nemotron-PII in B (+0.203); on ai4privacy in B, level in F2, GLiNER ahead in F1. |

**On comparison C**

| Hypothesis | Result |
|---|---|
| **H-C1.** From A to A-dates, the precision of Presidio's and BERT's date label rises sharply on Nemotron-PII and Gretel. | **Holds.** Presidio: 0.224 → 0.962 (Nemotron-PII), 0.038 → 0.519 (Gretel). BERT: 0.111 → 0.471 and 0.018 → 0.583. |
| **H-C2.** On Gretel, Presidio's date recall is lower than BERT's (its rules only know numeric formats). | **Holds.** 0.273 against 0.758. |
| **H-C3.** No expected direction between GLiNER (asked for "date") and BERT on dates. | GLiNER is better on ai4privacy (F2 0.933 against 0.737) and Nemotron-PII (0.790 against 0.655), worse on Gretel (0.397 against 0.715); the intervals do not overlap. |
| **H-C4.** Presidio's and BERT's other labels are identical in A and A-dates; GLiNER's may move a little. | **Holds.** Identical counts for Presidio and BERT; GLiNER's other labels move by 0.023 of F2 at most (card numbers in Gretel, 76 gold spans). |

### 6.3 Validity checks

*Before reading any score, check that the measurement itself is right.*

Before reading the scores, we checked that the measurement itself was right: every test set has
a prediction for every example, the gold given as predictions gets a perfect score, the span
counts are exact, and a second run of the detectors gives identical outputs. All these checks
passed.

### 6.4 Scores with 95% intervals

*Each score with a 95% interval from a paired cluster bootstrap. Examples that are not
independent (the two locales of a Nemotron-PII document, identical texts) are resampled
together.*

The resampling unit is a group of linked examples: 14 Nemotron-PII documents have their twin in
the other locale in our sample, and 74 ai4privacy texts and 3 Gretel texts appear twice. The
same 10,000 draws are used for the three detectors of a test set.

Micro F1, micro F2 and leak rate, each with its 95% interval:

**Comparison A (six labels)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0.223 [0.214, 0.232] | 0.768 [0.761, 0.774] | **0.853** [0.847, 0.858] |
|  | F2 | 0.157 [0.150, 0.164] | 0.796 [0.790, 0.803] | **0.867** [0.862, 0.873] |
|  | Leak rate | 0.868 [0.862, 0.874] | 0.113 [0.108, 0.119] | **0.061** [0.056, 0.065] |
| Nemotron-PII | F1 | 0.453 [0.437, 0.468] | 0.628 [0.612, 0.643] | **0.763** [0.747, 0.777] |
|  | F2 | 0.431 [0.415, 0.446] | 0.762 [0.749, 0.774] | **0.823** [0.811, 0.835] |
|  | Leak rate | 0.572 [0.555, 0.588] | 0.088 [0.076, 0.100] | 0.101 [0.089, 0.113] |
| Gretel | F1 | 0.194 [0.180, 0.208] | 0.572 [0.560, 0.583] | **0.637** [0.622, 0.652] |
|  | F2 | 0.171 [0.159, 0.184] | 0.718 [0.707, 0.728] | 0.713 [0.701, 0.725] |
|  | Leak rate | 0.839 [0.826, 0.850] | **0.083** [0.072, 0.095] | 0.195 [0.183, 0.207] |

**Comparison B (without NAME and ADDRESS)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0.659 [0.644, 0.674] | 0.846 [0.834, 0.858] | **0.881** [0.872, 0.891] |
|  | F2 | 0.616 [0.600, 0.632] | 0.891 [0.881, 0.901] | 0.898 [0.888, 0.907] |
|  | Leak rate | 0.406 [0.389, 0.423] | **0.054** [0.046, 0.063] | 0.071 [0.062, 0.080] |
| Nemotron-PII | F1 | 0.632 [0.613, 0.650] | 0.568 [0.547, 0.589] | **0.935** [0.926, 0.944] |
|  | F2 | 0.757 [0.741, 0.773] | 0.757 [0.742, 0.772] | **0.960** [0.953, 0.967] |
|  | Leak rate | 0.103 [0.089, 0.118] | 0.024 [0.017, 0.032] | 0.015 [0.009, 0.021] |
| Gretel | F1 | 0.385 [0.359, 0.410] | 0.264 [0.247, 0.281] | **0.698** [0.677, 0.720] |
|  | F2 | 0.569 [0.543, 0.595] | 0.459 [0.437, 0.480] | **0.737** [0.716, 0.757] |
|  | Leak rate | 0.157 [0.137, 0.178] | **0.030** [0.020, 0.040] | 0.205 [0.182, 0.230] |

**Comparison A-dates (six labels, DATE in place of DATE_OF_BIRTH)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0.251 [0.242, 0.260] | 0.738 [0.732, 0.745] | **0.853** [0.848, 0.858] |
|  | F2 | 0.178 [0.172, 0.185] | 0.790 [0.783, 0.796] | **0.872** [0.867, 0.877] |
|  | Leak rate | 0.850 [0.844, 0.855] | 0.106 [0.101, 0.111] | **0.057** [0.053, 0.061] |
| Nemotron-PII | F1 | 0.610 [0.596, 0.625] | 0.713 [0.699, 0.727] | **0.762** [0.749, 0.775] |
|  | F2 | 0.532 [0.516, 0.548] | 0.778 [0.764, 0.791] | **0.801** [0.788, 0.813] |
|  | Leak rate | 0.501 [0.485, 0.517] | 0.141 [0.127, 0.156] | 0.137 [0.123, 0.152] |
| Gretel | F1 | 0.289 [0.273, 0.304] | **0.741** [0.732, 0.751] | 0.573 [0.560, 0.585] |
|  | F2 | 0.227 [0.214, 0.240] | **0.787** [0.777, 0.796] | 0.595 [0.583, 0.606] |
|  | Leak rate | 0.796 [0.784, 0.808] | **0.090** [0.082, 0.099] | 0.355 [0.343, 0.367] |

**Comparison B-dates (without NAME and ADDRESS, DATE in place of DATE_OF_BIRTH)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0.615 [0.602, 0.629] | 0.726 [0.716, 0.737] | **0.887** [0.879, 0.894] |
|  | F2 | 0.544 [0.529, 0.558] | 0.835 [0.826, 0.843] | **0.909** [0.902, 0.916] |
|  | Leak rate | 0.493 [0.478, 0.507] | 0.049 [0.043, 0.056] | 0.057 [0.050, 0.064] |
| Nemotron-PII | F1 | 0.797 [0.783, 0.810] | 0.716 [0.697, 0.734] | **0.873** [0.859, 0.886] |
|  | F2 | 0.791 [0.774, 0.807] | 0.784 [0.766, 0.801] | **0.868** [0.850, 0.885] |
|  | Leak rate | 0.200 [0.182, 0.219] | 0.139 [0.121, 0.159] | **0.110** [0.092, 0.130] |
| Gretel | F1 | 0.445 [0.424, 0.466] | **0.690** [0.675, 0.704] | 0.502 [0.485, 0.519] |
|  | F2 | 0.408 [0.387, 0.429] | **0.744** [0.730, 0.758] | 0.467 [0.450, 0.484] |
|  | Leak rate | 0.606 [0.585, 0.628] | **0.084** [0.076, 0.094] | 0.519 [0.501, 0.536] |

In bold: a value better than both others with an established difference (section 6.5). F2 per
label is in appendix A. Labels with fewer than 100 gold spans (card numbers in Gretel, 76) give
indicative scores only.

### 6.5 Paired comparisons

*For each pair of detectors on the same test set: the difference of micro F1, of
micro F2 and of leak rate, its 95% interval, a two-sided paired approximate randomization test (10,000 rounds),
Holm's correction over the 36 tests of each family. A detector is called better only if the
corrected p-value is below 0.05, the interval of the difference excludes 0, and the difference
is at least 0.01.*

The randomization test swaps the outputs of the two detectors group by group with probability
1/2 and counts how often the difference is at least as large as the one observed; with 10,000
rounds, the smallest possible p-value is about 0.0001, which Holm's correction turns into
0.0036 for the most significant tests.

**Micro F2: 32 of the 36 differences are established.** The four that are not:

| Comparison | Dataset | Pair | Difference [95% interval] | Corrected p |
|---|---|---|---|---|
| A | Gretel | BERT − GLiNER | 0.005 [−0.008, 0.018] | 1.00 |
| B | ai4privacy | BERT − GLiNER | −0.006 [−0.017, 0.005] | 1.00 |
| B | Nemotron-PII | Presidio − BERT | 0.000 [−0.017, 0.017] | 1.00 |
| B-dates | Nemotron-PII | Presidio − BERT | 0.007 [−0.015, 0.028] | 1.00 |

In these four cases, the difference is below 0.01, its interval contains 0 and the corrected
p-value is 1: we cannot say which of the two detectors is better. This does not mean they are
equal, but the interval bounds the possible difference: it very likely stays below 0.03 in
absolute value. The two detectors are therefore close.

The resulting order on each test set ("=" means no established difference):

| Comparison | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| A | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | BERT = GLiNER > Presidio |
| B | GLiNER = BERT > Presidio | GLiNER > Presidio = BERT | GLiNER > Presidio > BERT |
| A-dates | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | BERT > GLiNER > Presidio |
| B-dates | GLiNER > BERT > Presidio | GLiNER > Presidio = BERT | BERT > GLiNER > Presidio |

The smallest established F2 difference is GLiNER against BERT on Nemotron-PII in A-dates: 0.023
[0.009, 0.037], corrected p = 0.010.

**Leak rate: 31 of the 36 differences are established.** The five that are not:

| Comparison | Dataset | Pair | Difference [95% interval] | Corrected p |
|---|---|---|---|---|
| A | Nemotron-PII | BERT − GLiNER | −0.014 [−0.027, −0.000] | 0.16 |
| B | Nemotron-PII | BERT − GLiNER | 0.010 [0.002, 0.018] | 0.09 |
| B | Gretel | Presidio − GLiNER | −0.049 [−0.084, −0.014] | 0.05 |
| A-dates | Nemotron-PII | BERT − GLiNER | 0.005 [−0.011, 0.020] | 0.58 |
| B-dates | ai4privacy | BERT − GLiNER | −0.008 [−0.017, 0.001] | 0.17 |

Elsewhere, Presidio always leaks more than both models. Between the two models, GLiNER leaks less
on ai4privacy in A and A-dates and on Nemotron-PII in B-dates; BERT leaks less on Gretel in all
four comparisons and on ai4privacy in B.

The resulting order for the leak rate (">" means "leaks less", "=" no established
difference):

| Comparison | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| A | GLiNER > BERT > Presidio | BERT = GLiNER > Presidio | BERT > GLiNER > Presidio |
| B | BERT > GLiNER > Presidio | GLiNER = BERT > Presidio | BERT > Presidio = GLiNER |
| A-dates | GLiNER > BERT > Presidio | GLiNER = BERT > Presidio | BERT > GLiNER > Presidio |
| B-dates | BERT = GLiNER > Presidio | GLiNER > BERT > Presidio | BERT > GLiNER > Presidio |

A better F2 does not guarantee fewer leaks: in A on Nemotron-PII, GLiNER has the best F2 but does
not leak less than BERT; in B on ai4privacy, the two are level in F2 but BERT leaks less. On
Gretel, BERT leaks least in all four comparisons, including B, where GLiNER has the best F2.

**Micro F1: all 36 differences are established.** F1 reverses none of the F2 orders. It
separates the four pairs that F2 leaves level, each time in favour of the more precise detector,
which F2 offsets by the higher recall of the other one:

| Comparison | Dataset | Pair | F1 difference [95% interval] | Precision | Recall |
|---|---|---|---|---|---|
| A | Gretel | BERT − GLiNER | −0.065 [−0.080, −0.051] | 0.427 / 0.541 | 0.865 / 0.774 |
| B | ai4privacy | BERT − GLiNER | −0.035 [−0.048, −0.023] | 0.780 / 0.855 | 0.924 / 0.909 |
| B | Nemotron-PII | Presidio − BERT | 0.063 [0.045, 0.083] | 0.495 / 0.401 | 0.873 / 0.973 |
| B-dates | Nemotron-PII | Presidio − BERT | 0.081 [0.060, 0.102] | 0.807 / 0.625 | 0.787 / 0.837 |

Precision and recall are given in the order of the pair. The corrected p-value is 0.0036 for all
four.

The resulting order in F1 (every difference is established):

| Comparison | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| A | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio |
| B | GLiNER > BERT > Presidio | GLiNER > Presidio > BERT | GLiNER > Presidio > BERT |
| A-dates | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | BERT > GLiNER > Presidio |
| B-dates | GLiNER > BERT > Presidio | GLiNER > Presidio > BERT | BERT > GLiNER > Presidio |

It is the same as the F2 order, except at the four F2 ties.

**Across datasets** (no average is taken, since the datasets differ in labels, sizes and text):

- GLiNER against BERT, F2: GLiNER is better on 2 of 3 datasets in every comparison; on the
  third, the difference is not established (A on Gretel, B on ai4privacy) or BERT is better
  (A-dates and B-dates on Gretel).
- Presidio is worse than both models on 3 of 3 datasets in A and A-dates. In B, it is level with
  BERT on Nemotron-PII and better than BERT on Gretel, an effect of the date label (section 6.2).

### 6.6 Sensitivity analyses (exploratory)

*Do the conclusions hold when a choice of method changes?*

A sensitivity analysis redoes part of the comparison while changing a single choice of method,
to see whether the conclusions depend on it. If the order of the detectors stays the same, the
conclusion is robust; if it changes, the conclusion depends on that choice and must be stated
with this reservation. These analyses are exploratory: no hypothesis was written for them in
advance, and their differences are not tested. Four choices were examined:

| Analysis | Choice of method examined | Question |
|---|---|---|
| S1 | strict matching | Does the ranking change if approximate boundaries or labels are accepted? |
| S2 | GLiNER's cut-off (0.35) | Would another reasonable cut-off change the ranking? |
| S3 | Gretel as a test set | Did BERT see our test documents during its training? |
| S4 | windows for long texts | Do the detectors do worse on long documents? |

**S1. Looser matching rules.**

*Why.* Strict matching counts as an error any detection whose boundaries or label differ from
the gold, even when it covers the PII or nearly all of it (section 4). A detector that often ends
its spans one character too early or too late is then penalized as much as one that misses the
PII. We check that the ranking does not come only from this strictness.

*Method.* F2 is recomputed under three looser rules, besides the strict one:

- same boundaries, label ignored: a detection is correct if it has exactly the boundaries of a
  gold span, whatever its label;
- overlap, same label: the detection only needs to share at least one character with a gold
  span of the same label;
- overlap, label ignored: one shared character is enough, whatever the label.

Recall is the share of gold spans hit by at least one detection, precision the share of
detections that hit a gold span. The table gives BERT and GLiNER in A, and on Nemotron-PII and
Gretel in A-dates; Presidio and the other cases are in the analysis outputs.

*Results.* Micro F2 under the four rules:

| Comparison, dataset | Detector | Strict | Same span, any label | Overlap, same label | Overlap, any label |
|---|---|---|---|---|---|
| A, ai4privacy | BERT / GLiNER | 0.796 / 0.867 | 0.804 / 0.870 | 0.891 / 0.926 | 0.905 / 0.930 |
| A, Nemotron-PII | BERT / GLiNER | 0.762 / 0.823 | 0.762 / 0.823 | 0.804 / 0.851 | 0.804 / 0.853 |
| A, Gretel | BERT / GLiNER | 0.718 / 0.713 | 0.718 / 0.715 | 0.780 / 0.740 | 0.786 / 0.754 |
| A-dates, Nemotron-PII | BERT / GLiNER | 0.778 / 0.801 | 0.778 / 0.801 | 0.852 / 0.833 | 0.853 / 0.835 |
| A-dates, Gretel | BERT / GLiNER | 0.787 / 0.595 | 0.787 / 0.597 | 0.908 / 0.652 | 0.913 / 0.661 |

Ignoring the label (same span, any label) changes F2 by at most 0.011: confusions between our
labels are rare. Accepting overlaps raises BERT more than GLiNER, because many of BERT's errors
are boundaries. The order changes in two places: on Gretel in A (level under strict matching,
BERT ahead with overlaps) and on Nemotron-PII in A-dates (GLiNER ahead under strict matching,
BERT ahead with overlaps). Presidio stays far behind under every rule.

**S2. GLiNER's cut-off.**

*Why.* GLiNER gives each candidate span a confidence score and keeps only those above a cut-off.
A low cut-off finds more PII (higher recall) but adds false alarms (lower precision); a high
cut-off does the opposite. The cut-off of 0.35 was chosen on training data (section 3); BERT has
no cut-off. If the ranking held only for this exact value, it would be fragile.

*Method.* GLiNER was run once at cut-off 0.05; keeping the spans scored above a given cut-off
gives its predictions at that cut-off. F2 is computed this way for every cut-off from 0.05 to
0.95 in steps of 0.05 and compared with BERT's. The plan asked to check that the ranking holds
between 0.25 and 0.60, the range where F2 hardly moved on the training data.

*Results.* Micro F2 on the clean test sets (six cut-offs shown; the full curve is in the analysis
outputs):

| Comparison, dataset | BERT | GLiNER 0.15 | 0.25 | 0.35 | 0.45 | 0.60 | 0.75 | GLiNER's best F2 (cut-off) |
|---|---|---|---|---|---|---|---|---|
| A, ai4privacy | 0.796 | 0.862 | 0.866 | 0.867 | 0.867 | 0.861 | 0.849 | 0.867 (0.30 to 0.45) |
| A, Nemotron-PII | 0.762 | 0.804 | 0.817 | 0.823 | 0.824 | 0.826 | 0.817 | 0.826 (0.60) |
| A, Gretel | 0.718 | 0.714 | 0.719 | 0.713 | 0.702 | 0.675 | 0.638 | 0.719 (0.20 and 0.25) |
| B, ai4privacy | 0.891 | 0.896 | 0.897 | 0.898 | 0.894 | 0.886 | 0.876 | 0.899 (0.20) |
| B, Nemotron-PII | 0.757 | 0.952 | 0.957 | 0.960 | 0.962 | 0.962 | 0.960 | 0.962 (0.45 to 0.70) |
| B, Gretel | 0.459 | 0.733 | 0.739 | 0.736 | 0.724 | 0.698 | 0.665 | 0.739 (0.25) |

At 0.35, these filtered predictions give the scores of the benchmark run to within 0.001:
13 documents out of 14,962 differ, because the stored scores are rounded to three decimals and
because, where two windows overlap, the longer span is kept even when its score is lower.

- Where the difference between GLiNER and BERT was established (A on ai4privacy and
  Nemotron-PII, B on Nemotron-PII and Gretel), GLiNER stays ahead for every cut-off from 0.10
  to 0.90.
- Where it was not (A on Gretel, B on ai4privacy), the two stay within 0.01 of each other
  between 0.25 and 0.35. On ai4privacy (B) this still holds up to 0.60. On Gretel (A), GLiNER's
  F2 falls as the cut-off rises, and BERT is ahead by more than 0.01 from 0.45 (0.702 against
  0.718). Differences at cut-offs other than 0.35 are not tested.
- Even the best cut-off for each test set, which would amount to tuning on the test, would
  raise GLiNER's F2 by at most 0.006 (A, Gretel). BERT has no cut-off and Presidio keeps its defaults:
  only GLiNER received any tuning, and it weighs little.

Answer to the plan's question: between 0.25 and 0.60, no established ranking changes; only the
tie on Gretel (A) turns in BERT's favour at the higher cut-offs.

**S3. Did BERT see our Gretel documents?**

*Why.* The training corpus of Gravitee's BERT contains 25,948 rows from Gretel PII finance, and
its card does not say which split they come from. If some of our test documents are among them,
BERT has seen the answers, and its Gretel score is inflated.

*Method.* Each of our 2,962 Gretel test documents is compared with the Gretel rows of the training
corpus. We first look for identical texts, then measure the share of the document's word 13-grams
(sequences of 13 consecutive words, lowercased) that also occur in the corpus: such a sequence
rarely repeats by chance, so a high share points to copied text. A document is called "seen" if
it is identical to a training row or shares at least half of its 13-grams with them. F2 is then
recomputed without the seen documents.

*Results.* 8 documents are identical to a training row, 91 share at least half of their 13-grams
with the training rows, 5 share all of them; the median share is 0.05. Without these 95
documents, BERT's F2 on Gretel stays 0.718 (GLiNER: 0.717). BERT did not learn our test
documents, but it was trained on documents from the same generator, with the same annotation
conventions; this likely helps it on Gretel.

**S4. Document length.**

*Why.* The models have a maximum length: GLiNER reads 384 words at a time, BERT 512 tokens.
Longer texts are read in overlapping windows (section 3). The windows prevent truncation, but a
model may still do worse on a long document, for instance because a window cuts the context.

*Method.* Documents of at most 384 words are separated from longer ones, and micro F2
(comparison A) is computed on each group. ai4privacy has only short texts.

*Results.* Micro F2 on short / long documents:

| Dataset | Documents short / long | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | 10,000 / 0 | 0.157 / n/a | 0.796 / n/a | 0.867 / n/a |
| Nemotron-PII | 1,769 / 231 | 0.417 / 0.530 | 0.784 / 0.622 | 0.828 / 0.785 |
| Gretel | 2,199 / 763 | 0.181 / 0.141 | 0.734 / 0.672 | 0.744 / 0.633 |

Both models do worse on long documents, which are also more structured (tables, forms, code);
the windows prevent truncation but do not remove this drop.

## 7. Conclusions for question 1

**Measured and established** (95% intervals, paired tests, Holm's correction):

1. On the labels of the proposal (A), GLiNER finds PII best on ai4privacy and Nemotron-PII, and
   leaks least on ai4privacy; on Gretel, BERT and GLiNER have the same F2 (GLiNER is
   ahead in F1) and BERT leaks less.
   The established orders hold for any GLiNER cut-off between 0.25 and 0.60 (S2).
2. Rules alone (Presidio) leak 57% to 87% of the PII, because they do not cover names and
   addresses. On e-mails, they have the best F2 of the three (0.96 to 0.99).
3. With every date counted (A-dates), GLiNER stays ahead on ai4privacy and Nemotron-PII and BERT
   is clearly ahead on Gretel (F2 0.787 against 0.595).

**Overall ranking** (count of the differences established in 6.5, over the 12 cases: 4
comparisons × 3 datasets; for the leak rate, winning means leaking less):

| Head-to-head (cases won – cases lost) | F2 | F1 | Leak rate |
|---|---|---|---|
| GLiNER against BERT | 8 – 2 (2 ties) | 10 – 2 | 3 – 5 (4 ties) |
| BERT against Presidio | 9 – 1 (2 ties) | 9 – 3 | 12 – 0 |
| GLiNER against Presidio | 12 – 0 | 12 – 0 | 11 – 0 (1 tie) |

For detection quality, the ranking is GLiNER, then BERT, then Presidio. BERT is ahead of GLiNER
only on Gretel when every date counts, and Presidio is ahead of BERT only on fixed-format labels.
For leaks, Presidio is last everywhere, but BERT and GLiNER cannot be separated. These 12 cases
are not independent, since the same datasets come back in the four comparisons: this count
summarizes the tests of section 6.5; it is not a new test.

**Explained** (by comparison C, section 6.2):

4. The apparent strength of Presidio against BERT in comparison B comes from mapping a generic
   date class to "date of birth".

**Supposed** (not tested):

5. BERT's advantage on Gretel probably comes from its training data, which includes documents
   from the same generator with the same conventions (it did not see our test documents, S3).

## 8. Limitations

- All three datasets are synthetic; real prompts may differ in style and in PII formats.
- English only.
- Label correspondences for Presidio and BERT are approximate (dates, locations); comparison C
  addresses dates only.
- GLiNER is the only detector with a tuned setting; even tuning it on the test sets would add
  at most 0.006 of F2 (S2).
- Strict matching penalizes boundary conventions heavily; S1 shows how much.

## 9. Reproducing the results

```bash
uv sync                                               # environment
uv run python experiments/download_data.py            # datasets, fixed revisions
uv run python experiments/prepare_data.py             # test sets, seed 0 (clean and dates versions)
uv run python experiments/tune_gliner.py              # GLiNER cut-off on training data (0.35)
uv run python experiments/run_benchmark.py presidio   # also bert, gliner; add --dates for comparison C
uv run python experiments/score.py                    # results/scores.md
uv run python experiments/significance.py             # intervals and tests (6.4, 6.5)
uv run python experiments/sensitivity.py              # S1, S2, S4
uv run python experiments/contamination.py            # S3
```

Result files: `results/scores.md`, `results/without_name_address/scores.md`, and in
`results/analysis/`: `significance.md` (with all the tests and the per-label intervals), `sensitivity.md`, `contamination.md`.

## Appendix A. F2 per label [95% interval]

**Comparison A**

| Dataset | Detector | NAME | EMAIL | PHONE | ADDRESS | DATE_OF_BIRTH | CARD_NUMBER |
|---|---|---|---|---|---|---|---|
| ai4privacy | Presidio | 0.000 | 0.990 [0.984, 0.995] | 0.371 [0.347, 0.395] | 0.000 | - | 0.073 [0.035, 0.116] |
| ai4privacy | BERT | 0.832 [0.824, 0.840] | 0.986 [0.978, 0.992] | 0.840 [0.824, 0.855] | 0.473 [0.451, 0.495] | - | 0.762 [0.708, 0.810] |
| ai4privacy | GLiNER | 0.923 [0.918, 0.928] | 0.970 [0.963, 0.977] | 0.892 [0.879, 0.904] | 0.475 [0.448, 0.501] | - | 0.432 [0.363, 0.502] |
| Nemotron-PII | Presidio | 0.000 | 0.988 [0.980, 0.994] | 0.727 [0.697, 0.756] | 0.000 | 0.590 [0.555, 0.624] | 0.141 [0.093, 0.192] |
| Nemotron-PII | BERT | 0.839 [0.819, 0.857] | 0.954 [0.939, 0.968] | 0.934 [0.919, 0.948] | 0.447 [0.403, 0.491] | 0.383 [0.351, 0.415] | 0.949 [0.932, 0.963] |
| Nemotron-PII | GLiNER | 0.712 [0.690, 0.734] | 0.970 [0.960, 0.980] | 0.931 [0.917, 0.944] | 0.664 [0.608, 0.717] | 0.978 [0.965, 0.988] | 0.947 [0.920, 0.972] |
| Gretel | Presidio | 0.000 | 0.960 [0.945, 0.973] | 0.542 [0.501, 0.582] | 0.000 | 0.158 [0.133, 0.186] | 0.409 [0.270, 0.543] |
| Gretel | BERT | 0.850 [0.833, 0.867] | 0.845 [0.820, 0.870] | 0.910 [0.888, 0.930] | 0.801 [0.782, 0.819] | 0.085 [0.072, 0.098] | 0.714 [0.613, 0.807] |
| Gretel | GLiNER | 0.744 [0.728, 0.760] | 0.617 [0.585, 0.649] | 0.897 [0.875, 0.917] | 0.621 [0.597, 0.645] | 0.876 [0.835, 0.911] | 0.579 [0.477, 0.668] |

**DATE in comparison A-dates** (the other labels are within 0.023 of comparison A)

| Dataset | Presidio | BERT | GLiNER |
|---|---|---|---|
| ai4privacy | 0.374 [0.349, 0.400] | 0.737 [0.722, 0.751] | 0.933 [0.923, 0.943] |
| Nemotron-PII | 0.771 [0.744, 0.797] | 0.655 [0.627, 0.682] | 0.790 [0.759, 0.818] |
| Gretel | 0.302 [0.277, 0.325] | 0.715 [0.699, 0.732] | 0.397 [0.377, 0.417] |

## Methods used

Strict span matching follows the usual named-entity evaluation (MUC; CoNLL-2003,
Tjong Kim Sang and De Meulder, 2003). Confidence
intervals use the percentile bootstrap (Efron and Tibshirani, 1993), resampled by groups of
linked examples (Field and Welsh, 2007); differences are tested with paired approximate
randomization (Noreen, 1989; Yeh, 2000), corrected with Holm's method (Holm, 1979); the 13-gram overlap follows the
decontamination check of Brown et al. (2020).
