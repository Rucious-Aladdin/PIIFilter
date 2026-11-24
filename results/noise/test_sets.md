# Noisy test sets: the edits

From `data/test/<dataset>_noisy_<kind>.meta.json` (experiments/make_noisy.py, seed 1, round(0.10 x words) edits per example, at least one). The `_dates` versions have the same edits.

| Dataset | Examples | Copy | Examples changed | Edits | Edits per example | Inside a PII value |
|---|---:|---|---:|---:|---:|---:|
| ai4privacy | 10,000 | typo | 9,922 | 17,218 | 1.7 | 16.0% |
| ai4privacy | 10,000 | space | 9,923 | 17,219 | 1.7 | 10.8% |
| ai4privacy | 10,000 | case | 9,874 | 16,938 | 1.7 | 38.4% |
| ai4privacy | 10,000 | punct | 9,381 | 15,414 | 1.5 | 0.0% |
| ai4privacy | 10,000 | number | 2,487 | 2,609 | 0.3 | 58.5% |
| ai4privacy | 10,000 | mixed | 9,996 | 17,292 | 1.7 | 18.1% |
| nemotron | 2,000 | typo | 2,000 | 26,445 | 13.2 | 3.5% |
| nemotron | 2,000 | space | 2,000 | 26,445 | 13.2 | 2.4% |
| nemotron | 2,000 | case | 2,000 | 26,097 | 13.0 | 6.1% |
| nemotron | 2,000 | punct | 1,891 | 24,009 | 12.0 | 0.1% |
| nemotron | 2,000 | number | 1,711 | 4,674 | 2.3 | 23.3% |
| nemotron | 2,000 | mixed | 2,000 | 26,445 | 13.2 | 5.2% |
| gretel | 2,962 | typo | 2,961 | 56,550 | 19.1 | 4.5% |
| gretel | 2,962 | space | 2,956 | 56,574 | 19.1 | 3.6% |
| gretel | 2,962 | case | 2,960 | 56,544 | 19.1 | 9.4% |
| gretel | 2,962 | punct | 2,680 | 53,330 | 18.0 | 4.3% |
| gretel | 2,962 | number | 1,992 | 8,297 | 2.8 | 8.6% |
| gretel | 2,962 | mixed | 2,961 | 56,599 | 19.1 | 5.8% |

Edits of each kind in the mixed copies:

| Dataset | typo | space | case | punct | number |
|---|---:|---:|---:|---:|---:|
| ai4privacy | 4,158 | 4,271 | 4,088 | 3,964 | 811 |
| nemotron | 5,941 | 6,086 | 5,884 | 5,749 | 2,785 |
| gretel | 13,172 | 13,207 | 13,210 | 12,947 | 4,063 |

Every noisy test set has the same gold spans per label as its clean version (checked here).
