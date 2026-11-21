# Did BERT see the Gretel test documents?

Training corpus: gravitee-io/pii-detection-dataset at c90469d2e603, 175881 rows, 25948 from Gretel PII finance.
Gretel test documents: 2962.

| Measure | Documents |
|---|---|
| identical text in the training rows | 8 |
| at least 50% of their 13-grams in the training rows | 91 |
| at least 80% of their 13-grams in the training rows | 6 |
| at least 100% of their 13-grams in the training rows | 5 |
| median share of 13-grams in the training rows | 0.050 |

Micro F2 on Gretel (comparison A), all documents and documents not seen in training (no identical text, under 50% of 13-grams):

| Detector | All | Not seen | Documents not seen |
|---|---|---|---|
| bert | 0.718 | 0.718 | 2867 |
| gliner | 0.713 | 0.717 | 2867 |
