# Over-masking, comparison A

Share of the non-space characters outside every gold span, of any label of the dataset, that the detections mask (after the span clean-up).

| Dataset | Presidio | BERT | GLiNER |
|---|---:|---:|---:|
| ai4privacy | 0.00% | 1.00% | 1.20% |
| Nemotron-PII | 0.03% | 0.66% | 0.69% |
| Gretel | 0.68% | 1.03% | 1.60% |

Gretel contains PII left unannotated, so its values are upper bounds: some of the text counted as masked by mistake is real PII.
