# Main table, comparison A (the six labels)

| Dataset | Detector | P | R | F1 | F2 | Leak |
|---|---|---:|---:|---:|---:|---:|
| ai4privacy | Presidio | 0.747 | 0.131 | 0.223 | 0.157 | 0.868 |
|  | BERT | 0.724 | 0.817 | 0.768 | 0.796 | 0.113 |
|  | GLiNER | 0.830 | 0.877 | **0.853** | **0.867** | **0.061** |
| Nemotron-PII | Presidio | 0.495 | 0.417 | 0.453 | 0.431 | 0.572 |
|  | BERT | 0.485 | 0.889 | 0.628 | 0.762 | 0.088 |
|  | GLiNER | 0.679 | 0.869 | **0.763** | **0.823** | 0.101 |
| Gretel | Presidio | 0.250 | 0.159 | 0.194 | 0.171 | 0.839 |
|  | BERT | 0.427 | 0.865 | 0.572 | 0.718 | **0.083** |
|  | GLiNER | 0.541 | 0.774 | **0.637** | 0.713 | 0.195 |

Bold: established as better than both other detectors (paired tests, Holm's correction). P and R were not tested.
