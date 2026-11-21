# Time and memory on CPU

x86_64, 8 cores. First 100 examples of each clean test set; each detector in a fresh process. The peak RAM includes Python and the libraries.

| Detector | Parameters | Threads | Peak RAM (MB) | ai4privacy: mean / median (ms) | Nemotron-PII: mean / median (ms) | Gretel: mean / median (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Presidio | none (rules) | 1 | 734 | 2.3 / 0.4 | 2.0 / 1.4 | 3.9 / 2.8 |
| BERT | 28.5 M | 4 | 829 | 11.5 / 11.4 | 42.8 / 27.0 | 68.1 / 69.4 |
| GLiNER | 288.9 M | 4 | 3110 | 161.4 / 170.7 | 557.8 / 337.6 | 1093.6 / 901.8 |

Presidio loads no model: almost all of its RAM comes from importing presidio-analyzer, which imports spaCy and torch.
