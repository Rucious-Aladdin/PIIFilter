# Questions 2 and 3: noisy test sets, comparison A-dates

Micro scores with strict matching. Noisy copies: about one edit per ten words (benchmark/noise.py, experiments/make_noisy.py).

## Question 2: F2 of each detector

### F2, ai4privacy

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.178 | 0.171 (-0.007) | 0.157 (-0.021) | 0.178 (+0.000) | 0.178 (-0.000) | 0.145 (-0.034) | 0.161 (-0.017) |
| bert | 0.790 | 0.708 (-0.082) | 0.680 (-0.110) | 0.790 (+0.000) | 0.774 (-0.016) | 0.768 (-0.022) | 0.731 (-0.059) |
| gliner | 0.872 | 0.837 (-0.035) | 0.797 (-0.075) | 0.846 (-0.026) | 0.861 (-0.011) | 0.862 (-0.010) | 0.832 (-0.040) |

### F2, nemotron

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.532 | 0.522 (-0.010) | 0.483 (-0.050) | 0.532 (+0.000) | 0.532 (+0.000) | 0.292 (-0.241) | 0.365 (-0.168) |
| bert | 0.778 | 0.735 (-0.042) | 0.710 (-0.067) | 0.778 (+0.000) | 0.776 (-0.001) | 0.712 (-0.066) | 0.705 (-0.072) |
| gliner | 0.801 | 0.797 (-0.004) | 0.768 (-0.033) | 0.792 (-0.008) | 0.796 (-0.004) | 0.766 (-0.035) | 0.767 (-0.033) |

### F2, gretel

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.227 | 0.222 (-0.006) | 0.215 (-0.013) | 0.227 (+0.000) | 0.228 (+0.000) | 0.146 (-0.081) | 0.177 (-0.050) |
| bert | 0.787 | 0.711 (-0.075) | 0.710 (-0.077) | 0.787 (+0.000) | 0.771 (-0.016) | 0.764 (-0.022) | 0.736 (-0.051) |
| gliner | 0.595 | 0.585 (-0.010) | 0.580 (-0.015) | 0.597 (+0.002) | 0.583 (-0.012) | 0.582 (-0.013) | 0.579 (-0.016) |

### Leak rate, ai4privacy

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.850 | 0.856 (+0.006) | 0.864 (+0.014) | 0.850 (+0.000) | 0.850 (-0.000) | 0.871 (+0.021) | 0.861 (+0.011) |
| bert | 0.106 | 0.124 (+0.018) | 0.140 (+0.033) | 0.106 (+0.000) | 0.106 (-0.001) | 0.124 (+0.018) | 0.124 (+0.018) |
| gliner | 0.057 | 0.066 (+0.009) | 0.102 (+0.045) | 0.084 (+0.027) | 0.067 (+0.010) | 0.065 (+0.009) | 0.082 (+0.025) |

### Leak rate, nemotron

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.501 | 0.512 (+0.010) | 0.543 (+0.042) | 0.501 (+0.000) | 0.501 (-0.000) | 0.703 (+0.202) | 0.643 (+0.142) |
| bert | 0.141 | 0.149 (+0.007) | 0.163 (+0.021) | 0.141 (+0.000) | 0.133 (-0.009) | 0.221 (+0.079) | 0.200 (+0.059) |
| gliner | 0.137 | 0.139 (+0.002) | 0.159 (+0.022) | 0.143 (+0.006) | 0.136 (-0.001) | 0.176 (+0.040) | 0.171 (+0.035) |

### Leak rate, gretel

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.796 | 0.802 (+0.005) | 0.806 (+0.010) | 0.796 (+0.000) | 0.796 (-0.000) | 0.858 (+0.062) | 0.836 (+0.040) |
| bert | 0.090 | 0.133 (+0.043) | 0.137 (+0.047) | 0.090 (+0.000) | 0.096 (+0.006) | 0.119 (+0.029) | 0.124 (+0.034) |
| gliner | 0.355 | 0.370 (+0.015) | 0.376 (+0.022) | 0.354 (-0.000) | 0.371 (+0.016) | 0.372 (+0.017) | 0.378 (+0.024) |

In brackets: the change from the clean test set.

## Question 3: fine-tuned BERT (F2, mean ± sd over seeds)

- BERT, fine-tuned on clean: bert-clean-s0, bert-clean-s1, bert-clean-s2
- BERT, fine-tuned on clean + noisy: bert-noisy-s0, bert-noisy-s1, bert-noisy-s2

### ai4privacy

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.790 | 0.708 | 0.680 | 0.790 | 0.774 | 0.768 | 0.731 |
| BERT, fine-tuned on clean | 0.987 ± 0.001 | 0.881 ± 0.001 | 0.857 ± 0.001 | 0.987 ± 0.001 | 0.980 ± 0.000 | 0.950 ± 0.002 | 0.914 ± 0.000 |
| BERT, fine-tuned on clean + noisy | 0.985 ± 0.000 | 0.956 ± 0.000 | 0.945 ± 0.001 | 0.985 ± 0.000 | 0.978 ± 0.001 | 0.982 ± 0.000 | 0.965 ± 0.001 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.106 | +0.074 | 70% |
| space | 0.130 | +0.088 | 68% |
| case | 0.000 | -0.002 | n/a (no loss) |
| punct | 0.007 | -0.003 | -39% |
| number | 0.037 | +0.032 | 86% |
| mixed | 0.073 | +0.051 | 70% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): +0.002

### nemotron

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.778 | 0.735 | 0.710 | 0.778 | 0.776 | 0.712 | 0.705 |
| BERT, fine-tuned on clean | 0.743 ± 0.019 | 0.644 ± 0.012 | 0.645 ± 0.014 | 0.743 ± 0.019 | 0.718 ± 0.024 | 0.604 ± 0.014 | 0.597 ± 0.013 |
| BERT, fine-tuned on clean + noisy | 0.817 ± 0.007 | 0.782 ± 0.007 | 0.773 ± 0.007 | 0.817 ± 0.007 | 0.792 ± 0.005 | 0.802 ± 0.007 | 0.785 ± 0.008 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.099 | +0.139 | 139% |
| space | 0.098 | +0.128 | 131% |
| case | 0.000 | +0.074 | n/a (no loss) |
| punct | 0.025 | +0.073 | 297% |
| number | 0.139 | +0.199 | 143% |
| mixed | 0.146 | +0.187 | 129% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): -0.074

### gretel

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.787 | 0.711 | 0.710 | 0.787 | 0.771 | 0.764 | 0.736 |
| BERT, fine-tuned on clean | 0.492 ± 0.012 | 0.423 ± 0.005 | 0.427 ± 0.007 | 0.492 ± 0.012 | 0.491 ± 0.022 | 0.440 ± 0.014 | 0.430 ± 0.012 |
| BERT, fine-tuned on clean + noisy | 0.519 ± 0.006 | 0.489 ± 0.007 | 0.489 ± 0.007 | 0.519 ± 0.006 | 0.527 ± 0.008 | 0.513 ± 0.006 | 0.506 ± 0.005 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.069 | +0.066 | 96% |
| space | 0.065 | +0.062 | 95% |
| case | 0.000 | +0.027 | n/a (no loss) |
| punct | 0.001 | +0.035 | n/a (no loss) |
| number | 0.052 | +0.073 | 140% |
| mixed | 0.062 | +0.076 | 122% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): -0.027

