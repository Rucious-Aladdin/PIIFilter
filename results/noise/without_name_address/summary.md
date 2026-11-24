# Questions 2 and 3: noisy test sets, comparison A

Micro scores with strict matching. Noisy copies: about one edit per ten words (benchmark/noise.py, experiments/make_noisy.py).

## Question 2: F2 of each detector

### F2, ai4privacy

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.616 | 0.587 (-0.029) | 0.545 (-0.070) | 0.616 (+0.000) | 0.616 (-0.000) | 0.614 (-0.001) | 0.590 (-0.026) |
| bert | 0.891 | 0.858 (-0.033) | 0.837 (-0.055) | 0.891 (+0.000) | 0.889 (-0.002) | 0.883 (-0.008) | 0.861 (-0.030) |
| gliner | 0.898 | 0.880 (-0.017) | 0.842 (-0.055) | 0.894 (-0.004) | 0.889 (-0.008) | 0.885 (-0.013) | 0.871 (-0.027) |

### F2, nemotron

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.757 | 0.735 (-0.022) | 0.661 (-0.096) | 0.757 (+0.000) | 0.757 (+0.000) | 0.687 (-0.070) | 0.683 (-0.074) |
| bert | 0.757 | 0.746 (-0.011) | 0.715 (-0.042) | 0.757 (+0.000) | 0.746 (-0.011) | 0.762 (+0.004) | 0.737 (-0.020) |
| gliner | 0.960 | 0.952 (-0.009) | 0.913 (-0.048) | 0.959 (-0.001) | 0.960 (+0.000) | 0.953 (-0.008) | 0.936 (-0.024) |

### F2, gretel

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.569 | 0.546 (-0.024) | 0.517 (-0.052) | 0.569 (+0.000) | 0.569 (-0.000) | 0.611 (+0.042) | 0.549 (-0.020) |
| bert | 0.459 | 0.465 (+0.006) | 0.455 (-0.003) | 0.459 (+0.000) | 0.463 (+0.004) | 0.463 (+0.005) | 0.458 (-0.000) |
| gliner | 0.737 | 0.740 (+0.003) | 0.731 (-0.006) | 0.733 (-0.004) | 0.723 (-0.014) | 0.740 (+0.003) | 0.729 (-0.008) |

### Leak rate, ai4privacy

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.406 | 0.437 (+0.031) | 0.463 (+0.057) | 0.406 (+0.000) | 0.406 (+0.000) | 0.406 (+0.000) | 0.427 (+0.021) |
| bert | 0.054 | 0.073 (+0.018) | 0.065 (+0.011) | 0.054 (+0.000) | 0.059 (+0.005) | 0.061 (+0.006) | 0.064 (+0.010) |
| gliner | 0.071 | 0.083 (+0.012) | 0.095 (+0.024) | 0.074 (+0.003) | 0.077 (+0.006) | 0.085 (+0.015) | 0.085 (+0.014) |

### Leak rate, nemotron

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.103 | 0.134 (+0.030) | 0.207 (+0.104) | 0.103 (+0.000) | 0.103 (-0.000) | 0.269 (+0.165) | 0.240 (+0.137) |
| bert | 0.024 | 0.032 (+0.008) | 0.042 (+0.017) | 0.024 (+0.000) | 0.031 (+0.006) | 0.048 (+0.024) | 0.052 (+0.028) |
| gliner | 0.015 | 0.022 (+0.007) | 0.037 (+0.022) | 0.014 (-0.001) | 0.014 (-0.000) | 0.022 (+0.007) | 0.032 (+0.017) |

### Leak rate, gretel

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.157 | 0.199 (+0.042) | 0.239 (+0.083) | 0.157 (+0.000) | 0.157 (+0.000) | 0.217 (+0.060) | 0.237 (+0.080) |
| bert | 0.030 | 0.032 (+0.003) | 0.045 (+0.016) | 0.030 (+0.000) | 0.034 (+0.004) | 0.041 (+0.011) | 0.039 (+0.009) |
| gliner | 0.205 | 0.206 (+0.001) | 0.204 (-0.001) | 0.210 (+0.005) | 0.233 (+0.028) | 0.208 (+0.003) | 0.219 (+0.014) |

In brackets: the change from the clean test set.

## Question 3: fine-tuned BERT (F2, mean ± sd over seeds)

- BERT, fine-tuned on clean: bert-clean-s0, bert-clean-s1, bert-clean-s2
- BERT, fine-tuned on clean + noisy: bert-noisy-s0, bert-noisy-s1, bert-noisy-s2

### ai4privacy

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.891 | 0.858 | 0.837 | 0.891 | 0.889 | 0.883 | 0.861 |
| BERT, fine-tuned on clean | 0.993 ± 0.000 | 0.951 ± 0.001 | 0.947 ± 0.003 | 0.993 ± 0.000 | 0.991 ± 0.001 | 0.924 ± 0.006 | 0.950 ± 0.001 |
| BERT, fine-tuned on clean + noisy | 0.993 ± 0.001 | 0.990 ± 0.001 | 0.979 ± 0.002 | 0.993 ± 0.001 | 0.992 ± 0.001 | 0.983 ± 0.001 | 0.986 ± 0.001 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.042 | +0.038 | 92% |
| space | 0.046 | +0.032 | 70% |
| case | 0.000 | -0.000 | n/a (no loss) |
| punct | 0.002 | +0.001 | n/a (no loss) |
| number | 0.069 | +0.059 | 86% |
| mixed | 0.044 | +0.036 | 84% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): +0.000

### nemotron

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.757 | 0.746 | 0.715 | 0.757 | 0.746 | 0.762 | 0.737 |
| BERT, fine-tuned on clean | 0.618 ± 0.014 | 0.574 ± 0.014 | 0.572 ± 0.016 | 0.618 ± 0.014 | 0.592 ± 0.016 | 0.571 ± 0.020 | 0.558 ± 0.018 |
| BERT, fine-tuned on clean + noisy | 0.720 ± 0.020 | 0.694 ± 0.015 | 0.678 ± 0.019 | 0.720 ± 0.020 | 0.691 ± 0.021 | 0.738 ± 0.013 | 0.710 ± 0.014 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.044 | +0.120 | 273% |
| space | 0.046 | +0.106 | 229% |
| case | 0.000 | +0.102 | n/a (no loss) |
| punct | 0.026 | +0.098 | 385% |
| number | 0.047 | +0.168 | 355% |
| mixed | 0.060 | +0.151 | 254% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): -0.102

### gretel

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.459 | 0.465 | 0.455 | 0.459 | 0.463 | 0.463 | 0.458 |
| BERT, fine-tuned on clean | 0.507 ± 0.010 | 0.491 ± 0.016 | 0.495 ± 0.017 | 0.507 ± 0.010 | 0.497 ± 0.016 | 0.523 ± 0.018 | 0.491 ± 0.013 |
| BERT, fine-tuned on clean + noisy | 0.524 ± 0.008 | 0.521 ± 0.011 | 0.518 ± 0.012 | 0.524 ± 0.008 | 0.522 ± 0.009 | 0.536 ± 0.011 | 0.527 ± 0.006 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.016 | +0.030 | 191% |
| space | 0.012 | +0.023 | 191% |
| case | 0.000 | +0.018 | n/a (no loss) |
| punct | 0.010 | +0.026 | 258% |
| number | -0.016 | +0.013 | n/a (no loss) |
| mixed | 0.015 | +0.036 | 234% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): -0.018

