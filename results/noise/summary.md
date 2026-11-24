# Questions 2 and 3: noisy test sets, comparison A

Micro scores with strict matching. Noisy copies: about one edit per ten words (benchmark/noise.py, experiments/make_noisy.py).

## Question 2: F2 of each detector

### F2, ai4privacy

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.157 | 0.149 (-0.008) | 0.138 (-0.019) | 0.157 (+0.000) | 0.157 (-0.000) | 0.157 (-0.000) | 0.150 (-0.007) |
| bert | 0.796 | 0.708 (-0.089) | 0.681 (-0.116) | 0.796 (+0.000) | 0.780 (-0.016) | 0.794 (-0.002) | 0.738 (-0.058) |
| gliner | 0.867 | 0.829 (-0.038) | 0.792 (-0.076) | 0.839 (-0.028) | 0.856 (-0.011) | 0.864 (-0.003) | 0.828 (-0.040) |

### F2, nemotron

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.431 | 0.417 (-0.014) | 0.373 (-0.057) | 0.431 (+0.000) | 0.431 (+0.000) | 0.372 (-0.059) | 0.376 (-0.055) |
| bert | 0.762 | 0.713 (-0.049) | 0.685 (-0.077) | 0.762 (+0.000) | 0.755 (-0.007) | 0.762 (-0.000) | 0.724 (-0.038) |
| gliner | 0.823 | 0.813 (-0.010) | 0.787 (-0.037) | 0.815 (-0.008) | 0.818 (-0.005) | 0.818 (-0.005) | 0.806 (-0.017) |

### F2, gretel

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.171 | 0.163 (-0.008) | 0.154 (-0.017) | 0.171 (+0.000) | 0.171 (-0.000) | 0.164 (-0.007) | 0.157 (-0.014) |
| bert | 0.718 | 0.654 (-0.064) | 0.649 (-0.069) | 0.718 (+0.000) | 0.706 (-0.011) | 0.721 (+0.003) | 0.685 (-0.033) |
| gliner | 0.713 | 0.704 (-0.009) | 0.699 (-0.014) | 0.717 (+0.004) | 0.700 (-0.012) | 0.714 (+0.001) | 0.706 (-0.007) |

### Leak rate, ai4privacy

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.868 | 0.875 (+0.007) | 0.880 (+0.013) | 0.868 (+0.000) | 0.868 (-0.000) | 0.868 (+0.000) | 0.872 (+0.005) |
| bert | 0.113 | 0.132 (+0.018) | 0.147 (+0.034) | 0.113 (+0.000) | 0.114 (+0.000) | 0.115 (+0.002) | 0.127 (+0.014) |
| gliner | 0.061 | 0.071 (+0.011) | 0.104 (+0.043) | 0.090 (+0.030) | 0.070 (+0.010) | 0.064 (+0.003) | 0.085 (+0.024) |

### Leak rate, nemotron

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.572 | 0.586 (+0.014) | 0.621 (+0.050) | 0.572 (+0.000) | 0.571 (-0.000) | 0.650 (+0.079) | 0.637 (+0.065) |
| bert | 0.088 | 0.092 (+0.005) | 0.113 (+0.025) | 0.088 (+0.000) | 0.086 (-0.002) | 0.100 (+0.012) | 0.109 (+0.021) |
| gliner | 0.101 | 0.110 (+0.009) | 0.128 (+0.026) | 0.107 (+0.005) | 0.099 (-0.002) | 0.106 (+0.005) | 0.118 (+0.016) |

### Leak rate, gretel

| Detector | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| presidio | 0.839 | 0.847 (+0.008) | 0.854 (+0.016) | 0.839 (+0.000) | 0.839 (+0.000) | 0.850 (+0.011) | 0.854 (+0.015) |
| bert | 0.083 | 0.097 (+0.014) | 0.113 (+0.030) | 0.083 (+0.000) | 0.096 (+0.012) | 0.085 (+0.001) | 0.096 (+0.013) |
| gliner | 0.195 | 0.215 (+0.020) | 0.221 (+0.026) | 0.196 (+0.001) | 0.217 (+0.022) | 0.195 (+0.001) | 0.214 (+0.019) |

In brackets: the change from the clean test set.

## Question 3: fine-tuned BERT (F2, mean ± sd over seeds)

- BERT, fine-tuned on clean: bert-clean-s0, bert-clean-s1, bert-clean-s2
- BERT, fine-tuned on clean + noisy: bert-noisy-s0, bert-noisy-s1, bert-noisy-s2

### ai4privacy

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.796 | 0.708 | 0.681 | 0.796 | 0.780 | 0.794 | 0.738 |
| BERT, fine-tuned on clean | 0.986 ± 0.001 | 0.873 ± 0.000 | 0.848 ± 0.001 | 0.986 ± 0.001 | 0.978 ± 0.000 | 0.971 ± 0.001 | 0.915 ± 0.000 |
| BERT, fine-tuned on clean + noisy | 0.983 ± 0.000 | 0.951 ± 0.000 | 0.940 ± 0.001 | 0.983 ± 0.000 | 0.976 ± 0.001 | 0.981 ± 0.000 | 0.962 ± 0.001 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.113 | +0.079 | 70% |
| space | 0.138 | +0.092 | 67% |
| case | 0.000 | -0.003 | n/a (no loss) |
| punct | 0.007 | -0.003 | -37% |
| number | 0.015 | +0.011 | 69% |
| mixed | 0.071 | +0.047 | 66% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): +0.003

### nemotron

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.762 | 0.713 | 0.685 | 0.762 | 0.755 | 0.762 | 0.724 |
| BERT, fine-tuned on clean | 0.713 ± 0.009 | 0.596 ± 0.003 | 0.601 ± 0.005 | 0.713 ± 0.009 | 0.684 ± 0.013 | 0.690 ± 0.010 | 0.630 ± 0.007 |
| BERT, fine-tuned on clean + noisy | 0.781 ± 0.008 | 0.742 ± 0.007 | 0.730 ± 0.009 | 0.781 ± 0.008 | 0.750 ± 0.009 | 0.786 ± 0.005 | 0.756 ± 0.005 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.116 | +0.145 | 125% |
| space | 0.112 | +0.130 | 116% |
| case | 0.000 | +0.068 | n/a (no loss) |
| punct | 0.029 | +0.067 | 229% |
| number | 0.023 | +0.096 | 420% |
| mixed | 0.082 | +0.126 | 153% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): -0.068

### gretel

| Model | clean | typo | space | case | punct | number | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT, original | 0.718 | 0.654 | 0.649 | 0.718 | 0.706 | 0.721 | 0.685 |
| BERT, fine-tuned on clean | 0.508 ± 0.011 | 0.417 ± 0.001 | 0.425 ± 0.003 | 0.508 ± 0.011 | 0.509 ± 0.024 | 0.509 ± 0.012 | 0.461 ± 0.010 |
| BERT, fine-tuned on clean + noisy | 0.534 ± 0.009 | 0.499 ± 0.011 | 0.496 ± 0.011 | 0.534 ± 0.009 | 0.548 ± 0.011 | 0.536 ± 0.010 | 0.521 ± 0.008 |

| Noisy copy | Loss of the clean-trained BERT | Won back by the noisy-trained BERT | Share won back |
|---|---:|---:|---:|
| typo | 0.091 | +0.082 | 90% |
| space | 0.083 | +0.071 | 86% |
| case | 0.000 | +0.026 | n/a (no loss) |
| punct | -0.001 | +0.039 | n/a (no loss) |
| number | -0.001 | +0.027 | n/a (no loss) |
| mixed | 0.047 | +0.060 | 128% |

Clean cost (F2 on clean, clean-trained minus noisy-trained): -0.026

