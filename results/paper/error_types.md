# Error types, comparison A

FN: *boundary* if a detection with the same label overlaps the gold span, *wrong label* if only detections with another label do, else *missed*. FP: *boundary*, *wrong label* or *spurious*, on the same principle against the gold spans.

| Dataset | Detector | FN | missed | boundary | wrong label | FP | spurious | boundary | wrong label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai4privacy | Presidio | 11486 | 11332 | 149 | 5 | 587 | 433 | 149 | 5 |
| ai4privacy | BERT | 2420 | 850 | 1262 | 308 | 4116 | 2458 | 1382 | 276 |
| ai4privacy | GLiNER | 1621 | 750 | 812 | 59 | 2376 | 1641 | 683 | 52 |
| Nemotron-PII | Presidio | 2653 | 2595 | 1 | 57 | 1939 | 1881 | 1 | 57 |
| Nemotron-PII | BERT | 507 | 292 | 213 | 2 | 4290 | 4043 | 238 | 9 |
| Nemotron-PII | GLiNER | 595 | 447 | 135 | 13 | 1870 | 1732 | 127 | 11 |
| Gretel | Presidio | 7055 | 7006 | 27 | 22 | 3992 | 3943 | 27 | 22 |
| Gretel | BERT | 1131 | 468 | 614 | 49 | 9740 | 9003 | 669 | 68 |
| Gretel | GLiNER | 1891 | 1522 | 250 | 119 | 5514 | 5137 | 244 | 133 |

The FN and FP totals equal those of the saved scores (checked by the script).
