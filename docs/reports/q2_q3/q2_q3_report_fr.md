# PIIGuard, questions 2 et 3 : combien coûte le bruit, et un entraînement sur texte bruité le rattrape-t-il ?

*Rapport d'analyse. Équipe PIIGuard (Louis Lenoble, Sunho Cha, Suseong Kim), CSED490E, 24 novembre 2026.*

## Résumé

La question 2 fait tourner les trois détecteurs de la question 1 (Presidio avec ses seules
règles, le BERT de Gravitee, GLiNER) sur des copies bruitées des mêmes jeux de test. La
question 3 affine ce BERT sur du texte ai4privacy propre, ou sur le même texte plus une copie
bruitée de chaque exemple, et compare les deux versions. La notation, l'appariement et les
statistiques sont ceux de la question 1 ([rapport](../q1/q1_report_fr.md)) : F2 micro en
appariement strict, taux de fuite, intervalles à 95 % par bootstrap apparié par groupes, tests
de randomisation appariés avec correction de Holm.

1. **Les espaces oubliées ou en trop coûtent le plus**, puis les fautes de frappe. Sur les six
   labels, BERT perd 0,069 à 0,116 de F2 quand environ un mot sur dix a une espace retirée ou
   ajoutée, GLiNER 0,014 à 0,076.
2. **GLiNER résiste mieux que BERT** aux fautes de frappe, aux espaces et au bruit mélangé, sur
   les trois datasets : sur les copies mélangées, BERT perd 0,018 à 0,026 de F2 de plus
   (intervalles à 95 % sous 0). Les minuscules font exception : elles ne changent rien pour
   BERT (son tokenizer met tout en minuscules) mais coûtent à GLiNER 0,028 de F2 et 3,0 points
   de taux de fuite sur ai4privacy.
3. **Le format des nombres casse les dates.** Quand toutes les dates sont notées (comparaison
   A-dates), retirer ou mélanger les séparateurs coûte à Presidio 0,241 de F2 sur Nemotron-PII,
   et à BERT 0,066.
4. **Dans son propre domaine, l'entraînement sur copies bruitées rattrape deux tiers de la
   perte.** Sur ai4privacy, le BERT affiné sur texte propre perd 0,071 de F2 sur la copie
   mélangée ; la version entraînée sur texte propre + bruité y fait 0,047 [0,044 ; 0,050] de
   mieux, et coûte 0,003 sur texte propre (détectable, mais sous le seuil de 0,01 qui compte).
5. **Sur les autres datasets, la version bruitée est meilleure partout, texte propre compris** :
   +0,068 de F2 sur Nemotron-PII propre, +0,026 sur Gretel propre, +0,126 et +0,060 sur les
   copies mélangées, et des taux de fuite plus bas. Toutes ces différences sont établies, et
   chaque graine de la version bruitée fait mieux que chaque graine de la version propre.
6. **Affiner sur ai4privacy seul ne donne pas un meilleur détecteur pour d'autres données.** Les
   deux versions affinées sont sous le BERT d'origine sur Gretel (0,51 à 0,53 contre 0,72),
   surtout à cause des adresses, car ai4privacy ne marque que le numéro et la rue. Comme le
   demande le proposal (annexe B), les versions affinées ne sont comparées qu'entre elles.

Les tableaux clés des trois questions sont aussi dans un PDF de trois pages (en anglais) :
[results_tables.pdf](../results_tables/results_tables.pdf).

---

## 1. Les questions

> **Q2.** Combien Presidio, BERT et GLiNER perdent-ils quand le texte contient les erreurs que
> l'on fait en tapant vite (proposal, section 4.2) ?
>
> **Q3.** Affiner BERT sur du texte bruité rattrape-t-il cette perte, et que coûte-t-il sur
> texte propre (proposal, section 5.2 et annexe B) ?

Les deux questions reprennent les jeux de test, les labels, le nettoyage des spans, les
détecteurs, la notation et les comparaisons de la question 1. Seul le texte change (question 2)
ou les poids de BERT (question 3).

## 2. Jeux de test bruités

La procédure pas à pas, avec la somme de contrôle de chaque fichier, est dans [data_generation.pdf](../data/data_generation.pdf) (en anglais).

### 2.1 Les types de bruit

`benchmark/noise.py` fait une copie bruitée d'un texte pour chaque type d'erreur de la section
4.2, plus un mélange :

| Type | Modification | Exemple (inventé) |
| --- | --- | --- |
| `typo` | dans un mot de 3 lettres ou plus : une lettre ajoutée, retirée, échangée avec la suivante, ou remplacée par une touche voisine du clavier QWERTY | example → exmaple |
| `space` | une espace retirée entre deux mots, ou ajoutée dans un mot de 4 lettres ou plus | is minji.park@... → isminji.park@... |
| `case` | un mot avec majuscules tapé en minuscules | Minji Park → minji park |
| `punct` | ponctuation de fin de phrase ou de proposition retirée (pas dans `x@y.com` ni `1,234`) | les phrases se suivent sans coupure |
| `number` | les séparateurs d'un nombre retirés, ou mélangés (espace, tiret, rien) | 010-2345-6789 → 01023456789 |
| `mixed` | chaque modification d'un des cinq types ci-dessus, tiré au hasard | |

### 2.2 Comment les spans du gold suivent le texte

Chaque modification remplace une plage non vide du texte d'origine, et les modifications ne se
chevauchent jamais. Une modification est soit dans un span du gold, soit hors de tous : une
modification qui couperait un bord de span n'est pas faite. Les spans suivent alors le texte (un
début se déplace de la variation de longueur des modifications qui le précèdent, une fin aussi
de celles qui sont dans le span), gardent leur label (un e-mail mal tapé reste un e-mail) et
sont nettoyés avec les mêmes règles que tout span du gold.

Une règle de plus vient d'une vérification de la première version : une modification qui ne
laisserait que des espaces entre deux valeurs de même label n'est pas faite. Retirer la virgule
de « Roger Leblanc, Roger Leblanc » laisserait le nettoyage des spans fusionner les deux noms en
un seul span du gold : le bruit changerait alors le gold lui-même. Avec cette règle, **chaque
jeu de test bruité a exactement le même nombre de spans du gold par label que sa version
propre** (vérifié pour les 36 jeux).

### 2.3 Taux et graines

Une copie fait round(0,10 × mots) modifications, au moins une : environ une modification pour
dix mots. Un texte avec peu de places pour un type de bruit en reçoit moins (une copie `number`
d'un texte sans nombre reste telle quelle). Le taux et la graine (1) ont été fixés avant toute
notation. Chaque exemple tire de son propre générateur, `Random("<graine>:<type>:<id de
l'exemple>")`, donc une copie ne dépend pas de l'ordre des exemples. Les versions A-dates
reçoivent les mêmes modifications que les versions à six labels, car les deux listes de spans
du gold sont passées ensemble et aucune modification ne peut couper l'une ou l'autre.

Les jeux de test bruités sont `data/test/<dataset>_noisy_<type>[_dates].jsonl` (3 datasets ×
6 types × 2 ensembles de labels = 36 fichiers), avec les mêmes ids que les jeux propres. Le
`meta.noise` de chaque exemple liste ses modifications et dit si chacune tombe dans une valeur.

### 2.4 À quoi ressemblent les jeux de test bruités

D'après `results/noise/test_sets.md` :

| Dataset | Exemples | Modifications par exemple (mixed) | Modifications dans une valeur PII (mixed) | Exemples modifiés par `number` |
| ------------ | -------: | --------------------------------: | ----------------------------------------: | -----------------------------: |
| ai4privacy | 10 000 | 1,7 | 18,1 % | 2 487 |
| Nemotron-PII | 2 000 | 13,2 | 5,2 % | 1 711 |
| Gretel | 2 962 | 19,1 | 5,8 % | 1 992 |

Les phrases d'ai4privacy sont courtes et denses en PII, donc environ une modification sur cinq
touche une valeur ; dans les longs documents de Nemotron-PII et de Gretel, la plupart des
modifications tombent dans les mots autour des valeurs. Dans les valeurs, ce sont `case` (38 %
sur ai4privacy) et `number` (59 %) qui frappent le plus, `punct` presque jamais (une valeur se
termine rarement par une ponctuation de phrase).

## 3. Affiner BERT

La procédure d'entraînement complète est dans [training_procedure.pdf](../train/training_procedure.pdf) (en anglais).

### 3.1 Données

`experiments/prepare_train.py` lit les exemples anglais de la partition **train** d'ai4privacy
(120 533 ; notre jeu de test ai4privacy vient de la partition de validation), en garde 5 000 à
part comme jeu de dev (graine 0) et utilise les 115 533 autres pour l'entraînement. Il écrit
aussi une copie bruitée (mélangée) de chaque exemple, faite comme les jeux de test bruités mais
avec la **graine 2** : aucune copie de test n'est jamais vue à l'entraînement.

| Fichier | Exemples | NAME | EMAIL | PHONE | ADDRESS | DATE | CARD_NUMBER |
| --------------------- | -----------: | -------------------------------------------------------------------------: | -----: | -----: | ------: | -----: | ----------: |
| train, propre | 115 533 | 101 783 | 13 282 | 17 984 | 17 487 | 16 391 | 2 148 |
| train, bruité | 115 533 | mêmes spans, 201 528 modifications (1,7 par exemple, 21 % dans une valeur) |  |  |  |  |  |
| dev, propre et bruité | 5 000 chacun | 4 443 | 592 | 760 | 765 | 692 | 104 |

La version **(a)** apprend sur *train, propre* ; la version **(b)** sur *train, propre* +
*train, bruité*.

### 3.2 Labels et étiquettes

Les deux versions partent du BERT du benchmark (`gravitee-io/bert-small-pii-detection`,
révision `f8c27a85c5`) et gardent ses 51 étiquettes BIO, donc `baselines/bert_detector.py` les
lit exactement comme le modèle d'origine. Nos labels sont appris comme les classes du modèle :

| Notre label | Classe BERT |
| ----------------- | ------------- |
| NAME | PERSON |
| EMAIL | EMAIL_ADDRESS |
| PHONE | PHONE_NUMBER |
| ADDRESS | LOCATION |
| DATE (toute date) | DATE_TIME |
| CARD_NUMBER | CREDIT_CARD |

Toutes les dates sont apprises comme DATE_TIME, comme dans la comparaison C : ai4privacy n'a
pas de label de date de naissance, et laisser ses dates sans étiquette apprendrait au modèle à
oublier sa classe de date. Sur les comparaisons à six labels, le détecteur ramène toujours
DATE_TIME à DATE_OF_BIRTH, comme pour le modèle d'origine. Tous les autres tokens sont `O` : le
modèle apprend aussi que les villes, les titres, etc. ne sont pas nos labels. Un token prend le
span où tombe son premier caractère : `B-` pour le premier token d'un span, `I-` pour les
suivants, ignoré (-100) pour `[CLS]` et `[SEP]`.

### 3.3 Recette

| Réglage | Valeur |
| --- | --- |
| Entraîneur | `Trainer` de Hugging Face, AdamW (par défaut), décroissance linéaire |
| Taux d'apprentissage | 5e-5, montée progressive sur les premiers 6 % des pas |
| Weight decay | 0,01 |
| Batch | 32 |
| Pas | 10 832 pour **les deux** versions : trois passes sur le jeu d'entraînement propre. La version (b) voit chacun de ses exemples 1,5 fois |
| Longueur | 256 tokens, la fenêtre que lit `BertDetector` (exemples : médiane de 29 tokens ; 1 sur 115 533 est coupé) |
| Précision | bf16 |
| Point de sauvegarde | le dernier pas ; les jeux de dev sont seulement journalisés, jamais utilisés pour choisir un modèle (le dev bruité favoriserait la version b) |
| Graines | 0, 1 et 2 pour chaque version (6 modèles) |

La seule différence entre (a) et (b), ce sont les copies bruitées : même modèle de départ,
mêmes réglages, même nombre de pas, mêmes graines. Aucun réglage n'a été optimisé. Les réglages
sont fixés dans `experiments/train_bert.py`, et chaque entraînement écrit sa fiche (réglages,
sommes de contrôle des données, toutes les pertes journalisées, durée, GPU, versions des
paquets) dans `results/training/<modèle>.json`.

### 3.4 Entraînements

Chaque modèle s'est entraîné en environ 2 minutes sur un GPU (cinq sur RTX 4090, un sur L40S),
avec torch 2.14.1 et transformers 5.16.1 (`results/training/summary.md`). Perte sur le dev
pendant l'entraînement, moyenne des trois graines :

| Modèle | Jeu de dev | pas 1 805 | 3 610 | 5 415 | 7 220 | 9 025 | 10 830 |
| ------------------- | ---------- | --------: | -----: | -----: | -----: | -----: | -----: |
| (a) propre | propre | 0,0173 | 0,0135 | 0,0127 | 0,0128 | 0,0127 | 0,0122 |
| (a) propre | bruité | 0,0945 | 0,1048 | 0,1143 | 0,1250 | 0,1352 | 0,1335 |
| (b) propre + bruité | propre | 0,0210 | 0,0162 | 0,0146 | 0,0129 | 0,0134 | 0,0130 |
| (b) propre + bruité | bruité | 0,0411 | 0,0319 | 0,0285 | 0,0264 | 0,0269 | 0,0258 |

La version propre seule devient *moins bonne* sur le dev bruité au fil de l'entraînement
(0,095 à 0,134), pour les trois graines : elle colle de plus en plus à l'orthographe propre de
ses données. La version (b) progresse sur les deux jeux de dev.

## 4. Évaluation et statistiques

- **Exécutions.** Presidio, BERT et GLiNER sur les 36 jeux de test bruités, avec les réglages
  de la question 1 (seuil de GLiNER 0,35, révisions figées) : Presidio 3 minutes sur CPU, BERT
  16 minutes sur un GPU, GLiNER 48 minutes sur trois. Les six modèles affinés sur les jeux de
  test propres et bruités (42 jeux chacun), lus par le même `BertDetector`. Toutes les
  prédictions sont dans `results/predictions/`.
- **Scores.** `experiments/score.py`, inchangé : P, R, F1, F2 micro et taux de fuite,
  appariement strict, comparaisons A, B, A-dates et B-dates. Le score d'une version affinée est
  la moyenne de ses trois graines.
- **Intervalles et tests** (`experiments/significance_noise.py`). La méthode de la question 1 :
  bootstrap apparié par groupes (10 000 tirages) pour les intervalles à 95 %, et tests de
  randomisation approchée appariés. Les groupes sont ceux du jeu de test propre, donc les copies
  bruitées d'un exemple sont toujours tirées avec lui. Pour une version affinée, la moyenne sur
  les graines est prise à chaque tirage et à chaque tour, et le test échange les sorties des
  deux versions pour les trois paires de graines à la fois. Trois familles de tests, chacune
  pour le F1, le F2 et le taux de fuite, avec correction de Holm : l'écart de Presidio, BERT et
  GLiNER sur chaque copie bruitée (216 tests), version bruitée moins version propre sur chaque
  jeu de test (84), et l'écart de chaque version affinée sur chaque copie bruitée (144). Ces
  familles sont plus grandes que celles de la question 1, d'où 20 000 tours de randomisation.
- **Établie**, comme pour la question 1 : p-value corrigée sous 0,05, intervalle qui exclut 0,
  et écart d'au moins 0,01.

Ci-dessous, les valeurs en *italique* ne sont pas établies ; dans les tableaux de la question 3,
les différences en **gras** le sont. Tous les intervalles sont dans
`results/analysis/significance_noise.md`.

## 5. Question 2 : les détecteurs prêts à l'emploi sur texte bruité

#### F2 sur le jeu de test propre et écart sur chaque copie bruitée, comparaison A (six labels)

| Dataset | Détecteur | propre | typo | space | case | punct | number | mixed |
| ------------ | --------- | -----: | -------: | -----: | -------: | -------: | -------: | -------: |
| ai4privacy | Presidio | 0,157 | *−0,008* | −0,019 | *0,000* | *0,000* | *0,000* | *−0,007* |
|  | BERT | 0,796 | −0,089 | −0,116 | *0,000* | −0,016 | *−0,002* | −0,058 |
|  | GLiNER | 0,867 | −0,038 | −0,076 | −0,028 | −0,011 | *−0,003* | −0,040 |
| Nemotron-PII | Presidio | 0,431 | −0,014 | −0,057 | *0,000* | *0,000* | −0,059 | −0,055 |
|  | BERT | 0,762 | −0,049 | −0,077 | *0,000* | *−0,007* | *0,000* | −0,038 |
|  | GLiNER | 0,823 | −0,010 | −0,037 | *−0,008* | *−0,005* | *−0,005* | −0,017 |
| Gretel | Presidio | 0,171 | *−0,008* | −0,017 | *0,000* | *0,000* | *−0,007* | −0,014 |
|  | BERT | 0,718 | −0,064 | −0,069 | *0,000* | −0,011 | *+0,003* | −0,033 |
|  | GLiNER | 0,713 | *−0,009* | −0,014 | *+0,004* | −0,012 | *+0,001* | *−0,007* |

Tous les écarts de 0,01 ou plus sont établis, et aucun des plus petits.

**Écart de BERT moins écart de GLiNER** (intervalle à 95 % ; négatif : BERT perd plus) :

| Dataset | typo | space | mixed |
| ------------ | ------------------------ | ------------------------ | ------------------------ |
| ai4privacy | −0,050 [−0,056 ; −0,045] | −0,040 [−0,046 ; −0,034] | −0,018 [−0,023 ; −0,013] |
| Nemotron-PII | −0,038 [−0,046 ; −0,030] | −0,040 [−0,049 ; −0,032] | −0,021 [−0,028 ; −0,014] |
| Gretel | −0,054 [−0,062 ; −0,047] | −0,055 [−0,062 ; −0,048] | −0,026 [−0,033 ; −0,020] |

Cette différence d'écarts n'a que des intervalles, pas de test. Sur Gretel propre, les deux
détecteurs sont à égalité en F2 (question 1) ; sur les copies typo, space et mixed, le F2 de
GLiNER est le plus haut (0,706 contre 0,685 sur mixed ; les deux détecteurs n'ont pas été
testés l'un contre l'autre sur les copies bruitées).

#### Taux de fuite (%) sur le jeu de test propre et écart en points, comparaison A (plus haut = pire)

| Dataset | Détecteur | propre | typo | space | case | punct | number | mixed |
| ------------ | --------- | -----: | -----: | ----: | -----: | -----: | -----: | -----: |
| ai4privacy | Presidio | 86,8 | *+0,7* | +1,3 | *0,0* | *0,0* | *0,0* | *+0,5* |
|  | BERT | 11,3 | +1,8 | +3,4 | *0,0* | *0,0* | *+0,2* | +1,4 |
|  | GLiNER | 6,1 | +1,1 | +4,3 | +3,0 | *+1,0* | *+0,3* | +2,4 |
| Nemotron-PII | Presidio | 57,2 | +1,4 | +5,0 | *0,0* | *0,0* | +7,9 | +6,5 |
|  | BERT | 8,8 | *+0,5* | +2,5 | *0,0* | *−0,2* | +1,2 | +2,1 |
|  | GLiNER | 10,1 | *+0,9* | +2,6 | *+0,5* | *−0,2* | *+0,5* | +1,6 |
| Gretel | Presidio | 83,9 | *+0,8* | +1,6 | *0,0* | *0,0* | +1,1 | +1,5 |
|  | BERT | 8,3 | +1,4 | +3,0 | *0,0* | +1,2 | *+0,1* | +1,3 |
|  | GLiNER | 19,5 | +2,0 | +2,6 | *+0,1* | +2,2 | *+0,1* | +1,9 |

#### F2, comparaison A-dates (toutes les dates notées)

| Dataset | Détecteur | propre | typo | space | case | punct | number | mixed |
| ------------ | --------- | -----: | -------: | -----: | -------: | -------: | -------: | -----: |
| ai4privacy | Presidio | 0,178 | *−0,007* | −0,021 | *0,000* | *0,000* | −0,034 | −0,017 |
|  | BERT | 0,790 | −0,082 | −0,110 | *0,000* | −0,016 | −0,022 | −0,059 |
|  | GLiNER | 0,872 | −0,035 | −0,075 | −0,026 | −0,011 | *−0,010* | −0,040 |
| Nemotron-PII | Presidio | 0,532 | −0,010 | −0,050 | *0,000* | *0,000* | −0,241 | −0,168 |
|  | BERT | 0,778 | −0,042 | −0,067 | *0,000* | *−0,001* | −0,066 | −0,072 |
|  | GLiNER | 0,801 | *−0,004* | −0,033 | *−0,008* | *−0,004* | −0,035 | −0,033 |
| Gretel | Presidio | 0,227 | *−0,006* | −0,013 | *0,000* | *0,000* | −0,081 | −0,050 |
|  | BERT | 0,787 | −0,075 | −0,077 | *0,000* | −0,016 | −0,022 | −0,051 |
|  | GLiNER | 0,595 | −0,010 | −0,015 | *+0,002* | −0,012 | −0,013 | −0,016 |

### Lecture

1. **Les espaces, puis les fautes de frappe.** Un mot collé à son voisin ou coupé en deux change
   les tokens que voient les modèles : les bords des spans prédits bougent, et l'appariement
   strict compte un manqué. Des cinq types, la copie `space` est la plus grande perte de BERT et
   de GLiNER sur chaque dataset dans la comparaison A ; en A-dates, le bruit sur les nombres
   s'en approche, et la dépasse pour GLiNER sur Nemotron-PII.
2. **La casse teste le pipeline autant que les modèles.** Le tokenizer de BERT ignore la casse :
   ses sorties sur les copies `case` sont identiques à celles sur le texte propre, comme il se
   doit. GLiNER se sert des majuscules pour repérer les noms, et manque plus souvent les noms en
   minuscules.
3. **Le point faible de Presidio est le format.** Ses règles perdent peu sur les mots mais
   beaucoup sur les nombres, et surtout sur les dates quand elles sont notées (A-dates).
4. **Le bruit dans les mots autour d'une valeur compte aussi.** Dans Nemotron-PII et Gretel,
   seulement 2 à 4 % des modifications `space` tombent dans une valeur, et BERT perd pourtant
   environ 0,07 de F2 sur ces copies.

## 6. Question 3 : BERT affiné sur texte bruité

#### F2 des deux versions affinées, comparaison A (moyenne des trois graines)

| Dataset | Modèle | propre | typo | space | case | punct | number | mixed |
| ------------ | ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| ai4privacy | (a) propre | 0,986 | 0,873 | 0,848 | 0,986 | 0,978 | 0,971 | 0,915 |
|  | (b) propre + bruité | 0,983 | 0,951 | 0,940 | 0,983 | 0,976 | 0,981 | 0,962 |
|  | différence | *−0,003* | **+0,079** | **+0,092** | *−0,003* | *−0,003* | **+0,011** | **+0,047** |
| Nemotron-PII | (a) propre | 0,713 | 0,596 | 0,601 | 0,713 | 0,684 | 0,690 | 0,630 |
|  | (b) propre + bruité | 0,781 | 0,742 | 0,730 | 0,781 | 0,750 | 0,786 | 0,756 |
|  | différence | **+0,068** | **+0,145** | **+0,130** | **+0,068** | **+0,067** | **+0,096** | **+0,126** |
| Gretel | (a) propre | 0,508 | 0,417 | 0,425 | 0,508 | 0,509 | 0,509 | 0,461 |
|  | (b) propre + bruité | 0,534 | 0,499 | 0,496 | 0,534 | 0,548 | 0,536 | 0,521 |
|  | différence | **+0,026** | **+0,082** | **+0,071** | **+0,026** | **+0,039** | **+0,027** | **+0,060** |

Les écarts-types entre graines sont d'au plus 0,002 sur ai4privacy et 0,025 ailleurs. Partout
où une différence est établie, la plus basse graine de (b) est au-dessus de la plus haute graine
de (a) ; la marge la plus étroite est Gretel `punct` (0,002).

#### Perte, gain et coût, comparaison A, F2 [intervalle à 95 %]

| Dataset | Perte de (a) sur mixed | Rattrapé par (b) sur mixed | Part | Coût sur propre, (a) − (b) | Fuite sur mixed, (a) → (b) |
| ------------ | ---------------------- | -------------------------- | ----: | -------------------------- | -------------------------- |
| ai4privacy | 0,071 [0,067 ; 0,074] | +0,047 [0,044 ; 0,050] | 66 % | *+0,003* [0,001 ; 0,004] | *2,3 % → 2,0 %* |
| Nemotron-PII | 0,082 [0,074 ; 0,091] | +0,126 [0,117 ; 0,135] | 153 % | −0,068 [−0,075 ; −0,061] | 24,6 % → 12,9 % |
| Gretel | 0,047 [0,042 ; 0,051] | +0,060 [0,054 ; 0,065] | 128 % | −0,026 [−0,032 ; −0,021] | 43,1 % → 37,8 % |

La part dépasse 100 % hors domaine parce que (b) est aussi meilleure sur texte propre ; là, le
F2 de chaque version dit plus que la part. Sur ai4privacy, la part est de 70 % pour les fautes
de frappe, 67 % pour les espaces et 69 % pour les nombres. La version (b) perd encore sur texte
bruité, mais moins : sur ai4privacy mixed, 0,021 au lieu de 0,071.

#### Comparaison A-dates

| Dataset | Modèle | propre | typo | space | case | punct | number | mixed |
| ------------ | ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| ai4privacy | (a) propre | 0,987 | 0,881 | 0,857 | 0,987 | 0,980 | 0,950 | 0,914 |
|  | (b) propre + bruité | 0,985 | 0,956 | 0,945 | 0,985 | 0,978 | 0,982 | 0,965 |
|  | différence | *−0,002* | **+0,074** | **+0,088** | *−0,002* | *−0,003* | **+0,032** | **+0,051** |
| Nemotron-PII | (a) propre | 0,743 | 0,644 | 0,645 | 0,743 | 0,718 | 0,604 | 0,597 |
|  | (b) propre + bruité | 0,817 | 0,782 | 0,773 | 0,817 | 0,792 | 0,802 | 0,785 |
|  | différence | **+0,074** | **+0,139** | **+0,128** | **+0,074** | **+0,073** | **+0,199** | **+0,187** |
| Gretel | (a) propre | 0,492 | 0,423 | 0,427 | 0,492 | 0,491 | 0,440 | 0,430 |
|  | (b) propre + bruité | 0,519 | 0,489 | 0,489 | 0,519 | 0,527 | 0,513 | 0,506 |
|  | différence | **+0,027** | **+0,066** | **+0,062** | **+0,027** | **+0,035** | **+0,073** | **+0,076** |

Quand toutes les dates sont notées, les copies `number` montrent le plus grand gain : la
version propre seule manque les dates dont les séparateurs ont changé (Nemotron-PII 0,604), la
version bruitée beaucoup moins (0,802).

### 6.1 Pourquoi les deux versions affinées sont sous le BERT d'origine hors domaine

F2 par label sur les jeux de test propres, comparaison A (versions affinées : moyenne des trois
graines) :

| Dataset | Label | Valeurs du gold | BERT d'origine | (a) propre | (b) propre + bruité |
| ------------ | ----------- | --------------: | -------------: | ---------: | ------------------: |
| Gretel | ADDRESS | 2 322 | 0,80 | 0,13 | 0,13 |
| Gretel | NAME | 4 475 | 0,85 | 0,73 | 0,77 |
| Gretel | EMAIL | 837 | 0,85 | 0,70 | 0,81 |
| Gretel | CARD_NUMBER | 76 | 0,71 | 0,36 | 0,43 |
| Nemotron-PII | EMAIL | 1 091 | 0,95 | 0,63 | 0,82 |
| Nemotron-PII | CARD_NUMBER | 257 | 0,95 | 0,42 | 0,65 |
| Nemotron-PII | ADDRESS | 324 | 0,45 | 0,60 | 0,65 |

- **Adresses.** ai4privacy ne marque comme adresse que le numéro et la rue ; Gretel marque
  l'adresse entière. Entraînées sur ai4privacy, les deux versions apprennent les bords courts,
  et l'appariement strict échoue sur Gretel (0,80 à 0,13). C'est une différence de conventions
  entre datasets, pas un effet du bruit, et les copies bruitées n'y changent rien.
- **Données d'entraînement.** Le BERT d'origine a été entraîné sur des données qui incluent les
  documents PII finance de Gretel (question 1, analyse de contamination S3) ; ré-entraîné sur
  ai4privacy seul, il perd cet avantage.
- **E-mails et numéros de carte** baissent hors domaine pour les deux versions, et les copies
  bruitées divisent la baisse par deux (e-mails de Nemotron-PII : 0,63 à 0,82).
- Sans noms ni adresses (comparaison B), les deux versions affinées sont *au-dessus* du BERT
  d'origine sur Gretel (0,507 et 0,524 contre 0,459), ce qui concorde avec ce qui précède.

C'est pourquoi le proposal ne compare les deux versions affinées qu'entre elles : elles
partagent tout sauf les copies bruitées, alors que le BERT d'origine a été entraîné sur d'autres
données.

## 7. Conclusions

1. Le bruit coûte quelque chose à chaque détecteur, et les erreurs d'espacement coûtent le plus.
   Pour un filtre placé devant un modèle de langue, BERT est le plus fragile des deux détecteurs
   neuronaux ; GLiNER résiste mieux, sauf aux noms en minuscules.
2. L'entraînement sur copies bruitées marche dans son propre domaine : environ deux tiers de la
   perte sont rattrapés, sans coût notable sur texte propre.
3. Hors domaine, les copies bruitées semblent aussi jouer un rôle de régularisation : la version
   propre seule semble sur-apprendre l'orthographe propre d'ai4privacy (sa perte sur le dev
   bruité monte pendant l'entraînement), et la version bruitée est meilleure même sur texte
   propre.
4. Affiner sur un seul dataset, avec ses propres conventions de spans, ne suffit pas pour battre
   le BERT d'origine sur d'autres données ; un meilleur détecteur demanderait des données
   d'entraînement dont les bords de labels correspondent à ceux du texte visé.

## 8. Limites

- **Bruit synthétique.** Les modifications viennent d'un script, à un seul taux (0,10). Les
  vraies erreurs de frappe suivent une autre distribution, et d'autres taux n'ont pas été
  essayés.
- **Un seul domaine d'entraînement.** Les deux versions apprennent sur ai4privacy seul, donc les
  résultats hors domaine mêlent l'effet du bruit à celui des conventions de spans (section 6.1).
- **Graines.** Les tests couvrent la variation des exemples de test, les trois graines
  d'entraînement étant prises comme fixes ; ils ne disent rien d'autres entraînements. La
  vérification par graine (chaque graine de (b) au-dessus de chaque graine de (a) partout où une
  différence est établie) en couvre une partie.
- **Recouvrement.** 89 des 10 000 phrases de test d'ai4privacy (0,9 %) apparaissent aussi, mot
  pour mot, dans la partition train d'ai4privacy. Les deux versions les ont vues, donc la
  comparaison entre versions est juste, mais leurs scores dans le domaine sont un peu optimistes.
- **Pas d'optimisation.** Un seul jeu de réglages, choisi avant les entraînements, le même pour
  les deux versions. De meilleurs réglages peuvent exister pour l'une ou l'autre.
- **Appariement strict seulement.** Comme pour la question 1, une valeur trouvée avec un bord
  faux compte comme manquée ; les règles plus souples de l'analyse de sensibilité de la question
  1 n'ont pas été relancées sur les copies bruitées.

## 9. Reproduire les résultats

Les datasets et les modèles ne sont pas dans git (voir le README) ; les prédictions, scores,
fiches d'entraînement et intervalles y sont.

```bash
uv run python experiments/make_noisy.py                 # 36 jeux de test bruités (environ 10 s)
uv run python experiments/run_benchmark.py bert --versions noisy_typo noisy_space noisy_case \
    noisy_punct noisy_number noisy_mixed                 # aussi presidio, gliner ; et --dates
uv run python experiments/prepare_train.py              # données d'entraînement (environ 15 s)
CUDA_VISIBLE_DEVICES=0 uv run python experiments/train_bert.py clean --seed 0   # environ 2 min ; graines 0-2, clean et noisy
uv run python experiments/run_benchmark.py bert --model models/bert-clean-s0 --name bert-clean-s0 \
    --versions clean noisy_typo noisy_space noisy_case noisy_punct noisy_number noisy_mixed   # et --dates
uv run python experiments/score.py
uv run python experiments/summarize_noise.py            # results/noise/summary.md (--dates)
uv run python experiments/summarize_data.py             # results/noise/test_sets.md, results/training/summary.md
uv run python experiments/significance_noise.py         # results/analysis/significance_noise.md (environ 3 min)
uv run python experiments/results_tables.py             # les tableaux de docs/reports/results_tables/
```

| Fichier | Contenu |
| ------------------------------------------------ | ------------------------------------------------ |
| `benchmark/noise.py`, `tests/test_noise.py` | le bruit et ses tests |
| `results/noise/test_sets.md` | modifications de chaque jeu de test bruité |
| `results/training/` | la fiche de chaque entraînement, et `summary.md` |
| `results/predictions/<model>/<dataset>/` | toutes les sorties |
| `results/scores.md`, `results/noise/summary*.md` | les scores |
| `results/analysis/significance_noise.md` | tous les intervalles et tests |
