# PIIGuard, question 1 : les détecteurs prêts à l'emploi trouvent-ils les PII dans un texte propre ?

*Rapport d'analyse. Équipe PIIGuard (Louis Lenoble, Sunho Cha, Suseong Kim), CSED490E, 20 novembre 2026.*

## Résumé

Nous avons comparé trois détecteurs de PII prêts à l'emploi sur trois jeux de test anglais
annotés : Presidio avec ses seules règles, un petit modèle BERT entraîné pour les PII
(Gravitee) et GLiNER (urchade, version PII). Comme dans le proposal, les scores principaux sont le F1 et le F2
micro en appariement strict (*strict matching*). Nous donnons aussi la part des PII du gold qui
atteindraient le modèle de langue sans masque (taux de fuite). Chaque score a un intervalle de
confiance à 95 %, et chaque différence entre deux détecteurs a été testée par un test apparié,
corrigé pour comparaisons multiples.

1. **Sur les six labels du proposal**, GLiNER a le meilleur F2 sur ai4privacy (0,867) et
   Nemotron-PII (0,823). Sur Gretel, BERT (0,718) et GLiNER (0,713) ne se distinguent pas en F2 ; en F1,
   GLiNER est devant (0,637 contre 0,572).
   Presidio obtient 0,16 à 0,43 : ses règles ne trouvent ni noms ni adresses.
2. **Sur les labels que Presidio sait trouver** (e-mail, téléphone, date de naissance, numéro de
   carte), Presidio fait jeu égal en F2 avec BERT sur Nemotron-PII et le devance sur Gretel. Cela
   vient du label de date : ni Presidio ni BERT n'ont de classe « date de naissance », seulement
   une classe pour toute date, que nous avons dû compter comme date de naissance.
3. **Compter toute date comme PII** (comparaison C, décidée par l'équipe) supprime cet artefact.
   La précision des dates de Presidio passe de 0,22 à 0,96 sur Nemotron-PII, celle de BERT de
   0,02 à 0,58 sur Gretel. GLiNER est alors en tête sur ai4privacy et Nemotron-PII, et BERT
   nettement en tête sur Gretel (0,787 contre 0,595).
4. **Taux de fuite** : GLiNER fuit le moins sur ai4privacy (6,1 %), BERT le moins sur Gretel
   (8,3 %). Sur Nemotron-PII, la différence entre les deux n'est pas établie.
5. **Le F1 confirme les ordres du F2.** Il n'en inverse aucun et départage les quatre paires que
   le F2 laisse à égalité, chaque fois en faveur du détecteur le plus précis : GLiNER devant BERT
   sur Gretel (A) et ai4privacy (B), Presidio devant BERT sur Nemotron-PII (B et B-dates).
6. La chaîne de mesure a passé tous les contrôles de validité (score parfait du gold,
   comptabilité exacte, sorties identiques quand on relance). Aucun classement établi ne change
   quand le seuil de GLiNER varie entre 0,25 et 0,60.

---

## 1. La question

Le proposal pose trois questions. Ce rapport répond à la première seulement :

> **Q1.** Sur un texte propre, comment se comparent un système à règles (Presidio, sans aucun
> modèle neuronal), un BERT fine-tuné (Gravitee) et GLiNER pour trouver les six labels de PII
> du proposal ?

La question 2 (les mêmes détecteurs sur des copies bruitées des jeux de test) et la question 3
(BERT après fine-tuning sur du texte bruité) réutilisent les données, le code et la notation
décrits ici.

## 2. Données

### 2.1 Datasets et jeux de test

| Dataset | Split utilisé | Enregistrements anglais | Notre jeu de test | Longueur médiane | Type de texte |
|---|---|---|---|---|---|
| ai4privacy open-pii-masking-500k | validation (pas de split test) | 30 160 | 10 000 phrases | 102 caractères | phrases courtes générées et lignes de chat |
| Nemotron-PII | test | 100 000 (50 000 locale `us` + 50 000 `intl`) | 2 000 documents | 745 caractères | formulaires, factures, e-mails, notes générés |
| Gretel synthetic PII finance | test, enregistrements anglais | 2 962 (sur 5 594) | les 2 962 documents | 1 414 caractères | documents financiers et juridiques générés |

Chaque dataset est téléchargé à une révision figée. Chaque jeu de test est un échantillon
aléatoire tiré avec la graine 0 : chacun peut ainsi refaire exactement les mêmes expériences. Les tailles ont été fixées après avoir chronométré les détecteurs, pour que tout
le plan (questions 1 à 3) tienne sur des processeurs de portable.

Les enregistrements sans aucun de nos labels sont gardés (1 493, 415 et 711) : ils montrent si
un détecteur masque du texte qui n'est pas du PII.

### 2.2 Labels

Les labels du gold sont ramenés aux six labels du proposal :

| Notre label | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| NAME | GIVENNAME, SURNAME | first_name, last_name | name, first_name, last_name |
| EMAIL | EMAIL | email | email |
| PHONE | TELEPHONENUM | phone_number | phone_number |
| ADDRESS | BUILDINGNUM, STREET | street_address | street_address |
| DATE_OF_BIRTH | (aucun) | date_of_birth | date_of_birth |
| CARD_NUMBER | CREDITCARDNUMBER | credit_debit_card | credit_card_number |
| DATE (comparaison C) | DATE | date, date_of_birth, date_time | date, date_of_birth, date_time |

Les autres labels (ville, heure, âge, numéros de compte…) ne sont pas notés. ai4privacy n'a pas
de label de date de naissance : DATE_OF_BIRTH n'y est pas noté. DATE ne compte que les dates :
les heures seules (`TIME`, `time`) sont exclues, alors qu'une date suivie d'une heure
(« 2025-07-13T17:43:03 ») compte, comme ai4privacy lui-même range ces valeurs parmi les dates.

Spans gold par label dans nos jeux de test :

| Label | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| NAME | 8 748 | 2 053 | 4 475 |
| EMAIL | 1 185 | 1 091 | 837 |
| PHONE | 1 567 | 464 | 531 |
| ADDRESS | 1 533 | 324 | 2 322 |
| DATE_OF_BIRTH | - | 363 | 143 |
| CARD_NUMBER | 185 | 257 | 76 |
| DATE (comparaison C) | 1 387 | 2 126 | 5 768 |

### 2.3 Nettoyage des spans

Les quatre mêmes règles s'appliquent aux spans du gold et aux sorties de chaque détecteur, pour
que les deux côtés soient comparés de la même façon :

1. les bornes d'un span se resserrent en sautant les espaces, et une paire de crochets ou de
   guillemets qui entoure tout le span (`[x@y.com]` devient `x@y.com`, mais `(555) 123-4567`
   garde sa parenthèse) ;
2. un titre en tête est retiré des spans NAME (Mr, Mrs, Ms, Miss, Mx, Dr, Prof, Sir, Madam, et
   Mister, Madame, Master, Mayoress, Mstr, qu'ai4privacy annote comme titres aussi souvent que
   Mr) ;
3. spans qui se chevauchent : celui qui commence en premier est gardé ;
4. deux spans de même label séparés seulement par des espaces sont réunis (« Minji » + « Park ») ;
   une virgule ou un saut de ligne les laisse séparés.

La règle des crochets et les cinq titres supplémentaires ont été ajoutés après examen des
erreurs, parce qu'il s'agissait de conventions d'annotation et non d'erreurs de détection :
dans Gretel, les e-mails sont souvent annotés avec leurs crochets Markdown ; dans ai4privacy,
« Mister Maffongelli » face au gold « Maffongelli » causait environ 430 erreurs de GLiNER et
370 de BERT. Aucune des deux règles ne change les jeux de test ; elles changent seulement la comparaison
des détections.

## 3. Détecteurs

| | Presidio | BERT | GLiNER |
|---|---|---|---|
| Modèle | presidio-analyzer 2.2.366, reconnaisseurs à motifs seulement | `gravitee-io/bert-small-pii-detection`, 28,5 M de paramètres | `urchade/gliner_multi_pii-v1`, environ 289 M de paramètres |
| Comment il trouve le PII | expressions régulières, une bibliothèque de numéros de téléphone, une somme de contrôle pour les cartes | une étiquette par token (25 classes) | note chaque span pour des noms de labels écrits en mots |
| Nos labels | EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, DATE_TIME | PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME, CREDIT_CARD (19 autres classes écartées) | « person », « email », « phone number », « street address », « date of birth » (ou « date » en comparaison C), « credit card number » |
| Réglages | par défaut, aucun modèle de langue chargé | par défaut ; lit le texte en minuscules | seuil 0,35 |
| Textes longs | pas de limite | fenêtres de 256 tokens qui se chevauchent de 64 | fenêtres de 384 mots qui se chevauchent de 64 |

- **Presidio, la baseline, sans modèle neuronal.** Presidio est notre baseline : un système
  uniquement à base de règles et très rapide (voir le temps de calcul ci-dessous), qui permet de
  montrer les différences avec les deux modèles de deep learning. Presidio fait d'habitude tourner un modèle spaCy pour
  trouver noms et lieux. Nous lui donnons un moteur de langue qui ne charge aucun modèle : seules
  les règles autonomes tournent. Il ne peut donc pas trouver de noms ni d'adresses, par
  construction.
- **Correspondance des labels.** La classe DATE_TIME de Presidio et de BERT couvre toute date ou
  heure ; elle est comptée comme DATE_OF_BIRTH dans les comparaisons A et B, et comme DATE dans
  la comparaison C. La classe LOCATION de BERT couvre aussi villes et pays, et est comptée comme
  ADDRESS.
- **Textes longs.** Sans fenêtres, BERT échouerait sur les textes de plus de 512 tokens (environ
  6 % des documents de Nemotron-PII et de Gretel), et GLiNER ignorerait sans prévenir tout ce qui
  suit son 384e mot (11 % des documents de Nemotron-PII, 25 % de ceux de Gretel).
- **Le seuil de GLiNER** a été choisi une fois, avant tout test, comme le prévoit le proposal :
  la valeur au meilleur F2 micro sur 300 phrases du split *train* d'ai4privacy. Le F2 bouge à
  peine entre 0,25 et 0,60 (0,859 à 0,861) ; 0,35 était le meilleur.
- **Temps de calcul** sur un processeur de portable (8 cœurs), pour les trois jeux de test :
  Presidio 20 s, BERT 6 min, GLiNER 1 h 20 (1 h 33 avec le label « date »).

## 4. Notation

- **Appariement strict** (choix du proposal). Une détection n'est correcte (vrai positif, TP) que si son début, sa fin
  et son label sont exactement ceux d'un span gold. Toute autre détection est un faux positif
  (FP) ; tout span gold sans correspondance exacte est un faux négatif (FN). Un span partiel
  compte donc pour un FP et un FN : trouver « Minji » dans « Minji Park » envoie quand même « Park » au modèle de langue.
  C'est la règle habituelle en NER, ce qui rend nos scores comparables à d'autres travaux. Elle
  est en revanche sévère sur les bornes : masquer « [x@y.com] » au lieu de « x@y.com » compte
  comme une erreur alors que rien ne fuit. Le taux de fuite ci-dessous et l'analyse S1 mesurent
  ce que ce choix change.
- **Scores**, calculés par label puis en moyenne micro (TP, FP et FN additionnés sur les labels
  d'abord) :

  ```
  P  = TP / (TP + FP)        R  = TP / (TP + FN)
  F1 = 2·P·R / (P + R)       F2 = 5·P·R / (4·P + R)
  ```

  Comme dans le proposal, F1 et F2 sont les deux scores principaux : le F1 est le score usuel en
  NER, et le F2 compte le rappel deux fois plus que la précision, car pour un filtre de vie
  privée, une fuite coûte plus qu'un masque en trop.
- **Taux de fuite** : la part des spans gold que les détections, quel que soit leur label, ne
  couvrent pas entièrement. Ces valeurs atteindraient le modèle de langue en clair, au moins en
  partie.
- Un score qui n'existe pas (par exemple la précision d'un détecteur qui n'a rien trouvé) est
  noté « n/a », pas 0.

## 5. Choix de l'équipe

| Quand | Choix | Pourquoi |
|---|---|---|
| 3 nov. | Garder les trois datasets du proposal | des types de texte différents (phrases, formulaires, documents financiers) |
| 3 nov. | Tous les benchmarks sur des portables sans GPU | les ordinateurs dont dispose l'équipe |
| 3 nov. | Presidio en règles seules, sans spaCy | c'est la baseline : un système uniquement à base de règles et très rapide, qui permet de montrer les différences avec les modèles de deep learning ; Presidio peut tourner avec un moteur qui ne charge aucun modèle |
| 3 nov. | Seuil de GLiNER choisi sur des données d'entraînement, comme dans le proposal | aucun réglage sur les jeux de test |
| 3 nov. | Fenêtres pour les textes longs, tailles tirées des limites de chaque modèle | sinon les textes sont tronqués, sans prévenir pour GLiNER |
| 3 nov. | Scores indéfinis notés n/a, pas 0 | un score absent n'est pas un mauvais score |
| 3 nov. | Règle des crochets dans le nettoyage des spans | e-mails annotés avec leurs crochets Markdown dans Gretel |
| 3 nov. | Tailles des jeux de test 10 000 / 2 000 / 2 962, après chronométrage | tout le plan doit tenir sur des processeurs de portable |
| 4 nov. | Cinq titres de plus dans le nettoyage des spans | la liste de titres d'ai4privacy elle-même |
| 4 nov. | Comparaison B, sans NAME ni ADDRESS | les règles de Presidio ne peuvent pas les trouver |
| 4 au 20 nov. | Comparaison C : toute date en DATE, dates seulement, pas les heures | Presidio et BERT n'ont pas de classe date de naissance (décidé en réunion d'équipe) |
| 20 nov. | Garder le taux de fuite à côté des scores stricts | il mesure ce qui atteint le modèle de langue |
| 20 nov. | Plan d'analyse écrit avant tout score de la comparaison C ; hypothèses et plus petit écart qui compte (0,01 de F2) validés | pour que la comparaison C soit un test, pas une observation après coup |

Les quatre comparaisons utilisées dans la suite :

| Comparaison | Labels notés |
|---|---|
| **A** | les six labels du proposal |
| **B** | EMAIL, PHONE, DATE_OF_BIRTH, CARD_NUMBER (sans NAME ni ADDRESS) |
| **A-dates** | comme A, avec DATE à la place de DATE_OF_BIRTH |
| **B-dates** | comme B, avec DATE à la place de DATE_OF_BIRTH |

---

## 6. Le plan d'analyse, point par point

### 6.0 Règles générales

*Cette première partie du plan fixe les règles communes à toute l'analyse. Elle distingue deux
sortes d'analyses : les analyses confirmatoires, dont la question et la règle de réponse sont
fixées avant de voir les résultats, et les analyses exploratoires, qui cherchent ce qui ressort
des résultats et sont signalées comme telles. Elle impose ensuite trois exigences : aucun score
n'est donné sans son intervalle à 95 % ; aucun détecteur n'est dit meilleur qu'un autre sans un
test apparié ; chaque
résultat peut être recalculé par une commande.*

Les paragraphes ci-dessous définissent ces termes, dans l'ordre où ils apparaissent dans la
suite du rapport.

**Confirmatoire ou exploratoire.** Une analyse est confirmatoire quand la question et la règle
pour y répondre sont fixées avant de voir les résultats : elle permet de conclure. Une analyse
exploratoire cherche dans les résultats ce qui ressort : elle donne des explications et des
pistes, pas des preuves. La partie confirmatoire comprend les hypothèses de la section 6.2 (cinq
sur A et B, tirées du proposal ; quatre sur C) et les 72 tests appariés de la section 6.5 :
4 comparaisons × 3 datasets × 3 paires de détecteurs, sur le F2 et sur le taux de fuite. Les 36
tests sur le F1 ont été ajoutés ensuite, pour suivre le proposal. Les analyses de
sensibilité (6.6) sont exploratoires.

**Pourquoi des statistiques.** Chaque jeu de test est un échantillon : avec d'autres documents du
même dataset, les scores seraient un peu différents. Les outils suivants mesurent cette
incertitude.

- **Intervalle à 95 % (bootstrap).** C'est la plage des valeurs plausibles du score qu'aurait le
  détecteur sur un très grand nombre de documents du même type. On le calcule en simulant
  d'autres jeux de test à partir du nôtre : on tire au hasard, avec remise, autant de documents
  qu'en contient le jeu (certains sortent deux fois, d'autres pas du tout), on recalcule le
  score, et on répète 10 000 fois. Le 250e et le 9 750e des 10 000 scores triés bornent
  l'intervalle. Un intervalle étroit signale un score stable (ai4privacy, 10 000 documents) ; un
  intervalle large, un score moins sûr (Gretel). Les mêmes tirages servent aux trois détecteurs,
  ce qui donne aussi l'intervalle de l'écart entre deux détecteurs.
- **Test apparié (randomisation approchée).** Il dit si l'écart entre deux détecteurs pourrait
  venir du hasard. « Apparié » : les deux détecteurs sont comparés sur les mêmes documents, si
  bien que la difficulté propre à chaque document s'annule. Le test suppose d'abord que les deux
  détecteurs sont interchangeables ; dans ce cas, échanger leurs sorties sur un document ne
  change rien en moyenne. On échange donc leurs sorties document par document, à pile ou face,
  on recalcule l'écart, et on répète 10 000 fois. La p-value est la part des tours où l'écart
  simulé est au moins aussi grand que l'écart observé : plus elle est petite, moins le hasard
  suffit à expliquer l'écart.
- **Correction de Holm.** Avec 36 tests au seuil de 5 %, environ deux sortiraient
  « significatifs » par hasard, même sans aucune différence réelle. La correction relève les
  p-values en conséquence : la plus petite est multipliée par 36, la suivante par 35, et ainsi de
  suite. Avec 10 000 tours, la plus petite p-value corrigée possible est 0,0036.
- **Documents liés.** Les deux locales d'un même document Nemotron-PII et les textes présents
  deux fois ne sont pas indépendants : ils sont tirés et échangés ensemble.

**Règle de décision.** Un détecteur est dit meilleur qu'un autre seulement si trois conditions
sont réunies : la p-value corrigée est sous 0,05, l'intervalle à 95 % de l'écart exclut 0, et
l'écart vaut au moins 0,01 (en dessous, il n'a pas d'intérêt pratique). Sinon, la différence est
dite « non établie » : les deux détecteurs ne sont pas forcément égaux, mais nos données ne
permettent pas de les départager.

Toutes les commandes sont en section 9.

### 6.1 Ce qui est figé avant l'analyse

*Code, versions des données, versions des modèles, réglages, métriques et paramètres
statistiques.*

| Élément | Valeur |
|---|---|
| Modèles | BERT révision f8c27a85, GLiNER révision 1fcf13e8, presidio-analyzer 2.2.366 |
| Score principal | F2 micro, appariement strict (F1 ajouté ensuite) |
| Statistiques | intervalles à 95 % ; 10 000 tirages bootstrap et 10 000 tours de permutation ; graine 0 ; α = 0,05 ; correction de Holm ; plus petit écart qui compte 0,01 |
| Code | commité à la fin de la question 1, avec ce rapport |

### 6.2 Hypothèses

*Sur A et B : tirées des attentes écrites dans le proposal, avant tout score. Sur C : écrites et
validées avant tout score de la comparaison C.*

**Sur les comparaisons A et B**

| Hypothèse | Attente du proposal | Résultat |
|---|---|---|
| **H-AB1.** En A, Presidio a le F2 le plus bas et le taux de fuite le plus haut des trois détecteurs, sur les trois datasets. | Les règles ne peuvent pas trouver les noms et les adresses, qui n'ont pas de format fixe. | **Vérifiée.** F2 0,157 / 0,431 / 0,171 et fuite 0,868 / 0,572 / 0,839 (ai4privacy / Nemotron-PII / Gretel) ; tous ses écarts avec les deux modèles sont établis (6.5). |
| **H-AB2.** Sur les labels à format fixe (comparaison B), Presidio se rapproche nettement des modèles. | Les règles couvrent les formats fixes (e-mails, téléphones, cartes, dates) ; ce qui leur échappe, ce sont les noms et les adresses. | **Vérifiée.** L'écart de F2 avec le meilleur modèle passe de 0,71 / 0,39 / 0,55 en A à 0,28 / 0,20 / 0,17 en B. Les formats ne se valent pas : Presidio a le meilleur F2 des trois sur les e-mails (0,990 / 0,988 / 0,960), le plus bas sur les téléphones (0,371 / 0,727 / 0,542). |
| **H-AB3.** La précision de BERT est basse sur ADDRESS et DATE_OF_BIRTH. | Ses classes LOCATION et DATE_TIME couvrent tout lieu et toute date, plus large que nos labels, ce qui baisse la précision. | **Vérifiée, sauf ADDRESS sur Gretel.** Précision de BERT sur ADDRESS : 0,281 / 0,206 / 0,667 ; sur DATE_OF_BIRTH : 0,111 sur Nemotron-PII, 0,018 sur Gretel (ai4privacy n'a pas ce label). Ce sont ses deux labels les moins précis, sauf ADDRESS sur Gretel, où ses numéros de carte sont moins précis (0,460). GLiNER, à qui l'on demande exactement nos labels, est plus précis sur ces deux labels partout (ADDRESS 0,541 / 0,655 / 0,755 ; DATE_OF_BIRTH 0,935 / 0,617). |
| **H-AB4.** Face à GLiNER, BERT se place mieux sur Gretel que sur les deux autres datasets. | BERT a été entraîné entre autres sur Gretel PII finance : son score sur Gretel n'est pas un résultat hors domaine. | **Vérifiée en A, pas en B.** Écart de F2 BERT − GLiNER en A : −0,071 / −0,061 / +0,005 ; en B : −0,006 / −0,203 / −0,278. En B, le label de date l'emporte : avec DATE (B-dates), BERT passe devant sur Gretel (+0,277). S3 montre que cet avantage ne vient pas de documents de test vus à l'entraînement. |
| **H-AB5.** Pas de sens attendu entre BERT et GLiNER sur ai4privacy et Nemotron-PII. | Le proposal ne tranche pas entre les deux modèles. | GLiNER est meilleur sur les deux en A (F2 +0,071 et +0,061, écarts établis) et sur Nemotron-PII en B (+0,203) ; sur ai4privacy en B, égalité en F2, GLiNER devant en F1. |

**Sur la comparaison C**

| Hypothèse | Résultat |
|---|---|
| **H-C1.** De A à A-dates, la précision du label de date de Presidio et de BERT monte fortement sur Nemotron-PII et Gretel. | **Vérifiée.** Presidio : 0,224 → 0,962 (Nemotron-PII), 0,038 → 0,519 (Gretel). BERT : 0,111 → 0,471 et 0,018 → 0,583. |
| **H-C2.** Sur Gretel, le rappel des dates de Presidio est plus bas que celui de BERT (ses règles ne connaissent que les formats numériques). | **Vérifiée.** 0,273 contre 0,758. |
| **H-C3.** Pas de sens attendu entre GLiNER (à qui l'on demande « date ») et BERT sur les dates. | GLiNER est meilleur sur ai4privacy (F2 0,933 contre 0,737) et Nemotron-PII (0,790 contre 0,655), moins bon sur Gretel (0,397 contre 0,715) ; les intervalles ne se recouvrent pas. |
| **H-C4.** Les autres labels de Presidio et de BERT sont identiques en A et en A-dates ; ceux de GLiNER peuvent bouger un peu. | **Vérifiée.** Comptes identiques pour Presidio et BERT ; les autres labels de GLiNER bougent de 0,023 de F2 au plus (numéros de carte de Gretel, 76 spans gold). |

### 6.3 Contrôles de validité

*Avant de lire un seul score, vérifier que la mesure elle-même est juste.*

Avant de lire les scores, nous avons vérifié que la mesure elle-même était juste : chaque jeu de
test a une prédiction pour chaque exemple, le gold donné comme prédiction obtient un score
parfait, les comptes de spans sont exacts, et un second passage des détecteurs donne des sorties
identiques. Tous ces contrôles sont réussis.

### 6.4 Scores avec intervalles à 95 %

*Chaque score avec un intervalle à 95 % issu d'un bootstrap apparié par groupes. Les exemples
qui ne sont pas indépendants (les deux locales d'un document Nemotron-PII, les textes
identiques) sont tirés ensemble.*

L'unité tirée est un groupe d'exemples liés : 14 documents Nemotron-PII ont leur jumeau dans
l'autre locale dans notre échantillon, et 74 textes d'ai4privacy et 3 de Gretel apparaissent
deux fois. Les mêmes 10 000 tirages servent aux trois détecteurs d'un jeu de test.

F1 micro, F2 micro et taux de fuite, chacun avec son intervalle à 95 % :

**Comparaison A (six labels)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0,223 [0,214 ; 0,232] | 0,768 [0,761 ; 0,774] | **0,853** [0,847 ; 0,858] |
|  | F2 | 0,157 [0,150 ; 0,164] | 0,796 [0,790 ; 0,803] | **0,867** [0,862 ; 0,873] |
|  | Taux de fuite | 0,868 [0,862 ; 0,874] | 0,113 [0,108 ; 0,119] | **0,061** [0,056 ; 0,065] |
| Nemotron-PII | F1 | 0,453 [0,437 ; 0,468] | 0,628 [0,612 ; 0,643] | **0,763** [0,747 ; 0,777] |
|  | F2 | 0,431 [0,415 ; 0,446] | 0,762 [0,749 ; 0,774] | **0,823** [0,811 ; 0,835] |
|  | Taux de fuite | 0,572 [0,555 ; 0,588] | 0,088 [0,076 ; 0,100] | 0,101 [0,089 ; 0,113] |
| Gretel | F1 | 0,194 [0,180 ; 0,208] | 0,572 [0,560 ; 0,583] | **0,637** [0,622 ; 0,652] |
|  | F2 | 0,171 [0,159 ; 0,184] | 0,718 [0,707 ; 0,728] | 0,713 [0,701 ; 0,725] |
|  | Taux de fuite | 0,839 [0,826 ; 0,850] | **0,083** [0,072 ; 0,095] | 0,195 [0,183 ; 0,207] |

**Comparaison B (sans NAME ni ADDRESS)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0,659 [0,644 ; 0,674] | 0,846 [0,834 ; 0,858] | **0,881** [0,872 ; 0,891] |
|  | F2 | 0,616 [0,600 ; 0,632] | 0,891 [0,881 ; 0,901] | 0,898 [0,888 ; 0,907] |
|  | Taux de fuite | 0,406 [0,389 ; 0,423] | **0,054** [0,046 ; 0,063] | 0,071 [0,062 ; 0,080] |
| Nemotron-PII | F1 | 0,632 [0,613 ; 0,650] | 0,568 [0,547 ; 0,589] | **0,935** [0,926 ; 0,944] |
|  | F2 | 0,757 [0,741 ; 0,773] | 0,757 [0,742 ; 0,772] | **0,960** [0,953 ; 0,967] |
|  | Taux de fuite | 0,103 [0,089 ; 0,118] | 0,024 [0,017 ; 0,032] | 0,015 [0,009 ; 0,021] |
| Gretel | F1 | 0,385 [0,359 ; 0,410] | 0,264 [0,247 ; 0,281] | **0,698** [0,677 ; 0,720] |
|  | F2 | 0,569 [0,543 ; 0,595] | 0,459 [0,437 ; 0,480] | **0,737** [0,716 ; 0,757] |
|  | Taux de fuite | 0,157 [0,137 ; 0,178] | **0,030** [0,020 ; 0,040] | 0,205 [0,182 ; 0,230] |

**Comparaison A-dates (six labels, DATE à la place de DATE_OF_BIRTH)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0,251 [0,242 ; 0,260] | 0,738 [0,732 ; 0,745] | **0,853** [0,848 ; 0,858] |
|  | F2 | 0,178 [0,172 ; 0,185] | 0,790 [0,783 ; 0,796] | **0,872** [0,867 ; 0,877] |
|  | Taux de fuite | 0,850 [0,844 ; 0,855] | 0,106 [0,101 ; 0,111] | **0,057** [0,053 ; 0,061] |
| Nemotron-PII | F1 | 0,610 [0,596 ; 0,625] | 0,713 [0,699 ; 0,727] | **0,762** [0,749 ; 0,775] |
|  | F2 | 0,532 [0,516 ; 0,548] | 0,778 [0,764 ; 0,791] | **0,801** [0,788 ; 0,813] |
|  | Taux de fuite | 0,501 [0,485 ; 0,517] | 0,141 [0,127 ; 0,156] | 0,137 [0,123 ; 0,152] |
| Gretel | F1 | 0,289 [0,273 ; 0,304] | **0,741** [0,732 ; 0,751] | 0,573 [0,560 ; 0,585] |
|  | F2 | 0,227 [0,214 ; 0,240] | **0,787** [0,777 ; 0,796] | 0,595 [0,583 ; 0,606] |
|  | Taux de fuite | 0,796 [0,784 ; 0,808] | **0,090** [0,082 ; 0,099] | 0,355 [0,343 ; 0,367] |

**Comparaison B-dates (sans NAME ni ADDRESS, DATE à la place de DATE_OF_BIRTH)**

| Dataset | Score | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | F1 | 0,615 [0,602 ; 0,629] | 0,726 [0,716 ; 0,737] | **0,887** [0,879 ; 0,894] |
|  | F2 | 0,544 [0,529 ; 0,558] | 0,835 [0,826 ; 0,843] | **0,909** [0,902 ; 0,916] |
|  | Taux de fuite | 0,493 [0,478 ; 0,507] | 0,049 [0,043 ; 0,056] | 0,057 [0,050 ; 0,064] |
| Nemotron-PII | F1 | 0,797 [0,783 ; 0,810] | 0,716 [0,697 ; 0,734] | **0,873** [0,859 ; 0,886] |
|  | F2 | 0,791 [0,774 ; 0,807] | 0,784 [0,766 ; 0,801] | **0,868** [0,850 ; 0,885] |
|  | Taux de fuite | 0,200 [0,182 ; 0,219] | 0,139 [0,121 ; 0,159] | **0,110** [0,092 ; 0,130] |
| Gretel | F1 | 0,445 [0,424 ; 0,466] | **0,690** [0,675 ; 0,704] | 0,502 [0,485 ; 0,519] |
|  | F2 | 0,408 [0,387 ; 0,429] | **0,744** [0,730 ; 0,758] | 0,467 [0,450 ; 0,484] |
|  | Taux de fuite | 0,606 [0,585 ; 0,628] | **0,084** [0,076 ; 0,094] | 0,519 [0,501 ; 0,536] |

En gras : une valeur meilleure que les deux autres, avec des différences établies (section 6.5).
Le F2 par label est en annexe A. Les labels de moins de 100 spans gold (numéros de carte de
Gretel, 76) ne donnent que des scores indicatifs.

### 6.5 Comparaisons appariées

*Pour chaque paire de détecteurs sur un même jeu de test : la différence de F1 micro, de F2 micro
et de taux de fuite, son intervalle à 95 %, un test de randomisation approchée apparié et bilatéral
(10 000 tours), la correction de Holm sur les 36 tests de chaque famille. Un détecteur n'est dit
meilleur que si la p-value corrigée est sous 0,05, l'intervalle de la différence exclut 0, et la
différence vaut au moins 0,01.*

Le test de randomisation échange les sorties des deux détecteurs groupe par groupe avec une
probabilité 1/2 et compte combien de fois la différence est au moins aussi grande que celle
observée. Avec 10 000 tours, la plus petite p-value possible vaut environ 0,0001, que la
correction de Holm porte à 0,0036 pour les tests les plus significatifs.

**F2 micro : 32 des 36 différences sont établies.** Les quatre qui ne le sont pas :

| Comparaison | Dataset | Paire | Différence [intervalle à 95 %] | p corrigée |
|---|---|---|---|---|
| A | Gretel | BERT − GLiNER | 0,005 [−0,008 ; 0,018] | 1,00 |
| B | ai4privacy | BERT − GLiNER | −0,006 [−0,017 ; 0,005] | 1,00 |
| B | Nemotron-PII | Presidio − BERT | 0,000 [−0,017 ; 0,017] | 1,00 |
| B-dates | Nemotron-PII | Presidio − BERT | 0,007 [−0,015 ; 0,028] | 1,00 |

Dans ces quatre cas, l'écart est inférieur à 0,01, son intervalle contient 0 et la p-value
corrigée vaut 1 : on ne peut pas dire lequel des deux détecteurs est meilleur. Cela ne veut pas
dire qu'ils sont égaux, mais l'intervalle borne l'écart possible : il reste très probablement
sous 0,03 en valeur absolue. Les deux détecteurs sont donc proches.

L'ordre qui en résulte sur chaque jeu de test (« = » signifie différence non établie) :

| Comparaison | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| A | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | BERT = GLiNER > Presidio |
| B | GLiNER = BERT > Presidio | GLiNER > Presidio = BERT | GLiNER > Presidio > BERT |
| A-dates | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | BERT > GLiNER > Presidio |
| B-dates | GLiNER > BERT > Presidio | GLiNER > Presidio = BERT | BERT > GLiNER > Presidio |

La plus petite différence de F2 établie est celle de GLiNER contre BERT sur Nemotron-PII en
A-dates : 0,023 [0,009 ; 0,037], p corrigée = 0,010.

**Taux de fuite : 31 des 36 différences sont établies.** Les cinq qui ne le sont pas :

| Comparaison | Dataset | Paire | Différence [intervalle à 95 %] | p corrigée |
|---|---|---|---|---|
| A | Nemotron-PII | BERT − GLiNER | −0,014 [−0,027 ; −0,000] | 0,16 |
| B | Nemotron-PII | BERT − GLiNER | 0,010 [0,002 ; 0,018] | 0,09 |
| B | Gretel | Presidio − GLiNER | −0,049 [−0,084 ; −0,014] | 0,05 |
| A-dates | Nemotron-PII | BERT − GLiNER | 0,005 [−0,011 ; 0,020] | 0,58 |
| B-dates | ai4privacy | BERT − GLiNER | −0,008 [−0,017 ; 0,001] | 0,17 |

Partout ailleurs, Presidio fuit plus que les deux modèles. Entre les deux modèles, GLiNER fuit
moins sur ai4privacy en A et A-dates et sur Nemotron-PII en B-dates ; BERT fuit moins sur Gretel
dans les quatre comparaisons et sur ai4privacy en B.

L'ordre qui en résulte pour le taux de fuite (« > » signifie « laisse fuir moins », « = »
différence non établie) :

| Comparaison | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| A | GLiNER > BERT > Presidio | BERT = GLiNER > Presidio | BERT > GLiNER > Presidio |
| B | BERT > GLiNER > Presidio | GLiNER = BERT > Presidio | BERT > Presidio = GLiNER |
| A-dates | GLiNER > BERT > Presidio | GLiNER = BERT > Presidio | BERT > GLiNER > Presidio |
| B-dates | BERT = GLiNER > Presidio | GLiNER > BERT > Presidio | BERT > GLiNER > Presidio |

Un meilleur F2 ne garantit pas moins de fuites : en A sur Nemotron-PII, GLiNER a le meilleur F2
mais ne fuit pas moins que BERT ; en B sur ai4privacy, les deux sont à égalité en F2 mais BERT
fuit moins. Sur Gretel, BERT fuit le moins dans les quatre comparaisons, y compris en B, où
GLiNER a le meilleur F2.

**F1 micro : les 36 différences sont établies.** Le F1 n'inverse aucun ordre du F2. Il départage
les quatre paires que le F2 laisse à égalité, chaque fois en faveur du détecteur le plus précis,
alors que le F2 compense par le meilleur rappel de l'autre :

| Comparaison | Dataset | Paire | Différence de F1 [intervalle à 95 %] | Précision | Rappel |
|---|---|---|---|---|---|
| A | Gretel | BERT − GLiNER | −0,065 [−0,080 ; −0,051] | 0,427 / 0,541 | 0,865 / 0,774 |
| B | ai4privacy | BERT − GLiNER | −0,035 [−0,048 ; −0,023] | 0,780 / 0,855 | 0,924 / 0,909 |
| B | Nemotron-PII | Presidio − BERT | 0,063 [0,045 ; 0,083] | 0,495 / 0,401 | 0,873 / 0,973 |
| B-dates | Nemotron-PII | Presidio − BERT | 0,081 [0,060 ; 0,102] | 0,807 / 0,625 | 0,787 / 0,837 |

Précision et rappel sont donnés dans l'ordre de la paire. La p-value corrigée vaut 0,0036 pour
les quatre.

L'ordre qui en résulte en F1 (toutes les différences sont établies) :

| Comparaison | ai4privacy | Nemotron-PII | Gretel |
|---|---|---|---|
| A | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio |
| B | GLiNER > BERT > Presidio | GLiNER > Presidio > BERT | GLiNER > Presidio > BERT |
| A-dates | GLiNER > BERT > Presidio | GLiNER > BERT > Presidio | BERT > GLiNER > Presidio |
| B-dates | GLiNER > BERT > Presidio | GLiNER > Presidio > BERT | BERT > GLiNER > Presidio |

Il est identique à celui du F2, sauf aux quatre égalités du F2.

**D'un dataset à l'autre** (aucune moyenne n'est faite, les datasets différant par leurs labels,
leurs tailles et leurs textes) :

- GLiNER contre BERT, F2 : GLiNER est meilleur sur 2 datasets sur 3 dans chaque comparaison ; sur
  le troisième, la différence n'est pas établie (A sur Gretel, B sur ai4privacy) ou BERT est
  meilleur (A-dates et B-dates sur Gretel).
- Presidio est moins bon que les deux modèles sur 3 datasets sur 3 en A et A-dates. En B, il fait
  jeu égal avec BERT sur Nemotron-PII et le dépasse sur Gretel, un effet du label de date
  (section 6.2).

### 6.6 Analyses de sensibilité (exploratoires)

*Les conclusions tiennent-elles quand on change un choix de méthode ?*

Une analyse de sensibilité refait une partie de la comparaison en changeant un seul choix de
méthode, pour voir si les conclusions en dépendent. Si l'ordre des détecteurs reste le même, la
conclusion est robuste ; s'il change, elle dépend de ce choix et doit être présentée avec cette
réserve. Ces analyses sont exploratoires : aucune hypothèse n'avait été écrite pour elles, et
leurs écarts ne sont pas testés. Quatre choix ont été examinés :

| Analyse | Choix de méthode examiné | Question posée |
|---|---|---|
| S1 | l'appariement strict | Le classement change-t-il si l'on accepte des bornes ou des labels approximatifs ? |
| S2 | le seuil de GLiNER (0,35) | Un autre seuil raisonnable changerait-il le classement ? |
| S3 | Gretel comme jeu de test | BERT a-t-il vu nos documents de test pendant son entraînement ? |
| S4 | les fenêtres pour les textes longs | Les détecteurs font-ils moins bien sur les documents longs ? |

**S1. Règles d'appariement plus souples.**

*Pourquoi.* L'appariement strict compte comme une erreur toute détection dont les bornes ou le
label diffèrent du gold, même quand elle couvre le PII ou presque (partie 4). Un détecteur qui
termine souvent ses spans un caractère trop tôt ou trop tard est alors pénalisé autant qu'un
détecteur qui rate le PII. On vérifie que le classement ne vient pas seulement de cette sévérité.

*Méthode.* Le F2 est recalculé avec trois règles plus souples, en plus de la règle stricte :

- mêmes bornes, label ignoré : une détection est correcte si elle a exactement les bornes d'un
  span gold, quel que soit son label ;
- chevauchement, même label : il suffit que la détection partage au moins un caractère avec un
  span gold de même label ;
- chevauchement, label ignoré : un caractère en commun suffit, quel que soit le label.

Le rappel est la part des spans gold touchés par au moins une détection, la précision la part des
détections qui touchent un span gold. Le tableau donne BERT et GLiNER en A, et sur Nemotron-PII
et Gretel en A-dates ; Presidio et les autres cas sont dans les sorties de l'analyse.

*Résultats.* F2 micro selon les quatre règles :

| Comparaison, dataset | Détecteur | Strict | Mêmes bornes, label ignoré | Chevauchement, même label | Chevauchement, label ignoré |
|---|---|---|---|---|---|
| A, ai4privacy | BERT / GLiNER | 0,796 / 0,867 | 0,804 / 0,870 | 0,891 / 0,926 | 0,905 / 0,930 |
| A, Nemotron-PII | BERT / GLiNER | 0,762 / 0,823 | 0,762 / 0,823 | 0,804 / 0,851 | 0,804 / 0,853 |
| A, Gretel | BERT / GLiNER | 0,718 / 0,713 | 0,718 / 0,715 | 0,780 / 0,740 | 0,786 / 0,754 |
| A-dates, Nemotron-PII | BERT / GLiNER | 0,778 / 0,801 | 0,778 / 0,801 | 0,852 / 0,833 | 0,853 / 0,835 |
| A-dates, Gretel | BERT / GLiNER | 0,787 / 0,595 | 0,787 / 0,597 | 0,908 / 0,652 | 0,913 / 0,661 |

Ignorer le label (mêmes bornes, label ignoré) change le F2 de 0,011 au plus : les confusions entre
nos labels sont rares. Accepter les chevauchements fait plus monter BERT que GLiNER, car beaucoup
d'erreurs de BERT sont des erreurs de bornes. L'ordre change à deux endroits : sur Gretel en A
(égalité en strict, BERT devant avec chevauchement) et sur Nemotron-PII en A-dates (GLiNER devant
en strict, BERT devant avec chevauchement). Presidio reste loin derrière avec toutes les règles.

**S2. Le seuil de GLiNER.**

*Pourquoi.* GLiNER donne à chaque span candidat un score de confiance et ne garde que ceux
au-dessus d'un seuil. Un seuil bas fait trouver plus de PII (rappel plus haut) mais ajoute des
fausses alertes (précision plus basse) ; un seuil haut fait l'inverse. Le seuil 0,35 a été choisi
sur des données d'entraînement (partie 3) ; BERT n'a pas de seuil. Si le classement ne tenait
qu'à cette valeur précise, il serait fragile.

*Méthode.* GLiNER a été lancé une fois au seuil 0,05 ; garder les spans dont le score dépasse un
seuil donné donne ses prédictions à ce seuil. Le F2 est ainsi calculé pour chaque seuil de 0,05 à
0,95, par pas de 0,05, et comparé à celui de BERT. Le plan demandait de vérifier que le
classement tient entre 0,25 et 0,60, la plage où le F2 bougeait à peine sur les données
d'entraînement.

*Résultats.* F2 micro sur les jeux de test propres (six seuils montrés ; la courbe complète est
dans les sorties de l'analyse) :

| Comparaison, dataset | BERT | GLiNER 0,15 | 0,25 | 0,35 | 0,45 | 0,60 | 0,75 | Meilleur F2 de GLiNER (seuil) |
|---|---|---|---|---|---|---|---|---|
| A, ai4privacy | 0,796 | 0,862 | 0,866 | 0,867 | 0,867 | 0,861 | 0,849 | 0,867 (0,30 à 0,45) |
| A, Nemotron-PII | 0,762 | 0,804 | 0,817 | 0,823 | 0,824 | 0,826 | 0,817 | 0,826 (0,60) |
| A, Gretel | 0,718 | 0,714 | 0,719 | 0,713 | 0,702 | 0,675 | 0,638 | 0,719 (0,20 et 0,25) |
| B, ai4privacy | 0,891 | 0,896 | 0,897 | 0,898 | 0,894 | 0,886 | 0,876 | 0,899 (0,20) |
| B, Nemotron-PII | 0,757 | 0,952 | 0,957 | 0,960 | 0,962 | 0,962 | 0,960 | 0,962 (0,45 à 0,70) |
| B, Gretel | 0,459 | 0,733 | 0,739 | 0,736 | 0,724 | 0,698 | 0,665 | 0,739 (0,25) |

À 0,35, ces prédictions filtrées redonnent les scores du benchmark à 0,001 près : 13 documents
sur 14 962 diffèrent, parce que les scores enregistrés sont arrondis à trois décimales et que,
là où deux fenêtres se chevauchent, le span le plus long est gardé même si son score est plus bas.

- Là où l'écart entre GLiNER et BERT était établi (A sur ai4privacy et Nemotron-PII, B sur
  Nemotron-PII et Gretel), GLiNER reste devant pour tout seuil de 0,10 à 0,90.
- Là où il ne l'était pas (A sur Gretel, B sur ai4privacy), les deux restent à moins de 0,01
  l'un de l'autre entre 0,25 et 0,35. Sur ai4privacy (B), c'est encore vrai jusqu'à 0,60. Sur
  Gretel (A), le F2 de GLiNER baisse quand le seuil monte, et BERT passe devant de plus de 0,01
  à partir de 0,45 (0,702 contre 0,718). Les écarts aux seuils autres que 0,35 ne sont pas
  testés.
- Même le meilleur seuil de chaque jeu de test, ce qui reviendrait à régler sur le test, ne
  ferait gagner à GLiNER que 0,006 de F2 au plus (A, Gretel). BERT n'a pas de seuil et Presidio garde ses valeurs par
  défaut : seul GLiNER a bénéficié d'un réglage, qui pèse donc peu.

Réponse à la question du plan : entre 0,25 et 0,60, aucun classement établi ne change ; seule
l'égalité sur Gretel (A) tourne en faveur de BERT aux seuils les plus hauts.

**S3. BERT a-t-il vu nos documents Gretel ?**

*Pourquoi.* Le corpus d'entraînement du BERT de Gravitee contient 25 948 lignes de Gretel PII
finance, et sa fiche ne dit pas de quel split elles viennent. Si certains de nos documents de test
en font partie, BERT a vu les réponses, et son score sur Gretel est gonflé.

*Méthode.* Chacun de nos 2 962 documents de test Gretel est comparé aux lignes Gretel du corpus
d'entraînement. On cherche d'abord les textes identiques, puis on mesure la part des 13-grammes de
mots du document (suites de 13 mots consécutifs, en minuscules) qu'on retrouve dans le corpus :
une telle suite se répète rarement par hasard, si bien qu'une part élevée signale un texte
recopié. Un document est dit « vu » s'il est identique à une ligne d'entraînement ou s'il partage
au moins la moitié de ses 13-grammes avec elles. Le F2 est ensuite recalculé sans les documents
vus.

*Résultats.* 8 documents sont identiques à une ligne d'entraînement, 91 partagent au moins la
moitié de leurs 13-grammes avec elles, 5 les partagent tous ; la part médiane est de 0,05. Sans
ces 95 documents, le F2 de BERT sur Gretel reste 0,718 (GLiNER : 0,717). BERT n'a pas appris nos
documents de test, mais il a été entraîné sur des documents du même générateur, avec les mêmes
conventions d'annotation ; cela l'aide probablement sur Gretel.

**S4. Longueur des documents.**

*Pourquoi.* Les modèles ont une longueur maximale : GLiNER lit 384 mots à la fois, BERT 512
tokens. Les textes plus longs sont lus par fenêtres qui se chevauchent (partie 3). Les fenêtres
évitent la troncature, mais un modèle peut quand même faire moins bien sur un long document, par
exemple parce qu'une fenêtre coupe le contexte.

*Méthode.* Les documents d'au plus 384 mots sont séparés des plus longs, et le F2 micro
(comparaison A) est calculé sur chaque groupe. ai4privacy n'a que des textes courts.

*Résultats.* F2 micro sur les documents courts / longs :

| Dataset | Documents courts / longs | Presidio | BERT | GLiNER |
|---|---|---|---|---|
| ai4privacy | 10 000 / 0 | 0,157 / n/a | 0,796 / n/a | 0,867 / n/a |
| Nemotron-PII | 1 769 / 231 | 0,417 / 0,530 | 0,784 / 0,622 | 0,828 / 0,785 |
| Gretel | 2 199 / 763 | 0,181 / 0,141 | 0,734 / 0,672 | 0,744 / 0,633 |

Les deux modèles font moins bien sur les documents longs, qui sont aussi plus structurés
(tableaux, formulaires, code) ; les fenêtres empêchent la troncature mais n'effacent pas cette
baisse.

## 7. Conclusions pour la question 1

**Mesuré et établi** (intervalles à 95 %, tests appariés, correction de Holm) :

1. Sur les labels du proposal (A), GLiNER trouve le mieux les PII sur ai4privacy et Nemotron-PII,
   et fuit le moins sur ai4privacy ; sur Gretel, BERT et GLiNER ont le même F2 (GLiNER est
   devant en F1) et BERT fuit moins.
   Les ordres établis tiennent pour tout seuil de GLiNER entre 0,25 et 0,60 (S2).
2. Les règles seules (Presidio) laissent fuir 57 % à 87 % des PII, faute de couvrir noms et
   adresses. Sur les e-mails, elles ont le meilleur F2 des trois (0,96 à 0,99).
3. Avec toutes les dates (A-dates), GLiNER reste devant sur ai4privacy et Nemotron-PII, et BERT est
   nettement devant sur Gretel (F2 0,787 contre 0,595).

**Classement d'ensemble** (décompte des écarts établis en 6.5, sur les 12 cas : 4 comparaisons
× 3 datasets ; pour le taux de fuite, gagner veut dire laisser fuir moins) :

| Duel (cas gagnés – cas perdus) | F2 | F1 | Taux de fuite |
|---|---|---|---|
| GLiNER contre BERT | 8 – 2 (2 égalités) | 10 – 2 | 3 – 5 (4 égalités) |
| BERT contre Presidio | 9 – 1 (2 égalités) | 9 – 3 | 12 – 0 |
| GLiNER contre Presidio | 12 – 0 | 12 – 0 | 11 – 0 (1 égalité) |

Pour la qualité de détection, le classement est GLiNER, puis BERT, puis Presidio. BERT ne devance
GLiNER que sur Gretel quand toutes les dates comptent, et Presidio ne devance BERT que sur les
labels à format fixe. Pour la fuite, Presidio est dernier partout, mais BERT et GLiNER ne se
départagent pas. Ces 12 cas ne sont pas indépendants, puisque les mêmes datasets reviennent dans
les quatre comparaisons : ce décompte résume les tests de la section 6.5, il n'en est pas un
nouveau.

**Expliqué** (par la comparaison C, section 6.2) :

4. L'avantage apparent de Presidio sur BERT dans la comparaison B vient de la correspondance d'une
   classe de date générique avec « date de naissance ».

**Supposé** (non testé) :

5. L'avantage de BERT sur Gretel vient probablement de ses données d'entraînement, qui contiennent
   des documents du même générateur avec les mêmes conventions (il n'a pas vu nos documents de
   test, S3).

## 8. Limites

- Les trois datasets sont synthétiques ; de vrais prompts peuvent différer par le style et les
  formats de PII.
- Anglais seulement.
- Les correspondances de labels de Presidio et BERT sont approchées (dates, lieux) ; la
  comparaison C ne traite que les dates.
- GLiNER est le seul détecteur avec un réglage ajusté ; même réglé sur les jeux de test, il ne
  gagnerait que 0,006 de F2 au plus (S2).
- L'appariement strict pénalise fortement les conventions de bornes ; S1 montre de combien.

## 9. Reproduire les résultats

```bash
uv sync                                               # environnement
uv run python experiments/download_data.py            # datasets, révisions figées
uv run python experiments/prepare_data.py             # jeux de test, graine 0 (versions clean et dates)
uv run python experiments/tune_gliner.py              # seuil de GLiNER sur données d'entraînement (0,35)
uv run python experiments/run_benchmark.py presidio   # aussi bert, gliner ; ajouter --dates pour la comparaison C
uv run python experiments/score.py                    # results/scores.md
uv run python experiments/significance.py             # intervalles et tests (6.4, 6.5)
uv run python experiments/sensitivity.py              # S1, S2, S4
uv run python experiments/contamination.py            # S3
```

Fichiers de résultats : `results/scores.md`, `results/without_name_address/scores.md`, et dans
`results/analysis/` : `significance.md` (avec tous les tests et les intervalles par label), `sensitivity.md`, `contamination.md`.

## Annexe A. F2 par label [intervalle à 95 %]

**Comparaison A**

| Dataset | Détecteur | NAME | EMAIL | PHONE | ADDRESS | DATE_OF_BIRTH | CARD_NUMBER |
|---|---|---|---|---|---|---|---|
| ai4privacy | Presidio | 0,000 | 0,990 [0,984 ; 0,995] | 0,371 [0,347 ; 0,395] | 0,000 | - | 0,073 [0,035 ; 0,116] |
| ai4privacy | BERT | 0,832 [0,824 ; 0,840] | 0,986 [0,978 ; 0,992] | 0,840 [0,824 ; 0,855] | 0,473 [0,451 ; 0,495] | - | 0,762 [0,708 ; 0,810] |
| ai4privacy | GLiNER | 0,923 [0,918 ; 0,928] | 0,970 [0,963 ; 0,977] | 0,892 [0,879 ; 0,904] | 0,475 [0,448 ; 0,501] | - | 0,432 [0,363 ; 0,502] |
| Nemotron-PII | Presidio | 0,000 | 0,988 [0,980 ; 0,994] | 0,727 [0,697 ; 0,756] | 0,000 | 0,590 [0,555 ; 0,624] | 0,141 [0,093 ; 0,192] |
| Nemotron-PII | BERT | 0,839 [0,819 ; 0,857] | 0,954 [0,939 ; 0,968] | 0,934 [0,919 ; 0,948] | 0,447 [0,403 ; 0,491] | 0,383 [0,351 ; 0,415] | 0,949 [0,932 ; 0,963] |
| Nemotron-PII | GLiNER | 0,712 [0,690 ; 0,734] | 0,970 [0,960 ; 0,980] | 0,931 [0,917 ; 0,944] | 0,664 [0,608 ; 0,717] | 0,978 [0,965 ; 0,988] | 0,947 [0,920 ; 0,972] |
| Gretel | Presidio | 0,000 | 0,960 [0,945 ; 0,973] | 0,542 [0,501 ; 0,582] | 0,000 | 0,158 [0,133 ; 0,186] | 0,409 [0,270 ; 0,543] |
| Gretel | BERT | 0,850 [0,833 ; 0,867] | 0,845 [0,820 ; 0,870] | 0,910 [0,888 ; 0,930] | 0,801 [0,782 ; 0,819] | 0,085 [0,072 ; 0,098] | 0,714 [0,613 ; 0,807] |
| Gretel | GLiNER | 0,744 [0,728 ; 0,760] | 0,617 [0,585 ; 0,649] | 0,897 [0,875 ; 0,917] | 0,621 [0,597 ; 0,645] | 0,876 [0,835 ; 0,911] | 0,579 [0,477 ; 0,668] |

**DATE dans la comparaison A-dates** (les autres labels restent à moins de 0,023 de la comparaison A)

| Dataset | Presidio | BERT | GLiNER |
|---|---|---|---|
| ai4privacy | 0,374 [0,349 ; 0,400] | 0,737 [0,722 ; 0,751] | 0,933 [0,923 ; 0,943] |
| Nemotron-PII | 0,771 [0,744 ; 0,797] | 0,655 [0,627 ; 0,682] | 0,790 [0,759 ; 0,818] |
| Gretel | 0,302 [0,277 ; 0,325] | 0,715 [0,699 ; 0,732] | 0,397 [0,377 ; 0,417] |

## Méthodes utilisées

L'appariement strict des spans suit l'évaluation usuelle de la reconnaissance d'entités nommées
(MUC ; CoNLL-2003,
Tjong Kim Sang et De Meulder, 2003). Les intervalles de confiance utilisent le bootstrap par percentiles (Efron et
Tibshirani, 1993), tiré par groupes d'exemples liés (Field et Welsh, 2007) ; les différences sont
testées par randomisation approchée appariée (Noreen, 1989 ; Yeh, 2000), corrigée par la méthode de
Holm (Holm, 1979) ;
le recouvrement par 13-grammes suit le contrôle de décontamination de Brown et al. (2020).
