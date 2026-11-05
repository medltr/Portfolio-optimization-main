# Rapport d'exploration (phase A)

*Pour un lecteur débutant. Les termes techniques sont expliqués dans le glossaire de `RAPPORT_FINAL.md`.*

## L'idée : se protéger contre la chance

Quand on teste beaucoup d'idées sur les mêmes données, l'une d'elles finit toujours par sembler bonne, ne serait-ce que par hasard. Pour s'en protéger, on a coupé les 19 mois de test en deux :

- **Sélection : les signaux de 2022** (12 mois, avec une forte baisse des marchés). C'est là qu'on **choisit**.
- **Confirmation : les signaux de 2023** (7 mois, une reprise). C'est là qu'on **vérifie**, sans rien changer.

Les règles de choix ont été écrites dans `JOURNAL.md` **avant** les calculs. Toutes les variantes testées sont rapportées ci-dessous, pas seulement les meilleures.

## A1 : d'autres façons de classer les actions

Chaque signal donne une note aux actions chaque mois. L'**IC** mesure si cette note classe correctement les actions selon leur rendement du mois suivant (+ : bon classement, 0 : aucun lien, − : classement inversé). La **statistique t** indique si l'effet est assez net pour ne pas être dû au hasard : il faudrait environ 2.

| Signal | L'idée | IC 2022 (t) | IC 2023 (t) | Tient en 2023 ? |
|---|---|---|---|---|
| Momentum 12-1 | ce qui a monté sur un an continue | **+0,12 (1,28)** | −0,02 (−0,19) | **Non** |
| Momentum + faible volatilité | les deux idées combinées | +0,13 (1,15) | −0,01 (−0,08) | Non |
| Faible volatilité | les actions calmes font mieux qu'attendu | +0,09 (0,74) | +0,01 (0,14) | De justesse, mais trop faible pour compter |
| Retournement 1 mois | ce qui a chuté le mois dernier rebondit | +0,02 (0,33) | 0,00 (0,00) | Non |
| Qualité (rentabilité) | les entreprises rentables font mieux | −0,04 (−0,62) | +0,02 (0,19) | Non |
| Faible bêta | les actions peu sensibles au marché font mieux | −0,00 (−0,05) | −0,09 (−1,00) | Non |
| LSTM relatif | réseau de neurones | −0,12 (−1,44) | −0,09 (−0,79) | Non (négatif deux fois) |

**Résultat.** Le meilleur signal en 2022, le **momentum**, disparaît en 2023. Aucun signal n'a d'effet net (toutes les statistiques t restent sous 1,3). Le LSTM est négatif dans les deux périodes.

## A2 : comment répartir l'argent entre Gagnants et Stables

Le projet impose 50 % sur les Gagnants dynamiques et 50 % sur les Stables de qualité. Or il y a en général **beaucoup moins de Gagnants** (environ 32 % des actions retenues). Le 50/50 **surpondère donc les Gagnants**. On a testé quatre règles, chacune avec l'équipondéré et avec une inclinaison vers le momentum (le meilleur signal sur la sélection) :

| Règle | Équipondéré : 2022 → 2023 (Sharpe) | Inclinaison momentum : 2022 → 2023 |
|---|---|---|
| 50/50 (actuelle) | −0,40 → 0,08 | **−0,24** → 0,09 |
| Aucune contrainte (1/N) | −0,64 → 0,36 | −0,52 → 0,37 |
| Proportionnelle au nombre d'actions | −0,64 → 0,36 | −0,58 → 0,33 |
| Chaque groupe entre 30 % et 70 % | −0,60 → 0,31 | −0,50 → 0,28 |
| *Repère : minimum de variance* | *0,18 → −0,52* | |
| *Repère : SPY* | *−0,34 → 1,34* | |

*(Règle « aucune contrainte » et règle proportionnelle donnent le même équipondéré, à savoir 1/N : c'est normal.)*

**Résultat.**
- **Le classement s'inverse d'une année à l'autre.** Le 50/50, qui surpondère les Gagnants, est le meilleur en 2022 et le pire en 2023. Les règles qui donnent moins de poids aux Gagnants font l'inverse. Choisir une règle revient donc à **parier sur le type de marché**, pas à améliorer durablement la méthode.
- **L'inclinaison vers le momentum** aide les 4 règles en 2022, puis n'apporte plus rien en 2023 (entre −0,03 et +0,01).
- **Même les repères s'inversent :** le minimum de variance est excellent en 2022 et mauvais en 2023 ; SPY fait l'inverse.

## Ce qui tient, ce qui ne tient pas

**Rien ne tient de façon convaincante sur la confirmation.**

- La combinaison choisie sur 2022 (inclinaison momentum + 50/50) fait à peine mieux que l'équipondéré 50/50 en 2023 (0,088 contre 0,080), et son signal n'est plus positif. Selon la règle fixée à l'avance, **elle n'est pas confirmée**.
- La faible volatilité est le seul signal positif deux années de suite, mais trop faiblement pour compter. La choisir maintenant serait justement choisir en regardant 2023.

**Décision pour la phase B :** le projet sera configuré par défaut avec la solution la plus simple et la plus robuste, l'**équipondéré 50/50**, qui ne dépend d'aucune prévision. Les autres signaux et constructions restent disponibles en option, et l'ancienne méthode (LSTM + max Sharpe) aussi.

## Limites propres à cette phase

- **7 mois de confirmation, c'est très peu.** Un « non confirmé » signifie « pas de preuve », pas « preuve que ça ne marche pas ».
- **Biais de survie, absence de coûts de transaction, taux sans risque ignoré :** les mêmes limites que dans `RAPPORT_FINAL.md`.

## Fichiers

- Code : `experiments/exploration.py`
- Résultats : `experiments/output/exploration_20261105_161112/` (`ic_summary.csv`, `a2_summary.csv`, `decision.json`, rendements journaliers, journal des poids)
