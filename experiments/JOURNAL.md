# Journal de l'étude

## Contexte (travaux précédents)

- **seed_study.py** (test de 6 mois, févr.–juil. 2023) : selon la graine, le Sharpe de la stratégie LSTM + max Sharpe va de 0,02 à 1,52 (moyenne 0,54, écart-type 0,66). L'IC moyen est d'environ −0,02. L'équipondéré 50/50 fait mieux (Sharpe 0,86).
- **walk_forward.py** (test de 19 mois, janv. 2022 → juil. 2023, réentraînement tous les 6 mois, ensemble de 5 graines) : le LSTM ensemble + max Sharpe a un Sharpe de −0,77 et un IC de −0,096 (t = −1,27). C'est le pire des six portefeuilles. Le minimum de variance a le meilleur Sharpe (0,00), SPY a 0,06.

## Règles fixées AVANT les nouvelles expériences (2026-11-05)

Elles sont écrites avant tout calcul et ne seront pas modifiées au vu des résultats du test.

- Uniquement le cache, le Python de `.venv`, aucun fichier existant modifié. Toutes les sorties vont dans `experiments/output/`.
- Calendrier : celui de `walk_forward.py` (réentraînements au 31/01/2022, 31/07/2022, 31/01/2023 et 31/07/2023 ; 19 mois de signaux de test). Même découpage entraînement / validation (`split_at`) et mêmes contrôles anti-fuite.
- LSTM : hyperparamètres de `config.py`, déterminisme TensorFlow, ensemble = moyenne des graines 0 à 4.
- Signaux de l'étape 1 :
  - (a) LSTM, cible = rendement brut du mois suivant (modèle actuel) ;
  - (b) LSTM, cible = rendement moins la moyenne des rendements des actions du même mois (cible relative) ;
  - (c) momentum 12-1 = (1 + return_12m)^12 / (1 + return_1m) − 1, soit la performance du mois t−12 au mois t−1 ;
  - (d) Ridge, cible relative, entrées = la séquence 12 × 18 mise à plat et normalisée sur l'entraînement. alpha est choisi dans {0,01 ; 0,1 ; 1 ; 10 ; 100 ; 1 000 ; 10 000} selon la MSE de validation, et le modèle est ajusté sur l'entraînement seul.
- Évaluation : IC de Spearman mensuel sur les mêmes actions pour tous les signaux. On rapporte l'IC moyen, le nombre de mois avec IC > 0, la statistique t (moyenne / écart-type × √n) et l'IC par bloc.
- Règle de choix pour l'étape 2 : le « meilleur signal » est celui qui a la statistique t d'IC la plus élevée sur les 19 mois, parmi (a), (b), (c) et (d). On teste toujours (b), et en plus le meilleur signal s'il est différent de (b).
- Portefeuilles de l'étape 2 (contraintes : 10 % maximum par action, 50 % Gagnants / 50 % Stables) :
  - (i) max Sharpe, comme dans le projet. Rendement attendu = la prévision ; pour le momentum, le momentum moins sa moyenne du mois ;
  - (ii) inclinaison : poids = (0,5 / n_groupe) × (1 + 0,5 × z), où z est le rang dans le groupe ramené à [−1, +1]. Plafond de 10 % avec redistribution dans le groupe, et liquidités si les contraintes sont irréalisables.
- Références : minimum de variance, équipondéré 50/50, 1/N, SPY, et le LSTM actuel + max Sharpe pour rappel.
- Aucun signe de signal n'est inversé. Aucun paramètre n'est ajusté sur le test.

## Étape 1 : qualité du signal (`signal_study.py`)

**Sortie :** `output/signal_study_20261105_120327/`

**Ce qui a été fait.** Les quatre signaux ont été recalculés à chaque réentraînement du walk-forward (4 blocs, 19 mois, 27 à 50 actions par mois, 40 en moyenne). Pour chacun, on a mesuré l'IC de Spearman mensuel entre le signal et le rendement réel du mois suivant, sur exactement les mêmes actions.

**Contrôles.**
- Le LSTM (a) est reproductible : les prévisions des 5 graines sont identiques au bit près à celles de `walk_forward_20261104_230330`. Le petit écart sur la moyenne (4,8e-9) vient seulement de l'arrondi lors de l'écriture du CSV.
- Aucun contrôle anti-fuite ne s'est déclenché.
- Limite de journalisation : `run_info.json` ne garde le nombre d'époques que du dernier bloc (les clés sont écrasées d'un bloc à l'autre). Cela n'a aucun effet sur les résultats.

**Résultats (19 mois).**

| Signal | IC moyen | Mois IC > 0 | t de l'IC | IC bloc 1 (S1 2022) | Bloc 2 (S2 2022) | Bloc 3 (S1 2023) | Bloc 4 (juil. 2023, 1 mois) |
|---|---|---|---|---|---|---|---|
| (a) LSTM, cible brute | −0,096 | 7 / 19 | −1,27 | −0,106 | −0,133 | −0,052 | −0,066 |
| (b) LSTM, cible relative | −0,109 | 6 / 19 | −1,66 | −0,112 | −0,127 | −0,156 | +0,291 |
| (c) Momentum 12-1 | **+0,073** | **11 / 19** | **+1,07** | +0,088 | +0,161 | −0,011 | −0,041 |
| (d) Ridge, cible relative | −0,022 | 9 / 19 | −0,28 | −0,109 | −0,033 | +0,118 | −0,270 |

- Ridge : alpha = 10 000 retenu aux 4 dates. C'est la borne haute de la grille, donc la régularisation la plus forte : selon la validation, le modèle a intérêt à prédire presque la même valeur pour toutes les actions. Autrement dit, il ne trouve pas de relation exploitable. La grille n'a pas été élargie, conformément aux règles.
- Corrélations de rang entre signaux : (a)–(b) +0,66 ; LSTM–momentum entre −0,33 et −0,41. **Les LSTM classent les actions plutôt à l'inverse du momentum.**

**Ce que ça signifie.**
- Changer la cible du LSTM (b) n'améliore rien. L'IC reste négatif, à peu près comme (a).
- Le Ridge, plus simple, n'apprend presque rien (IC proche de 0).
- Le seul signal à IC positif est le momentum, le plus simple, qui ne demande aucun apprentissage : positif 11 mois sur 19. Mais t = 1,07, ce qui est loin du seuil habituel d'environ 2. Avec 19 mois, on ne peut pas exclure le hasard.
- Selon la règle fixée à l'avance, le meilleur signal est **(c) momentum**. L'étape 2 teste donc (b) et (c).

## Étape 2 : utilisation dans un portefeuille (`portfolio_study.py`)

**Sortie :** `output/portfolio_study_20261105_132550/`

**Ce qui a été fait.** Le script a appliqué automatiquement la règle de choix : t de l'IC (a) −1,27, (b) −1,66, (c) +1,07, (d) −0,28, donc meilleur signal = (c) momentum. Il a testé (b) et (c), chacun de deux façons, (i) max Sharpe et (ii) inclinaison prudente, avec les mêmes actions chaque mois (40 en moyenne), 10 % maximum par action et 50/50. Période : 19 mois, 398 jours de bourse (2 février 2022 → 1er septembre 2023).

**Contrôles.**
- Max Sharpe : 19 mois optimisés sur 19 pour (a), (b) et (c). Aucun repli, aucun mois en liquidités.
- Inclinaison : plafond de 10 % atteint 3 mois (mai, juin et juillet 2022, peu de Gagnants), avec redistribution. Aucun mois en liquidités.
- Concentration du max Sharpe : en moyenne 11 à 12 actions ont un poids supérieur à 0,5 %, dont 8 à 9 au plafond de 10 %. Environ 85 à 90 % du portefeuille repose donc sur 9 actions sur 40.

**Résultats sur 19 mois.**

| Portefeuille | Rendement annuel | Volatilité | Sharpe | Max drawdown |
|---|---|---|---|---|
| (c) Momentum + max Sharpe | −1,9 % | 19,8 % | −0,10 | **−16,8 %** |
| (c) Momentum + inclinaison | −3,2 % | 21,2 % | −0,15 | −21,1 % |
| (b) LSTM relatif + max Sharpe | −16,1 % | 22,2 % | −0,79 | −31,8 % |
| (b) LSTM relatif + inclinaison | −10,0 % | 22,2 % | −0,48 | −28,9 % |
| (a) LSTM actuel + max Sharpe (rappel) | −16,7 % | 23,8 % | −0,77 | −34,8 % |
| Minimum de variance | 0,0 % | **18,0 %** | 0,00 | −18,6 % |
| Équipondéré 50/50 | −5,7 % | 21,7 % | −0,27 | −25,2 % |
| Équipondéré 1/N | −7,5 % | 20,6 % | −0,38 | −27,6 % |
| SPY | **+1,3 %** | 20,8 % | **0,06** | −22,1 % |

**Sharpe par bloc.**

| Portefeuille | Bloc 1 (S1 2022) | Bloc 2 (S2 2022) | Bloc 3 (S1 2023) | Bloc 4 (juil. 2023) |
|---|---|---|---|---|
| (c) Momentum + max Sharpe | −0,58 | 0,74 | 0,01 | −1,58 |
| (c) Momentum + inclinaison | −0,94 | 0,67 | 0,50 | −1,59 |
| (b) LSTM relatif + max Sharpe | −1,74 | 0,13 | −0,41 | −0,83 |
| (b) LSTM relatif + inclinaison | −1,39 | 0,37 | 0,20 | −1,31 |
| (a) LSTM actuel + max Sharpe | −1,60 | 0,26 | −0,42 | −2,33 |
| Minimum de variance | −0,48 | 0,98 | −0,48 | −0,70 |
| Équipondéré 50/50 | −1,14 | 0,55 | 0,44 | −1,47 |
| Équipondéré 1/N | −1,38 | 0,20 | 0,71 | −1,54 |
| SPY | −0,72 | 0,08 | 1,75 | −1,06 |

**Ce que ça signifie.**
- **Les résultats suivent la qualité du signal.** Partir de l'équipondéré 50/50 et le pencher vers le momentum (IC positif) améliore le Sharpe de −0,27 à −0,15, et ce dans les blocs 1, 2 et 3. Le pencher vers le LSTM (IC négatif) le dégrade de −0,27 à −0,48, là aussi dans les blocs 1, 2 et 3. L'inclinaison fait donc ce qu'on attend : elle traduit le signal sans prendre de gros risques.
- **Le max Sharpe amplifie le signal, qu'il soit bon ou mauvais.** Avec le LSTM, il donne les pires portefeuilles (Sharpe environ −0,8, drawdown d'environ −32 à −35 %). Avec le momentum, il donne le plus faible drawdown de tous (−16,8 %), mais un Sharpe (−0,10) encore inférieur au minimum de variance et à SPY.
- **Rien ne bat clairement SPY ni le minimum de variance sur cette période.** Les écarts entre les meilleurs portefeuilles (Sharpe de −0,15 à +0,06) sont petits face à l'incertitude de 19 mois.
- Le bloc 4 ne contient qu'un mois : ses chiffres ne permettent aucune conclusion.

## Étape 3 : rapport final

`RAPPORT_FINAL.md` est écrit pour un lecteur débutant : point de départ, les 5 expériences et leur raison d'être, tableaux de résultats, ce qui marche et ce qui ne marche pas, limites (biais de survie, période courte, pas de coûts de transaction, taux sans risque ignoré), pistes et glossaire.

**Note sur les durées.** Le calcul de l'étape 1 a pris environ 10 minutes, mais le lancement a duré 1 h 21 à cause d'une pause d'environ 71 minutes visible dans le journal d'exécution (probablement une mise en veille du PC). L'étape 2 a pris moins d'une minute. Comme tout est déterministe, ces pauses n'ont aucun effet sur les résultats.

---

# Phase A : exploration avec sélection et confirmation

## Règles fixées AVANT les calculs de la phase A (2026-11-05)

- **Périodes :** sélection = signaux de janvier à décembre 2022 (12 mois) ; confirmation = signaux de janvier à juillet 2023 (7 mois). Les choix se font uniquement sur la sélection, et rien n'est réajusté ensuite.
- **Signaux A1.** Le sens de chaque signal est fixé à l'avance d'après la littérature financière ; aucun ne sera inversé après coup.
  - `reversal_1m` = −return_1m (retournement court terme : on préfère les actions qui ont baissé le mois dernier) ;
  - `low_vol` = −(écart-type des log-rendements journaliers sur les 252 dernières séances jusqu'à la date du signal, au moins 200 observations) ;
  - `low_beta` = −(bêta Mkt-RF glissant du projet, décalé de 2 mois) ;
  - `quality` = +(bêta RMW glissant du projet) ;
  - `mom_lowvol` = moyenne des rangs centiles du mois de `momentum` et de `low_vol` ;
  - comparaisons : `momentum` 12-1 et `lstm_relative` (ensemble de 5 graines, prévisions de `signal_study_20261105_120327`, reproductibles).
- **Meilleur signal S\*** = la statistique t d'IC la plus élevée sur la sélection. **Confirmé** si son IC moyen sur la confirmation est > 0.
- **Règles de répartition A2** (plafond de 10 % partout ; si irréalisable, liquidités) :
  - (i) `50_50` ;
  - (ii) `none` (équipondéré = 1/N ; inclinaison classée sur tout l'univers) ;
  - (iii) `proportional` (part de chaque groupe = n_groupe / N) ;
  - (iv) `30_70` (part du groupe = n_groupe / N, bornée entre 30 % et 70 %).
- **Constructions A2 :** équipondéré par groupe, et inclinaison vers S\* (k = 0,5, rang dans le groupe, comme à l'étape 2). Repères : SPY, minimum de variance 50/50.
- **Choix pour la phase B :**
  - C\* = la combinaison A2 qui a le meilleur Sharpe sur la sélection.
  - Elle devient la configuration par défaut si son Sharpe sur la confirmation est supérieur à celui de l'équipondéré 50/50 et, si elle utilise un signal, si S\* est confirmé.
  - Sinon, la configuration par défaut est l'équipondéré 50/50.

## A1 : autres signaux (`exploration.py`, sortie `output/exploration_20261105_161112/`)

**Ce qui a été fait.** Calcul de 5 nouveaux signaux, sans entraînement, sur les mêmes actions chaque mois (761 lignes, 19 mois). On les compare au momentum et au LSTM relatif sur la sélection (2022, 12 mois) et sur la confirmation (2023, 7 mois). Aucun signal manquant, aucune fuite (tous les signaux n'utilisent que des données connues à la date du signal).

| Signal | IC sélection | t sélection | IC confirmation | t confirmation | IC total | Mois > 0 (sél. / conf. / total) |
|---|---|---|---|---|---|---|
| momentum 12-1 | +0,124 | **1,28** | −0,015 | −0,19 | +0,073 | 8 / 3 / 11 |
| LSTM relatif | −0,119 | −1,44 | −0,092 | −0,79 | −0,109 | 3 / 3 / 6 |
| retournement 1 mois | +0,023 | 0,33 | 0,000 | 0,00 | +0,014 | 6 / 4 / 10 |
| faible volatilité | +0,085 | 0,74 | +0,014 | 0,14 | +0,059 | 9 / 3 / 12 |
| faible bêta | −0,004 | −0,05 | −0,093 | −1,00 | −0,037 | 6 / 2 / 8 |
| qualité (RMW) | −0,041 | −0,62 | +0,016 | 0,19 | −0,020 | 6 / 5 / 11 |
| momentum + faible volatilité | +0,127 | 1,15 | −0,009 | −0,08 | +0,077 | 9 / 3 / 12 |

**Décision (règle fixée).** Meilleur signal sur la sélection : **momentum** (t = 1,28). Sur la confirmation, son IC vaut −0,015 : **il n'est pas confirmé**.

**Ce que ça signifie.**
- Aucun signal n'a une statistique t supérieure à 1,3, que ce soit sur la sélection ou sur la confirmation.
- Les signaux qui marchaient en 2022 (momentum et sa combinaison) tombent à zéro en 2023.
- La faible volatilité est la seule à rester légèrement positive dans les deux périodes, mais de façon minime (t de 0,74 puis 0,14). Elle n'a pas été choisie, car la règle se fonde sur la sélection seule.
- Le LSTM relatif reste négatif dans les deux périodes.

## A2 : règles de répartition entre rôles

**Ce qui a été fait.** 4 règles × (équipondéré, inclinaison vers le momentum), sur les mêmes actions. Contrôle : l'équipondéré 50/50 recalculé est identique à la référence des études précédentes. Aucun mois en liquidités. L'inclinaison 50/50 a été plafonnée 3 mois. `EW_none` et `EW_proportional` sont identiques par construction (tous deux = 1/N). Avec la règle `30_70`, les Gagnants reçoivent 30 % la plupart des mois (médiane 30 %, moyenne 36 %) ; leur part naturelle est d'environ 32 % en moyenne.

| Portefeuille | Sharpe sélection (2022) | Sharpe confirmation (2023) | Sharpe total | Pire baisse totale |
|---|---|---|---|---|
| Équipondéré 50/50 | −0,40 | 0,08 | −0,27 | −25,2 % |
| Équipondéré aucune contrainte (1/N) | −0,64 | 0,36 | −0,38 | −27,6 % |
| Équipondéré proportionnel | −0,64 | 0,36 | −0,38 | −27,6 % |
| Équipondéré 30–70 % | −0,60 | 0,31 | −0,36 | −27,0 % |
| **Inclinaison momentum 50/50 (C\*)** | **−0,24** | 0,09 | −0,15 | −21,1 % |
| Inclinaison aucune contrainte | −0,52 | 0,37 | −0,29 | −25,2 % |
| Inclinaison proportionnelle | −0,58 | 0,33 | −0,34 | −25,6 % |
| Inclinaison 30–70 % | −0,50 | 0,28 | −0,30 | −24,5 % |
| Minimum de variance 50/50 (repère) | 0,18 | −0,52 | 0,00 | −18,6 % |
| SPY (repère) | −0,34 | 1,34 | 0,06 | −22,1 % |

**Décision (règle fixée).** C\* = inclinaison momentum 50/50, le meilleur Sharpe sur la sélection. Sur la confirmation, elle fait à peine mieux que l'équipondéré 50/50 (0,088 contre 0,080), mais son signal (le momentum) n'est pas confirmé. **C\* n'est donc pas confirmée, et la configuration par défaut de la phase B sera l'équipondéré 50/50.**

**Ce que ça signifie.**
- **Le classement des règles s'inverse entre les deux périodes.** Le 50/50, qui surpondère les Gagnants (part naturelle d'environ 32 %), est le meilleur en 2022 et le pire en 2023. C'est l'inverse pour le 1/N. La règle de répartition est un pari sur la période, pas une amélioration durable.
- **L'inclinaison vers le momentum** améliore l'équipondéré pour les 4 règles en 2022, mais l'effet devient nul ou négatif en 2023 (+0,01 / +0,01 / −0,03 / −0,03), ce qui est cohérent avec son IC proche de zéro.
- **Les repères s'inversent aussi.** Le minimum de variance est le meilleur en 2022 et le pire en 2023 ; SPY fait l'inverse.

## A3 : rapport d'exploration

`RAPPORT_EXPLORATION.md` est écrit. Conclusion : aucune variante ne tient de façon convaincante sur la confirmation. La configuration par défaut de la phase B sera l'équipondéré 50/50, et toutes les autres options restent accessibles.

---

# Phase B : intégration dans le projet principal
