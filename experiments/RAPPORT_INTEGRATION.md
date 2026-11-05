# Rapport d'intégration (phase B)

*Pour un lecteur débutant. Ce document explique ce qui a changé dans le projet, comment s'en servir, et ce que donnent les résultats.*

## En bref

- Le projet évalue maintenant ses stratégies en **walk-forward** (réentraînement tous les 6 mois, 19 mois de test), avec un **ensemble de 5 LSTM** et un **entraînement reproductible**.
- On peut choisir dans `config.py` le **signal**, la **construction du portefeuille** et la **règle de répartition** entre Gagnants et Stables.
- Par défaut, le projet utilise l'**équipondéré 50/50**, parce qu'aucune variante choisie sur 2022 n'a été confirmée sur 2023 (voir `RAPPORT_EXPLORATION.md`).
- L'**ancienne méthode** reste disponible avec `python pipeline.py --legacy`, et elle redonne exactement les anciens résultats.
- Le tableau de bord affiche la comparaison de toutes les stratégies, les résultats par période et l'IC des signaux.
- 11 tests automatiques vérifient les règles de portefeuille et l'absence de fuite d'information. Ils passent tous.

## 1. Sauvegarde

Avant toute modification, un dépôt Git a été créé et l'état complet du projet enregistré (commit `d826717`). Pour voir tout ce qui a changé depuis :

```bash
git diff d826717
```

Pour revenir à l'état d'avant :

```bash
git checkout d826717 -- .
```

Les modifications de la phase B **ne sont pas encore enregistrées dans Git** : c'est à toi de décider si tu veux en faire un commit.

## 2. Ce qui a changé, fichier par fichier

| Fichier | Changements | Ce qui n'a pas changé |
|---|---|---|
| `config.py` | Nouvelles options : mode d'évaluation, walk-forward, graines, déterminisme, signal, construction, règle de répartition, inclinaison, signaux et stratégies affichés | Tous les anciens paramètres, dont les hyperparamètres du LSTM |
| `strategy.py` | `factor_signals` (momentum, retournement 1 mois, faible volatilité, faible bêta, qualité, momentum + faible volatilité), `low_volatility`, `group_totals`, `group_bounds` | Clustering, rôles, sélection des actions |
| `lstm.py` | La graine devient un paramètre de `build_lstm` et `train_lstm` (même valeur par défaut). Nouveaux : `enable_determinism`, `add_relative_target`, `walk_forward_blocks`, `split_at` (qui s'arrête avec une erreur s'il détecte une fuite), `predict_ensemble` | Architecture du réseau, préparation des séquences, ancien découpage |
| `backtest.py` | `rule_weights` (équipondéré / inclinaison), `optimize_weights` (max Sharpe / minimum de variance sous chaque règle), `run_backtest` (générique), `ic_by_month`, `ic_stats`, `label_days` | `optimize_with_predicted_mu`, `run_backtest_lstm`, `perf_metrics` : intacts |
| `pipeline.py` | `main()` choisit le mode. L'ancien déroulé devient `run_single_split()`, avec le déterminisme en plus. Nouveau : `run_walk_forward()`. Options en ligne de commande `--legacy` et `--mode` | La préparation des données et la sauvegarde de l'ancien mode |
| `app.py` | Choix du dossier de résultats. Pour le walk-forward : onglets « Comparaison », « Par période », « Signal (IC) », « Allocation », « Clustering », « Méthodologie » | Les onglets d'origine, utilisés pour les anciens résultats |
| `tests/test_portfolio.py` | Nouveau : 11 tests automatiques | — |
| `README.md` | Nouvelle description, utilisation et options | — |

## 3. Comment utiliser les nouvelles options

Lancer le projet avec la configuration de `config.py` (walk-forward par défaut) :

```bash
python pipeline.py
```

Lancer l'ancienne méthode (découpage unique, un LSTM, max Sharpe 50/50) :

```bash
python pipeline.py --legacy
```

Ouvrir le tableau de bord, puis choisir le dossier de résultats dans la barre latérale :

```bash
streamlit run app.py
```

Lancer les tests :

```bash
python -m unittest discover -s tests -v
```

**Les réglages principaux dans `config.py` :**

| Je veux… | Je règle… |
|---|---|
| l'ancienne méthode, sans utiliser la ligne de commande | `LEGACY_MODE = True` |
| changer le signal | `SIGNAL = "lstm_relative"` (ou `lstm_raw`, `momentum`, `low_vol`, `mom_lowvol`, `reversal_1m`, `low_beta`, `quality`) |
| changer la façon de répartir l'argent | `CONSTRUCTION = "tilt"` (ou `equal_weight`, `max_sharpe`, `min_variance`) |
| changer la répartition entre Gagnants et Stables | `GROUP_RULE = "none"` (ou `50_50`, `proportional`, `30_70`) |
| une inclinaison plus ou moins forte | `TILT_STRENGTH = 0.5` (0 = équipondéré) |
| moins ou plus de graines | `SEEDS = [0, 1, 2]` |
| aller plus vite, sans LSTM | retirer `lstm_raw` et `lstm_relative` de `IC_SIGNALS` et de `COMPARISONS` |
| changer la date du premier test ou la fréquence de réentraînement | `WF_FIRST_TEST`, `WF_BLOCK_MONTHS` |

Durée mesurée : environ 2 minutes pour l'ancienne méthode, environ 12 minutes pour le walk-forward avec les 40 LSTM des comparaisons. Sans LSTM, quelques minutes suffisent.

## 4. Vérifications effectuées

- **L'ancienne méthode est intacte.** `python pipeline.py --legacy` donne exactement les mêmes indicateurs que le run du 4 novembre (écart 0).
- **Le walk-forward intégré reproduit les études.** Les rendements jour par jour sont identiques à ceux de `experiments/` pour toutes les stratégies communes (écart 0, ou 2×10⁻¹⁴ pour le momentum + max Sharpe, ce qui revient au même).
- **Tests automatiques : 11 sur 11 réussis.** Ils vérifient :
  - que les poids somment à 100 % ;
  - que le plafond de 10 % est respecté ;
  - que les règles 50/50, sans contrainte, proportionnelle et 30–70 % sont respectées, pour l'équipondéré, l'inclinaison, le max Sharpe et le minimum de variance ;
  - que le portefeuille passe en liquidités si une règle est impossible à respecter ;
  - que la nouvelle optimisation est identique à l'ancienne ;
  - qu'il n'y a aucune fuite dans le calendrier walk-forward ;
  - que la cible relative est bien centrée chaque mois ;
  - que la faible volatilité n'utilise que le passé ;
  - que toutes les stratégies du dernier résultat réel respectent leurs règles.
- **Le tableau de bord démarre.**
  - Les 5 dossiers de résultats (1 walk-forward, 4 en découpage unique) se chargent sans erreur.
  - Le serveur Streamlit répond « ok » au contrôle de santé et la page s'affiche (HTTP 200).

**Un point découvert pendant les tests.** Le solveur d'optimisation (SCS) n'est pas parfaitement précis. Dans les portefeuilles max Sharpe et minimum de variance :
- un poids peut dépasser le plafond d'au plus **0,0014 point de pourcentage** (10,0014 % au lieu de 10 %) ;
- la part d'un groupe peut s'écarter de 50 % d'au plus **0,013 point** (49,987 %).

Cela existait déjà dans le projet d'origine : le run du 4 novembre avait un poids de 10,00013 %. Je n'ai pas modifié l'optimiseur, pour garder la reproduction exacte des anciens résultats. Les tests vérifient ces portefeuilles avec une tolérance explicite, et restent stricts pour l'équipondéré et l'inclinaison, qui sont exacts.

## 5. Résultats finaux

Lancement : `results/20261105_163456_565542`, 19 mois de test (398 jours de bourse, du 2 février 2022 au 1er septembre 2023), mêmes actions chaque mois pour toutes les stratégies.

**Sharpe sur toute la période et par sous-période**

| Stratégie | Total | Sélection (2022) | Confirmation (2023) | Pire baisse totale |
|---|---|---|---|---|
| **Équipondéré 50/50 ★ (par défaut)** | −0,27 | −0,40 | 0,08 | −25,2 % |
| Équipondéré sans contrainte de groupe (1/N) | −0,38 | −0,64 | 0,36 | −27,6 % |
| Minimum de variance 50/50 | 0,00 | 0,18 | −0,52 | −18,6 % |
| Momentum + inclinaison 50/50 | −0,15 | −0,24 | 0,09 | −21,1 % |
| Momentum + max Sharpe 50/50 | −0,10 | −0,02 | −0,32 | −16,8 % |
| LSTM relatif + inclinaison 50/50 | −0,48 | −0,63 | −0,08 | −28,9 % |
| LSTM brut + max Sharpe 50/50 (ancienne stratégie) | −0,77 | −0,79 | −0,78 | −34,8 % |
| SPY | 0,06 | −0,34 | 1,34 | −22,1 % |

**Qualité des signaux (statistique t de l'IC)**

| Signal | Sélection | Confirmation | Total |
|---|---|---|---|
| Momentum 12-1 | 1,28 | −0,19 | 1,07 |
| Momentum + faible volatilité | 1,15 | −0,08 | 0,97 |
| Faible volatilité | 0,74 | 0,14 | 0,74 |
| Retournement 1 mois | 0,33 | 0,00 | 0,32 |
| Qualité | −0,62 | 0,19 | −0,39 |
| Faible bêta | −0,05 | −1,00 | −0,55 |
| LSTM brut | −1,13 | −0,54 | −1,27 |
| LSTM relatif | −1,44 | −0,79 | −1,66 |

**Comment lire ces résultats**
- L'**ancienne stratégie** (LSTM + max Sharpe) est la pire, aussi bien en 2022 qu'en 2023. Son signal est négatif dans les deux périodes.
- La configuration par défaut, l'**équipondéré 50/50**, n'a pas été choisie parce qu'elle gagne. Elle a été choisie parce que **rien n'a fait mieux de façon confirmée**, et qu'elle ne dépend d'aucune prévision.
- Les stratégies qui brillent sur une période sont souvent mauvaises sur l'autre (minimum de variance, momentum + max Sharpe, 1/N). C'est exactement ce que la séparation sélection / confirmation permet de voir.
- Aucune stratégie ne bat SPY sur l'ensemble de la période, et toutes les statistiques t des signaux restent sous 2.

## 6. Limites (toujours valables)

- **Biais de survie :** la liste des actions est la composition actuelle du S&P 500.
- **Période courte :** 19 mois de test, dont seulement 7 de confirmation.
- **Aucun coût de transaction.**
- **Sharpe calculé avec un taux sans risque à 0 %.**
- **Imprécision numérique de l'optimiseur** (section 4), négligeable.

## 7. Pistes

1. Allonger l'historique et le test (`WF_FIRST_TEST`) et utiliser la composition historique du S&P 500.
2. Ajouter des coûts de transaction et mesurer la rotation du portefeuille.
3. N'adopter un nouveau signal que s'il est confirmé sur une période qu'on n'a jamais regardée.
4. Si l'on veut une optimisation exactement dans les bornes, ajouter une étape de correction finale des poids, en acceptant que les résultats changent très légèrement par rapport à l'ancienne méthode.
