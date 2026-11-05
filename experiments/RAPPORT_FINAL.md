# Rapport final : le LSTM aide-t-il à construire un meilleur portefeuille ?

*Écrit pour quelqu'un qui débute en finance et en machine learning. Les mots en **gras** à leur première apparition sont expliqués dans le glossaire à la fin.*

---

## En bref

1. **Le LSTM du projet ne sait pas prévoir quelles actions vont faire mieux que les autres.** Sur 19 mois de test, ses prévisions ont même été un peu « à l'envers » (IC moyen d'environ −0,10). Changer ce qu'on lui demande de prévoir n'y change rien.
2. **Le bon résultat observé au départ était un coup de chance.** Selon la **graine** aléatoire, le même modèle donnait un ratio de Sharpe entre 0,02 et 1,52 sur 6 mois.
3. **Le signal le plus simple, le momentum, est le seul à pointer dans la bonne direction** (positif 11 mois sur 19), mais trop faiblement pour qu'on puisse exclure le hasard.
4. **La méthode « max Sharpe » amplifie tout, le bon comme le mauvais.** Avec un mauvais signal, elle donne les pires portefeuilles. Une **inclinaison prudente** autour de l'équipondéré est plus sûre.
5. **Sur 2022–2023, aucune stratégie testée ne bat clairement le simple achat de l'indice (SPY) ni le portefeuille de minimum de variance.**

---

## 1. Le point de départ

### Ce que fait le projet

Chaque fin de mois :

1. **On choisit les actions.** On prend les ~150 actions les plus échangées du S&P 500, puis on les range en 4 groupes d'actions qui se ressemblent (**clustering**) d'après leurs rendements passés, leur volatilité et leur exposition à des **facteurs** de marché. Deux groupes sont gardés : les **Gagnants dynamiques** (fortes hausses récentes) et les **Stables de qualité**.
2. **On prévoit.** Un réseau de neurones **LSTM** regarde les 12 derniers mois de chaque action (18 variables par mois) et prévoit son rendement du mois suivant.
3. **On investit.** Un optimiseur « **max Sharpe** » choisit les poids qui donnent le meilleur rapport rendement prévu / risque, avec deux règles : **10 % maximum par action** et **50 % sur les Gagnants, 50 % sur les Stables**.
4. On garde le portefeuille un mois, puis on recommence.

### Le résultat initial

Le projet testait cette stratégie sur **6 mois seulement** (mars → août 2023). Deux lancements du même programme ont donné des résultats très différents :

| Lancement | Sharpe de la stratégie LSTM | Sharpe de l'équipondéré 1/N | Sharpe de SPY |
|---|---|---|---|
| 15 octobre | 0,64 | 1,33 | 2,20 |
| 4 novembre | 0,10 | 1,33 | 2,20 |

Dans les deux cas, la stratégie faisait moins bien qu'un simple partage égal de l'argent entre les mêmes actions. La question de départ était donc : **le LSTM apporte-t-il vraiment quelque chose ?**

---

## 2. Les expériences menées, et pourquoi

Toutes les expériences utilisent les mêmes données (en cache), les mêmes règles d'investissement et les mêmes réglages du LSTM que le projet. Aucun fichier du projet n'a été modifié. Tout le code est dans `experiments/`.

### Expérience 1 : regarder les prévisions de près

**Pourquoi :** avant de juger un portefeuille, il faut savoir si les prévisions sont bonnes.

**Ce qu'on a trouvé (sur les 6 mois de 2023) :**
- Le modèle devine le sens du mouvement (hausse ou baisse) **48 % du temps**, soit à peu près pile ou face. Pourtant, il prédit « hausse » une fois sur deux : ce n'est donc pas un modèle qui répète toujours la même chose.
- Dire simplement « hausse » à chaque fois aurait donné raison 60 % du temps, puisque 6 actions sur 10 ont monté.
- Les prévisions sont « timides » : elles varient deux à trois fois moins que la réalité. C'est typique d'un modèle qui n'a pas trouvé de vraie régularité et qui se rabat sur des valeurs proches de zéro.

### Expérience 2 : le rôle du hasard (`seed_study.py`)

**Pourquoi :** un réseau de neurones démarre avec des poids tirés au hasard. Si le résultat change beaucoup selon ce tirage, un seul lancement ne prouve rien.

**Ce qu'on a fait :** entraîner le même modèle avec 5 graines (0 à 4), sur exactement les mêmes données. On a aussi vérifié que le calcul est reproductible : la même graine redonne exactement les mêmes prévisions.

| | Sharpe | IC moyen |
|---|---|---|
| Graine 0 | 0,10 | −0,08 |
| Graine 1 | **1,52** | +0,10 |
| Graine 2 | 0,12 | −0,02 |
| Graine 3 | 0,92 | −0,02 |
| Graine 4 | 0,02 | −0,06 |
| **Moyenne (écart-type)** | **0,54 (0,66)** | **−0,02** |
| Équipondéré 50/50, sans prévision | 0,86 | — |

**Ce que ça signifie :** le résultat dépend surtout de la chance. L'écart-type (0,66) est plus grand que la moyenne (0,54). En moyenne, le LSTM fait moins bien que l'équipondéré 50/50, qui n'utilise aucune prévision.

### Expérience 3 : un test plus long et plus honnête (`walk_forward.py`)

**Pourquoi :** 6 mois, c'est trop court, et cette période (une forte hausse en 2023) n'est pas représentative. Il faut aussi éviter de dépendre d'une seule graine.

**Ce qu'on a fait :**
- Un **walk-forward** : tous les 6 mois, on réentraîne le modèle sur **tout le passé disponible à cette date, et rien d'autre**. On l'utilise ensuite pour les 6 mois suivants. Le test couvre ainsi **19 mois** (janvier 2022 → juillet 2023), avec la baisse de 2022 et la reprise de 2023.
- À chaque fois, on entraîne 5 LSTM (graines 0 à 4) et on fait la **moyenne de leurs prévisions** (un **ensemble**). Cela réduit la part de hasard.
- Des contrôles automatiques arrêtent le programme si une information du futur est utilisée par erreur (**fuite d'information**).

| Portefeuille | Rendement annuel | Sharpe | Pire baisse (max drawdown) |
|---|---|---|---|
| LSTM ensemble + max Sharpe | −16,7 % | **−0,77** | −34,8 % |
| Minimum de variance | 0,0 % | 0,00 | −18,6 % |
| Équipondéré 50/50 | −5,7 % | −0,27 | −25,2 % |
| SPY | +1,3 % | 0,06 | −22,1 % |

**Ce que ça signifie :** sur une période plus longue et plus variée, la stratégie LSTM est **la pire** de toutes. L'ensemble de 5 modèles ne la sauve pas : les 5 graines ont toutes un IC négatif.

### Expérience 4 : la qualité du signal, sans portefeuille (`signal_study.py`)

**Pourquoi :** un portefeuille mélange deux choses, la qualité de la prévision et la façon de s'en servir. Ici, on isole la première en mesurant seulement l'**IC**, c'est-à-dire si le signal classe correctement les actions de la meilleure à la pire.

**Les quatre signaux comparés :**
- **(a)** le LSTM actuel, qui prévoit le rendement brut ;
- **(b)** le même LSTM, mais qui prévoit le **rendement relatif** (le rendement de l'action moins la moyenne des actions du mois). L'idée : ce qui compte pour choisir, c'est de savoir quelle action fera *mieux que les autres*, pas si tout le marché va monter ;
- **(c)** le **momentum 12-1** : la performance de l'action sur les 12 derniers mois, sans le dernier. C'est un signal classique en finance, qui ne demande aucun apprentissage ;
- **(d)** une **régression Ridge**, un modèle linéaire simple, sur la même cible relative et les mêmes données que le LSTM.

| Signal | IC moyen | Mois où l'IC est positif | Statistique t |
|---|---|---|---|
| (a) LSTM, rendement brut | −0,096 | 7 / 19 | −1,27 |
| (b) LSTM, rendement relatif | −0,109 | 6 / 19 | −1,66 |
| **(c) Momentum 12-1** | **+0,073** | **11 / 19** | **+1,07** |
| (d) Ridge, rendement relatif | −0,022 | 9 / 19 | −0,28 |

**Ce que ça signifie :**
- Changer la cible du LSTM n'aide pas.
- Le Ridge, réglé sur les données de validation, choisit la régularisation maximale, ce qui revient à dire « je ne trouve rien ». Son IC est proche de zéro.
- Le momentum est le seul signal dans le bon sens. Mais une statistique t de 1,07 est loin du seuil habituel d'environ 2 : on ne peut pas exclure qu'il s'agisse de chance.
- Les LSTM classent les actions plutôt **à l'inverse** du momentum (corrélation d'environ −0,35 à −0,41), ce qui est cohérent avec leur IC négatif. Nous n'avons **pas** inversé le signal du LSTM pour en profiter : ce serait choisir une stratégie en regardant le test, une forme de tricherie.

### Expérience 5 : du signal au portefeuille (`portfolio_study.py`)

**Pourquoi :** savoir si un bon ou un mauvais signal se retrouve dans le résultat, et quelle façon de construire le portefeuille est la plus raisonnable.

**Ce qu'on a fait :** une règle fixée **avant** de voir les résultats désignait le meilleur signal (la statistique t la plus haute), soit le momentum. On l'a testé avec le LSTM relatif, de deux façons :
- **(i) max Sharpe**, comme dans le projet. En pratique, cet optimiseur met environ 85 à 90 % de l'argent sur 9 actions sur 40, souvent toutes au plafond de 10 %.
- **(ii) inclinaison prudente** : on part de l'équipondéré 50/50, puis on donne jusqu'à 1,5 fois plus de poids aux actions les mieux classées et jusqu'à 2 fois moins aux moins bien classées, sans changer la règle des 50/50.

| Portefeuille (19 mois) | Rendement annuel | Volatilité | Sharpe | Pire baisse |
|---|---|---|---|---|
| Momentum + max Sharpe | −1,9 % | 19,8 % | −0,10 | **−16,8 %** |
| Momentum + inclinaison | −3,2 % | 21,2 % | −0,15 | −21,1 % |
| LSTM relatif + max Sharpe | −16,1 % | 22,2 % | −0,79 | −31,8 % |
| LSTM relatif + inclinaison | −10,0 % | 22,2 % | −0,48 | −28,9 % |
| LSTM actuel + max Sharpe | −16,7 % | 23,8 % | −0,77 | −34,8 % |
| Minimum de variance | 0,0 % | **18,0 %** | 0,00 | −18,6 % |
| Équipondéré 50/50 | −5,7 % | 21,7 % | −0,27 | −25,2 % |
| Équipondéré 1/N | −7,5 % | 20,6 % | −0,38 | −27,6 % |
| SPY | **+1,3 %** | 20,8 % | **0,06** | −22,1 % |

**Sharpe par période de 6 mois :**

| Portefeuille | S1 2022 (baisse) | S2 2022 | S1 2023 (reprise) |
|---|---|---|---|
| Momentum + inclinaison | −0,94 | 0,67 | 0,50 |
| Équipondéré 50/50 (point de départ de l'inclinaison) | −1,14 | 0,55 | 0,44 |
| LSTM relatif + inclinaison | −1,39 | 0,37 | 0,20 |
| SPY | −0,72 | 0,08 | **1,75** |
| Minimum de variance | **−0,48** | **0,98** | −0,48 |

*(Le dernier bloc, juillet 2023, ne contient qu'un mois et n'est pas montré ; il figure dans `summary_by_block.csv`.)*

**Ce que ça signifie :**
- **Le résultat suit la qualité du signal.** Incliner l'équipondéré vers le momentum l'améliore dans chacune des trois périodes. L'incliner vers le LSTM le dégrade dans chacune des trois périodes.
- **Le max Sharpe est un amplificateur.** En concentrant l'argent sur quelques actions, il multiplie l'effet du signal. Avec un mauvais signal, c'est la catastrophe (−32 à −35 % au pire moment). Avec le momentum, il obtient la plus petite baisse de tous les portefeuilles, mais un Sharpe encore inférieur au minimum de variance et à SPY.
- **Aucun portefeuille ne gagne partout.** Le minimum de variance brille en 2022 (marché en baisse) et déçoit en 2023. SPY fait l'inverse. Le « meilleur » portefeuille dépend surtout de la période.

---

## 3. Ce qui marche, ce qui ne marche pas

**Ce qui marche**
- **La méthode d'évaluation.** Le walk-forward sans fuite, le déterminisme, l'ensemble de graines et les règles fixées à l'avance permettent de distinguer un vrai effet d'un coup de chance. C'est le principal acquis de cette étude.
- **L'inclinaison prudente** traduit fidèlement le signal sans prendre de gros risques. C'est une bonne façon de tester un nouveau signal.
- **Le momentum**, un signal simple et connu, va dans le bon sens. Il n'est pas prouvé statistiquement, mais il reste un point de comparaison utile.

**Ce qui ne marche pas**
- **Le LSTM, quelle que soit la cible** : IC négatif sur 19 mois, pour chacune des 5 graines.
- **Le max Sharpe avec des prévisions bruitées** : trop concentré, il transforme un signal médiocre en grosses pertes.
- **Juger sur 6 mois et une seule graine** : les conclusions s'inversent selon le tirage.

---

## 4. Les limites de cette étude

- **Biais de survie.** La liste des actions vient de la composition **actuelle** du S&P 500 (Wikipédia). Les entreprises sorties de l'indice entre 2015 et 2023, souvent parce qu'elles allaient mal, sont absentes. Les résultats historiques sont donc probablement trop optimistes pour **toutes** les stratégies sur actions.
- **Période courte.** 19 mois (398 jours de bourse) et environ 40 actions par mois, c'est peu. Aucun écart d'IC n'est **statistiquement significatif** (toutes les statistiques t sont entre −1,7 et +1,1). Les Sharpe sur 6 mois sont encore plus incertains.
- **Pas de coûts de transaction.** On réajuste le portefeuille chaque mois sans payer de frais. Le max Sharpe, qui change souvent ses quelques grosses positions, serait le plus pénalisé par des frais réels.
- **Taux sans risque ignoré.** Le Sharpe est calculé avec un taux sans risque de 0 %. En 2022–2023, les placements sans risque rapportaient jusqu'à environ 5 % par an. Face à cela, toutes les stratégies paraissent encore moins attractives.
- **Un seul marché, une seule période.** Grandes actions américaines, 2022–2023 : une hausse brutale des taux, puis une reprise portée par quelques géants. Rien ne garantit les mêmes conclusions ailleurs.
- **Choix fixés à l'avance mais arbitraires.** La force de l'inclinaison (0,5), la grille du Ridge et la taille de la validation (20 %) ont été choisies avant les tests et n'ont jamais été ajustées. D'autres valeurs donneraient d'autres chiffres, mais les choisir après coup serait tricher.

---

## 5. Pistes pour la suite

1. **Allonger le test.** Le script `walk_forward.py` accepte déjà `--first-test 2021-07-31` (25 mois). Mieux encore : télécharger un historique plus long pour tester sur 10 ans ou plus.
2. **Corriger le biais de survie** avec la composition historique du S&P 500 (les entreprises présentes dans l'indice à chaque date).
3. **Ajouter des coûts de transaction** (par exemple 0,1 % par opération) et mesurer la **rotation** du portefeuille.
4. **Utiliser le taux sans risque** dans le calcul du Sharpe.
5. **Prendre le momentum comme référence obligatoire.** Un modèle d'apprentissage ne vaut la peine que s'il bat ce signal gratuit.
6. **Élargir l'univers** (plus d'actions par mois) pour rendre l'IC moins bruité.
7. **Prévoir le risque plutôt que le rendement.** La volatilité est bien plus prévisible que le rendement. Le minimum de variance, qui ne prévoit aucun rendement, a d'ailleurs été l'un des meilleurs ici.
8. **Préférer l'inclinaison au max Sharpe** tant qu'aucun signal n'a fait ses preuves.

---

## 6. Glossaire

- **Action / rendement :** une action est une part d'entreprise. Son rendement sur un mois est la variation de son prix, dividendes compris (+5 % = le prix a monté de 5 %).
- **S&P 500 / SPY :** le S&P 500 est un indice des ~500 plus grandes entreprises américaines. SPY est un fonds qui le reproduit : l'acheter revient à « acheter le marché ».
- **Portefeuille / poids :** un ensemble d'actions. Le poids d'une action est la part de l'argent qu'on y met (10 % = un dixième).
- **Équipondéré (1/N, 50/50) :** la même somme sur chaque action (1/N), ou la moitié de l'argent répartie également entre les Gagnants et l'autre moitié entre les Stables (50/50).
- **Volatilité :** l'ampleur des variations. Plus elle est élevée, plus le portefeuille monte et descend fortement. C'est la mesure habituelle du risque.
- **Ratio de Sharpe :** le rendement divisé par la volatilité, soit « combien on gagne par unité de risque ». Au-dessus de 1, c'est très bon ; autour de 0, on ne gagne rien ; négatif, on perd.
- **Max drawdown (pire baisse) :** la plus forte chute, du plus haut au plus bas, pendant la période. −20 % : à un moment, on avait perdu 20 % depuis le sommet précédent.
- **Max Sharpe :** une méthode d'optimisation qui choisit les poids donnant le meilleur Sharpe *prévu*, d'après les rendements prévus et les risques passés.
- **Minimum de variance :** une méthode qui choisit les poids donnant le portefeuille le moins volatil possible, sans aucune prévision de rendement.
- **Inclinaison (tilt) :** on part d'un portefeuille simple (ici l'équipondéré) et on augmente un peu les actions préférées par le signal, en diminuant un peu les autres.
- **Clustering :** une méthode qui range automatiquement des éléments en groupes d'éléments qui se ressemblent (ici, KMeans).
- **Facteurs (Fama-French) :** de grandes tendances qui expliquent une partie des rendements de toutes les actions : le marché, la taille, la valeur, la rentabilité, l'investissement.
- **Momentum 12-1 :** la performance d'une action sur les 12 derniers mois, sans le dernier mois. L'idée observée en finance : ce qui a monté a tendance à continuer un peu.
- **LSTM :** un type de réseau de neurones conçu pour les séquences (ici, 12 mois d'historique). Il peut apprendre des motifs complexes, mais il a besoin de beaucoup de données.
- **Régression Ridge :** un modèle linéaire (une somme pondérée des variables) avec une pénalité qui empêche les poids de devenir trop grands. Plus la pénalité (alpha) est forte, plus le modèle est prudent.
- **Cible brute / cible relative :** ce qu'on demande au modèle de prévoir. Brute : le rendement de l'action. Relative : son rendement moins la moyenne des actions du mois, c'est-à-dire « fera-t-elle mieux que les autres ? ».
- **Graine (seed) :** le nombre qui fixe les tirages au hasard d'un programme. Même graine, même résultat ; graine différente, résultat différent.
- **Déterminisme :** la garantie qu'un calcul relancé donne exactement le même résultat.
- **Ensemble :** la moyenne des prévisions de plusieurs modèles. Le hasard propre à chacun se compense en partie.
- **Entraînement / validation / test :** trois parts des données. Le modèle apprend sur l'entraînement. La validation sert à décider quand arrêter l'apprentissage ou à choisir un réglage. Le test sert uniquement à juger, à la fin.
- **Arrêt anticipé (early stopping) :** on arrête l'apprentissage quand le modèle cesse de progresser sur la validation, pour éviter qu'il « apprenne par cœur ».
- **Walk-forward :** une façon de tester qui imite la réalité. À chaque date, on n'utilise que le passé pour entraîner, puis on teste sur la période suivante, et ainsi de suite.
- **Fuite d'information :** une erreur où le modèle utilise, sans le vouloir, une information du futur. Les résultats paraissent alors bien meilleurs qu'ils ne le seraient en réalité.
- **IC (coefficient d'information) :** chaque mois, la corrélation de rang (Spearman) entre le classement des actions par le signal et leur classement réel. +1 : classement parfait ; 0 : aucun lien ; −1 : classement inversé. En pratique, un IC durable de +0,05 est déjà intéressant.
- **Statistique t :** elle mesure si une moyenne (ici l'IC moyen) est assez loin de zéro pour ne pas être due au hasard. En dessous d'environ 2 en valeur absolue, on ne peut pas conclure.
- **Statistiquement significatif :** un résultat trop net pour s'expliquer raisonnablement par le hasard.
- **Biais de survie :** une erreur qui consiste à n'étudier que les « survivants », par exemple les entreprises encore dans l'indice aujourd'hui, ce qui embellit le passé.
- **Coûts de transaction :** les frais payés à chaque achat ou vente (commissions, écart entre prix d'achat et de vente).
- **Rotation (turnover) :** la part du portefeuille échangée à chaque réajustement. Plus elle est forte, plus les coûts sont élevés.

---

## 7. Où trouver les détails

| Fichier | Contenu |
|---|---|
| `experiments/JOURNAL.md` | Les règles fixées à l'avance et le résumé de chaque étape |
| `experiments/seed_study.py` → `output/seed_study_20261104_222552/` | Expérience 2 (graines) |
| `experiments/walk_forward.py` → `output/walk_forward_20261104_230330/` | Expérience 3 (walk-forward) |
| `experiments/signal_study.py` → `output/signal_study_20261105_120327/` | Expérience 4 (qualité des signaux) |
| `experiments/portfolio_study.py` → `output/portfolio_study_20261105_132550/` | Expérience 5 (portefeuilles) |

Chaque dossier de sortie contient un `summary.csv` (ou `signal_summary.csv`), les rendements jour par jour et un `run_info.json` qui décrit exactement comment le calcul a été fait.
