from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"

end_date = "2023-09-27"
start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=8)).strftime("%Y-%m-%d")
N_CLUSTERS = 4
TOP_FRAC = 0.50
MIN_NAMES = 10
MAX_NAMES = 60
LOOKBACK_MONTHS_DAILY = 12
MAX_WEIGHT = 0.10
SEQ_LEN = 12
LSTM_EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
RANDOM_SEED = 0

BETA_WINDOW = 24
FACTOR_LAG_MONTHS = 2
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
FF_START = "2017-01-01"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FF_MONTHLY_PATH = DATA_DIR / "fama_french_monthly.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

CLUSTER_DROP = ["Volume", "Open", "High", "Low", "Close", "Adj Close", "atr"]
ML_DROP = ["cluster", "role"]

# ---------------------------------------------------------------------------
# Évaluation et stratégie (voir experiments/RAPPORT_INTEGRATION.md)
# ---------------------------------------------------------------------------
# Mode d'évaluation :
#   "walk_forward" : réentraînement tous les WF_BLOCK_MONTHS mois sur tout le passé disponible
#   "single_split" : ancienne méthode (découpage unique 70/15/15, graine RANDOM_SEED,
#                    LSTM brut + max Sharpe 50/50) ; aussi accessible par `python pipeline.py --legacy`
EVAL_MODE = "walk_forward"
LEGACY_MODE = False              # True = force l'ancienne méthode (équivaut à EVAL_MODE = "single_split")

WF_FIRST_TEST = "2022-01-31"     # premier signal de test
WF_BLOCK_MONTHS = 6              # mois entre deux réentraînements
WF_VAL_FRAC = 0.20               # part des derniers mois disponibles utilisée en validation
WF_MIN_VAL_MONTHS = 4
SELECTION_END = "2022-12-31"     # signaux <= : période de sélection ; après : confirmation

SEEDS = [0, 1, 2, 3, 4]          # ensemble de LSTM : moyenne des prévisions des graines
DETERMINISTIC = True             # entraînement TensorFlow reproductible

# Signal : "lstm_raw", "lstm_relative", "momentum", "low_vol", "mom_lowvol",
#          "reversal_1m", "low_beta", "quality" (utilisé par "tilt" et "max_sharpe")
SIGNAL = "momentum"
# Construction : "equal_weight", "tilt", "max_sharpe", "min_variance"
CONSTRUCTION = "equal_weight"
# Répartition entre Gagnants et Stables : "50_50", "none", "proportional", "30_70"
GROUP_RULE = "50_50"
TILT_STRENGTH = 0.5              # inclinaison : poids de 0,5x à 1,5x le poids égal du groupe

# Signaux dont l'IC est affiché et stratégies calculées en comparaison (signal, construction, règle)
IC_SIGNALS = ["lstm_raw", "lstm_relative", "momentum", "low_vol", "mom_lowvol", "reversal_1m", "low_beta", "quality"]
COMPARISONS = [("lstm_raw", "max_sharpe", "50_50"),       # ancienne stratégie, évaluée en walk-forward
               ("lstm_relative", "tilt", "50_50"),
               ("momentum", "max_sharpe", "50_50"),
               ("momentum", "tilt", "50_50")]
RETRY_MISSING_PRICES = True      # nouvelle tentative de téléchargement si un prix de détention manque

if LEGACY_MODE:
    EVAL_MODE = "single_split"
