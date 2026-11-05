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
