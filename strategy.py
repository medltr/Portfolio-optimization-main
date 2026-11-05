import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import N_CLUSTERS, TOP_FRAC, MIN_NAMES, MAX_NAMES


def get_clusters_one_month(g, n_clusters=N_CLUSTERS, random_state=0):
    X = g.values
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)
    out = g.copy()
    out["cluster"] = labels
    return out


def add_cluster_roles(df_month):
    g = df_month.copy()
    feats = [c for c in g.columns if c != "cluster"]
    prof = g.groupby("cluster")[feats].mean(numeric_only=True)

    trend = 0.45*prof["return_12m"] + 0.35*prof["return_6m"] + 0.20*prof["return_3m"] + 0.05*(prof["rsi"]/100)
    risk  = 1.00*prof["GK"] + 1.00*prof["atr_pct"]
    quality = 1.00*prof["RMW"] - 0.30*prof["GK"]
    stress  = -1.00*prof["return_12m"] + 0.80*prof["GK"] + 0.20*((50 - prof["rsi"])/50)

    scores = pd.DataFrame({
        "TrendScore": trend,
        "RiskScore": risk,
        "QualityScore": quality,
        "StressScore": stress
    })

    role_map = {}
    cl_win = scores["TrendScore"].idxmax()
    role_map[cl_win] = "Gagnants dynamiques"

    remaining = scores.index.difference(role_map.keys())
    cl_stress = scores.loc[remaining, "StressScore"].idxmax()
    role_map[cl_stress] = "Sous pression"

    remaining = scores.index.difference(role_map.keys())
    stable_metric = scores.loc[remaining, "QualityScore"] - 0.5*scores.loc[remaining, "RiskScore"]
    cl_stable = stable_metric.idxmax()
    role_map[cl_stable] = "Stables de qualité"

    remaining = scores.index.difference(role_map.keys())
    for cl in remaining:
        role_map[cl] = "Volatils neutres"

    g["role"] = g["cluster"].map(role_map)
    g = g.join(scores, on="cluster")
    return g


def apply_roles_all_months(df_in):
    out = []
    for d in df_in.index.get_level_values("Date").unique().sort_values():
        g = df_in.xs(d, level=0).copy()
        g2 = add_cluster_roles(g)
        g2["Date"] = d
        out.append(g2)
    df_roles = (pd.concat(out)
                .reset_index()
                .set_index(["Date","Ticker"])
                .sort_index())
    return df_roles


def select_top_fraction(df_month, role_name, score_col, top_frac=TOP_FRAC, min_names=MIN_NAMES, max_names=MAX_NAMES):
    sub = df_month[df_month["role"] == role_name].copy().dropna(subset=[score_col])
    if sub.empty:
        return []
    sub = sub.sort_values(score_col, ascending=False)
    n = int(np.ceil(len(sub) * top_frac))
    n = max(min_names, n)
    n = min(max_names, n)
    n = min(len(sub), n)
    return sub.index[:n].tolist()


def build_universe_by_month(df_roles):
    universes = {}
    dates = df_roles.index.get_level_values("Date").unique().sort_values()
    for d in dates:
        m = df_roles.xs(d, level=0)
        winners = select_top_fraction(m, "Gagnants dynamiques", "TrendScore")
        stables = select_top_fraction(m, "Stables de qualité", "QualityScore")
        universes[d] = {"winners": winners, "stables": stables}
    return universes


# ---------------------------------------------------------------------------
# Signaux de classement (sans apprentissage) et règles de répartition entre rôles
# ---------------------------------------------------------------------------
FACTOR_SIGNALS = ["momentum", "reversal_1m", "low_vol", "low_beta", "quality", "mom_lowvol"]
LSTM_SIGNALS = ["lstm_raw", "lstm_relative"]
ALL_SIGNALS = LSTM_SIGNALS + FACTOR_SIGNALS
GROUP_RULES = ["50_50", "none", "proportional", "30_70"]


def low_volatility(prices_daily, index, window=252, min_obs=200):
    """-(écart-type des log-rendements journaliers sur les `window` dernières séances <= date du signal)."""
    lr = np.log(prices_daily.where(prices_daily > 0)).diff()
    vol = lr.rolling(window, min_periods=min_obs).std()
    out = []
    for d, t in index:
        pos = vol.index.searchsorted(d, side="right") - 1
        out.append(-vol.iloc[pos][t] if pos >= 0 and t in vol.columns else np.nan)
    return pd.Series(out, index=index, name="low_vol")


def factor_signals(df_roles, prices_daily, index):
    """Signaux calculés uniquement avec des données connues à la date du signal.

    Le sens de chaque signal est fixé à l'avance (une valeur élevée = action préférée) :
    momentum 12-1, retournement 1 mois (-return_1m), faible volatilité, faible bêta marché
    (-bêta Mkt-RF), qualité (+bêta RMW), moyenne des rangs momentum + faible volatilité.
    """
    roles = df_roles.reindex(index)
    s = pd.DataFrame(index=index)
    s["momentum"] = (1 + roles["return_12m"]) ** 12 / (1 + roles["return_1m"]) - 1
    s["reversal_1m"] = -roles["return_1m"]
    s["low_vol"] = low_volatility(prices_daily, index)
    s["low_beta"] = -roles["Mkt-RF"]
    s["quality"] = roles["RMW"]
    s["mom_lowvol"] = s[["momentum", "low_vol"]].groupby(level="Date").rank(pct=True).mean(axis=1)
    return s


def group_totals(n_winners, n_stables, rule):
    """Part fixe de chaque groupe pour l'équipondéré et l'inclinaison (None = pas de groupes)."""
    n = n_winners + n_stables
    if rule == "50_50":
        return 0.5, 0.5
    if rule == "proportional":
        return n_winners / n, n_stables / n
    if rule == "30_70":
        tw = min(max(n_winners / n, 0.3), 0.7)
        return tw, 1 - tw
    if rule == "none":
        return None
    raise ValueError(f"Règle de répartition inconnue : {rule}")


def group_bounds(n_winners, n_stables, rule):
    """Bornes (min, max) de la part des Gagnants et des Stables pour les optimiseurs (None = libre)."""
    if rule == "30_70":
        return (0.3, 0.7), (0.3, 0.7)
    totals = group_totals(n_winners, n_stables, rule)
    if totals is None:
        return None
    return (totals[0], totals[0]), (totals[1], totals[1])
