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
