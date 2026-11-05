import numpy as np
import pandas as pd
from cvxpy.error import SolverError
from pypfopt import EfficientFrontier, risk_models
from pypfopt.exceptions import OptimizationError

from config import LOOKBACK_MONTHS_DAILY, MAX_WEIGHT


def month_period(d, sessions):
    d = pd.Timestamp(d)
    following = d + pd.offsets.MonthEnd(1)
    entry = sessions[sessions > d]
    exit_dates = sessions[sessions > following]
    if entry.empty or exit_dates.empty:
        raise ValueError(f"Calendrier insuffisant pour le signal {d.date()}.")
    return entry[0], exit_dates[0]


def equal_weight_on_selection(tickers):
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0/len(tickers), index=tickers)


def optimize_with_predicted_mu(prices_lookback, mu_pred, max_weight=MAX_WEIGHT,
                              alloc_winners=0.5, alloc_stables=0.5,
                              winners=None, stables=None):
    tickers = [t for t in mu_pred.index if t in prices_lookback.columns]
    px = prices_lookback[tickers].dropna(axis=1, how="any")
    mu = mu_pred.reindex(px.columns)
    if not np.isfinite(mu).all():
        raise ValueError("Prévisions non finies fournies à l'optimiseur.")
    winners = [t for t in (winners or []) if t in px.columns]
    stables = [t for t in (stables or []) if t in px.columns]
    if set(winners) & set(stables) or set(winners + stables) != set(px.columns):
        raise ValueError("Chaque action doit appartenir à un seul des deux rôles retenus.")
    feasible = (len(winners)*max_weight >= alloc_winners - 1e-12
                and len(stables)*max_weight >= alloc_stables - 1e-12)
    if not feasible:
        w = pd.Series(dtype=float)
        w.attrs.update(method="cash", reason="Contraintes 50/50 et poids maximal irréalisables")
        return w

    S = risk_models.sample_cov(px, frequency=252) / 12
    if not np.isfinite(S.to_numpy()).all():
        raise ValueError("Covariance non finie : historique invalide.")
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight), solver="SCS")
    idx = list(px.columns)
    w_idx = [idx.index(t) for t in winners]
    s_idx = [idx.index(t) for t in stables]
    ef.add_constraint(lambda w: sum(w[i] for i in w_idx) == alloc_winners)
    ef.add_constraint(lambda w: sum(w[i] for i in s_idx) == alloc_stables)
    reason = ""
    try:
        ef.max_sharpe(risk_free_rate=0.0)
        w = pd.Series(ef.weights, index=px.columns, dtype=float)
        tol = 2e-5
        valid = (np.isfinite(w).all() and w.min() >= -tol and w.max() <= max_weight + tol
                 and abs(w.sum()-1) <= tol
                 and abs(w.reindex(winners).sum()-alloc_winners) <= tol
                 and abs(w.reindex(stables).sum()-alloc_stables) <= tol)
        if valid:
            w = w.clip(lower=0)
            w = w / w.sum()
            w.attrs.update(method="optimized", reason="")
            return w
        reason = "Poids retournés hors tolérance numérique"
    except (OptimizationError, SolverError, ValueError) as exc:
        reason = str(exc)

    w = pd.Series(0.0, index=px.columns)
    w.loc[winners] = alloc_winners / len(winners)
    w.loc[stables] = alloc_stables / len(stables)
    w.attrs.update(method="group_equal_weight", reason=reason)
    return w


def holding_prices(prices_daily, tickers, dates, retry_missing=False):
    px = prices_daily.reindex(index=dates, columns=tickers)
    bad = ~np.isfinite(px) | (px <= 0)
    if bad.any().any() and retry_missing:
        import yfinance as yf
        missing = bad.columns[bad.any()].tolist()
        try:
            raw = yf.download(missing, start=dates[0].strftime("%Y-%m-%d"),
                              end=(dates[-1]+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              auto_adjust=False, progress=False, threads=False)
            if not raw.empty:
                repaired = raw["Adj Close"]
                if isinstance(repaired, pd.Series):
                    repaired = repaired.to_frame(missing[0])
                for ticker in missing:
                    if ticker in repaired:
                        values = repaired[ticker].reindex(dates)
                        valid = bad[ticker] & values.notna() & (values > 0)
                        prices_daily.loc[dates[valid], ticker] = values.loc[valid]
                px = prices_daily.reindex(index=dates, columns=tickers)
                bad = ~np.isfinite(px) | (px <= 0)
        except (OSError, KeyError, ValueError) as exc:
            print(f"Nouvelle récupération impossible : {exc}")
    if bad.any().any():
        locations = np.argwhere(bad.to_numpy())[:5]
        details = ", ".join(f"{px.columns[j]} le {px.index[i].date()}" for i, j in locations)
        raise ValueError(f"Prix nécessaires manquants ou invalides : {details}. "
                         "Expérience non validée ; aucune position ni date n'a été supprimée.")
    return px


def portfolio_log_returns(px_hold, weights):
    if weights.empty:
        return pd.Series(0.0, index=px_hold.index[1:])
    weights = weights.reindex(px_hold.columns, fill_value=0.0)
    cash = max(0.0, 1.0 - weights.sum())
    equity = px_hold.div(px_hold.iloc[0]).dot(weights) + cash
    if (equity <= 0).any() or not np.isfinite(equity).all():
        raise ValueError("Valeur de portefeuille non positive ou non finie.")
    return np.log(equity).diff().iloc[1:]


def run_backtest_lstm(df_roles, pred_df, universes, prices_daily, test_dates,
                      sessions, cutoff, lookback_months=LOOKBACK_MONTHS_DAILY,
                      retry_missing=False):
    dates = pd.DatetimeIndex(sorted(test_dates))
    pred_dates = pd.DatetimeIndex(pred_df.index.get_level_values("Date").unique())
    if not pred_dates.isin(dates).all():
        raise ValueError("Le backtest a reçu des prédictions extérieures à Test.")
    if dates.empty:
        raise ValueError("Aucune date de signal Test.")
    expected = pd.period_range(dates[0], dates[-1], freq="M")
    if not dates.to_period("M").equals(expected):
        raise ValueError("Les signaux Test ne couvrent pas des mois consécutifs.")
    strat_daily, base_daily, allocations, periods = [], [], [], []
    for d in dates:
        start_hold, end_hold = month_period(d, sessions)
        if end_hold > pd.Timestamp(cutoff):
            continue
        winners = universes[d]["winners"]
        stables = universes[d]["stables"]
        selection = list(dict.fromkeys(winners + stables))
        mu_pred = (pred_df.xs(d, level="Date")["pred_return_next"]
                   if d in pred_dates else pd.Series(dtype=float))
        mu_pred = mu_pred.reindex(selection).dropna()
        start_lb = d + pd.offsets.MonthBegin(1) - pd.DateOffset(months=lookback_months)
        lb_dates = sessions[(sessions >= start_lb) & (sessions <= d)]
        px_lb = prices_daily.reindex(index=lb_dates, columns=mu_pred.index)
        usable = ((px_lb > 0) & np.isfinite(px_lb)).all()
        px_lb = px_lb.loc[:, usable]
        mu_pred = mu_pred.reindex(px_lb.columns)
        w_opt = optimize_with_predicted_mu(px_lb, mu_pred, winners=winners, stables=stables)
        method, reason = w_opt.attrs["method"], w_opt.attrs["reason"]
        w_eq = equal_weight_on_selection(px_lb.columns)

        hold_dates = sessions[(sessions >= start_hold) & (sessions <= end_hold)]
        px_hold = holding_prices(prices_daily, list(px_lb.columns), hold_dates, retry_missing)
        strat_daily.append(portfolio_log_returns(px_hold, w_opt).rename("Strategy_LSTM_Opt"))
        base_daily.append(portfolio_log_returns(px_hold, w_eq).rename("Baseline_EqualWeight"))
        for t in px_lb.columns:
            allocations.append({"Date": d, "entry_date": start_hold, "exit_date": end_hold,
                                "Ticker": t, "role": df_roles.loc[(d, t), "role"],
                                "weight_opt": w_opt.get(t, 0.0), "weight_eq": w_eq.get(t, 0.0)})
        if w_opt.empty or w_eq.empty:
            allocations.append({"Date": d, "entry_date": start_hold, "exit_date": end_hold,
                                "Ticker": "CASH", "role": "Liquidités",
                                "weight_opt": float(w_opt.empty), "weight_eq": float(w_eq.empty)})
        periods.append({"Date": d, "entry_date": start_hold, "exit_date": end_hold,
                        "n_available": len(px_lb.columns), "method": method, "reason": reason})
    if not strat_daily:
        raise ValueError("Aucune période Test complète et observable.")
    s = pd.concat(strat_daily).sort_index()
    b = pd.concat(base_daily).sort_index()
    if s.index.has_duplicates or not s.index.equals(b.index):
        raise ValueError("Chevauchement ou désalignement des périodes de détention.")
    return s, b, pd.DataFrame(allocations), pd.DataFrame(periods)


def get_spy_returns(prices_daily, entry, exit_date, sessions, retry_missing=False):
    dates = sessions[(sessions >= entry) & (sessions <= exit_date)]
    spy = holding_prices(prices_daily, ["SPY"], dates, retry_missing)["SPY"]
    return np.log(spy).diff().iloc[1:].rename("SPY")


def perf_metrics(log_returns, freq=252):
    r = log_returns.dropna()
    if len(r) == 0:
        return pd.Series({"AnnReturn": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDrawdown": np.nan})
    ann_ret = np.exp(r.mean() * freq) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = (r.mean() * freq) / (ann_vol + 1e-12)
    equity = np.exp(r.cumsum())
    dd = equity / equity.cummax() - 1
    mdd = dd.min()
    return pd.Series({"AnnReturn": ann_ret, "AnnVol": ann_vol,
                      "Sharpe": sharpe, "MaxDrawdown": mdd})


# ---------------------------------------------------------------------------
# Constructions de portefeuille sous une règle de répartition entre rôles
# (equal_weight, tilt, max_sharpe, min_variance) et backtest générique
# ---------------------------------------------------------------------------
from strategy import group_bounds, group_totals  # noqa: E402

CONSTRUCTIONS = ["equal_weight", "tilt", "max_sharpe", "min_variance"]
WEIGHT_TOL = 2e-5


def cap_redistribute(w, cap):
    """Plafonne les poids à `cap` et redistribue l'excédent proportionnellement aux autres."""
    w = w.astype(float).copy()
    capped = False
    for _ in range(200):
        over = w > cap + 1e-12
        if not over.any():
            break
        capped = True
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = w < cap - 1e-12
        if not free.any():
            raise ValueError("Plafond irréalisable dans un groupe.")
        w[free] += excess * w[free] / w[free].sum()
    return w, capped


def rule_weights(winners, stables, rule, signal=None, strength=0.0, max_weight=MAX_WEIGHT):
    """Équipondéré (strength=0) ou inclinaison vers `signal` sous la règle de répartition.

    Dans chaque groupe : poids = part_du_groupe / n * (1 + strength * z), z = rang centré dans [-1, 1].
    Renvoie une série vide (liquidités) si la règle et le plafond sont incompatibles.
    """
    winners, stables = list(winners), list(stables)
    if not winners and not stables:
        w = pd.Series(dtype=float)
        w.attrs.update(method="cash", reason="Univers vide")
        return w
    totals = group_totals(len(winners), len(stables), rule)
    groups = [(winners + stables, 1.0)] if totals is None else [(winners, totals[0]), (stables, totals[1])]
    if any(t > 1e-12 and len(g) * max_weight < t - 1e-12 for g, t in groups):
        w = pd.Series(dtype=float)
        w.attrs.update(method="cash", reason="Règle de répartition et poids maximal irréalisables")
        return w
    parts, any_cap = [], False
    for g, t in groups:
        if t <= 1e-12 or not g:
            continue
        n = len(g)
        if signal is None or strength == 0 or n == 1:
            w = pd.Series(t / n, index=g)
        else:
            s = signal.reindex(g)
            if s.isna().any():
                raise ValueError("Signal manquant pour une action de l'univers.")
            z = 2 * (s.rank(method="average") - 1) / (n - 1) - 1
            w = t / n * (1 + strength * z)
        w, capped = cap_redistribute(w, max_weight)
        parts.append(w)
        any_cap |= capped
    w = pd.concat(parts)
    w.attrs.update(method="capped" if any_cap else "ok", reason="")
    return w


def _bounds_feasible(n_w, n_s, bounds, max_weight):
    if bounds is None:
        return (n_w + n_s) * max_weight >= 1 - 1e-12
    (lw, hw), (ls, hs) = bounds
    return (n_w * max_weight >= lw - 1e-12 and n_s * max_weight >= ls - 1e-12
            and min(n_w * max_weight, hw) + min(n_s * max_weight, hs) >= 1 - 1e-12)


def optimize_weights(prices_lookback, mu, winners, stables, construction, rule, max_weight=MAX_WEIGHT):
    """max_sharpe (avec `mu`) ou min_variance sous la règle de répartition.

    Avec construction="max_sharpe" et rule="50_50", reproduit optimize_with_predicted_mu.
    En cas d'échec du solveur : équipondéré selon la règle (method="group_equal_weight").
    """
    px = prices_lookback.dropna(axis=1, how="any")
    winners = [t for t in winners if t in px.columns]
    stables = [t for t in stables if t in px.columns]
    if set(winners) & set(stables) or set(winners + stables) != set(px.columns):
        raise ValueError("Chaque action doit appartenir à un seul des deux rôles retenus.")
    bounds = group_bounds(len(winners), len(stables), rule)
    if not _bounds_feasible(len(winners), len(stables), bounds, max_weight):
        w = pd.Series(dtype=float)
        w.attrs.update(method="cash", reason="Règle de répartition et poids maximal irréalisables")
        return w
    if construction == "max_sharpe":
        mu = mu.reindex(px.columns)
        if not np.isfinite(mu).all():
            raise ValueError("Prévisions non finies fournies à l'optimiseur.")
    elif construction == "min_variance":
        mu = None
    else:
        raise ValueError(f"Construction non optimisée : {construction}")
    S = risk_models.sample_cov(px, frequency=252) / 12
    if not np.isfinite(S.to_numpy()).all():
        raise ValueError("Covariance non finie : historique invalide.")
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight), solver="SCS")
    idx = list(px.columns)
    groups = [[idx.index(t) for t in winners], [idx.index(t) for t in stables]]
    if bounds is not None:
        for g, (lo, hi) in zip(groups, bounds):
            if not g:
                continue
            if lo == hi:
                ef.add_constraint(lambda w, g=g, lo=lo: sum(w[i] for i in g) == lo)
            else:
                ef.add_constraint(lambda w, g=g, lo=lo: sum(w[i] for i in g) >= lo)
                ef.add_constraint(lambda w, g=g, hi=hi: sum(w[i] for i in g) <= hi)
    reason = ""
    try:
        if construction == "max_sharpe":
            ef.max_sharpe(risk_free_rate=0.0)
        else:
            ef.min_volatility()
        w = pd.Series(ef.weights, index=px.columns, dtype=float)
        valid = bool(np.isfinite(w).all() and w.min() >= -WEIGHT_TOL and w.max() <= max_weight + WEIGHT_TOL
                     and abs(w.sum() - 1) <= WEIGHT_TOL)
        if valid and bounds is not None:
            for names, (lo, hi) in zip([winners, stables], bounds):
                share = w.reindex(names).sum()
                valid = valid and (lo - WEIGHT_TOL <= share <= hi + WEIGHT_TOL)
        if valid:
            w = w.clip(lower=0)
            w = w / w.sum()
            w.attrs.update(method="optimized", reason="")
            return w
        reason = "Poids retournés hors tolérance numérique"
    except (OptimizationError, SolverError, ValueError) as exc:
        reason = str(exc)
    w = rule_weights(winners, stables, rule, max_weight=max_weight)
    w.attrs.update(method="group_equal_weight" if not w.empty else "cash", reason=reason)
    return w


def run_backtest(df_roles, signals, universes, prices_daily, test_dates, sessions, cutoff,
                 construction, rule, signal_col=None, strength=0.5, name="Strategy",
                 lookback_months=LOOKBACK_MONTHS_DAILY, retry_missing=False):
    """Backtest mensuel générique.

    `signals` est indexé par (Date, Ticker) : ses lignes définissent les actions candidates
    de chaque mois (comme les prévisions dans run_backtest_lstm). `signal_col` est le rendement
    attendu (max_sharpe) ou la note de classement (tilt) ; ignoré par equal_weight et min_variance.
    """
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"Construction inconnue : {construction}")
    if construction in ("tilt", "max_sharpe") and signal_col is None:
        raise ValueError(f"La construction {construction} a besoin d'un signal.")
    dates = pd.DatetimeIndex(sorted(test_dates))
    daily, allocations, periods = [], [], []
    for d in dates:
        start_hold, end_hold = month_period(d, sessions)
        if end_hold > pd.Timestamp(cutoff):
            continue
        winners, stables = universes[d]["winners"], universes[d]["stables"]
        selection = list(dict.fromkeys(winners + stables))
        rows = signals.xs(d, level="Date")
        candidates = [t for t in selection if t in rows.index]
        start_lb = d + pd.offsets.MonthBegin(1) - pd.DateOffset(months=lookback_months)
        lb_dates = sessions[(sessions >= start_lb) & (sessions <= d)]
        px_lb = prices_daily.reindex(index=lb_dates, columns=candidates)
        px_lb = px_lb.loc[:, ((px_lb > 0) & np.isfinite(px_lb)).all()]
        tickers = list(px_lb.columns)
        w_in = [t for t in winners if t in tickers]
        s_in = [t for t in stables if t in tickers]
        sig = rows[signal_col].reindex(tickers) if signal_col is not None else None
        if construction in ("max_sharpe", "min_variance"):
            w = optimize_weights(px_lb, sig, w_in, s_in, construction, rule)
        else:
            w = rule_weights(w_in, s_in, rule, signal=sig, strength=strength if construction == "tilt" else 0.0)
        hold_dates = sessions[(sessions >= start_hold) & (sessions <= end_hold)]
        px_hold = holding_prices(prices_daily, tickers, hold_dates, retry_missing)
        daily.append(portfolio_log_returns(px_hold, w).rename(name))
        for t in tickers:
            allocations.append({"Date": d, "entry_date": start_hold, "exit_date": end_hold, "Ticker": t,
                                "role": df_roles.loc[(d, t), "role"], "weight": float(w.get(t, 0.0))})
        if w.empty:
            allocations.append({"Date": d, "entry_date": start_hold, "exit_date": end_hold, "Ticker": "CASH",
                                "role": "Liquidités", "weight": 1.0})
        periods.append({"Date": d, "entry_date": start_hold, "exit_date": end_hold, "n_available": len(tickers),
                        "n_winners": len(w_in), "n_stables": len(s_in),
                        "w_winners": float(w.reindex(w_in).sum()) if not w.empty else 0.0,
                        "method": w.attrs["method"], "reason": w.attrs["reason"]})
    if not daily:
        raise ValueError("Aucune période Test complète et observable.")
    r = pd.concat(daily).sort_index()
    if r.index.has_duplicates:
        raise ValueError("Chevauchement des périodes de détention.")
    return r, pd.DataFrame(allocations), pd.DataFrame(periods)


# ---------------------------------------------------------------------------
# Qualité du signal (IC) et découpage des résultats par période
# ---------------------------------------------------------------------------
def ic_by_month(signals, col, observed="observed"):
    """Corrélation de rang (Spearman) mensuelle entre un signal et le rendement observé."""
    ev = signals.dropna(subset=[observed, col])
    return ev.groupby(level="Date").apply(lambda g: g[col].corr(g[observed], method="spearman")).rename(col)


def ic_stats(ic):
    ic = pd.Series(ic, dtype=float).dropna()
    n = len(ic)
    sd = ic.std(ddof=1) if n > 1 else np.nan
    t = ic.mean() / sd * np.sqrt(n) if n > 1 and sd > 0 else np.nan
    return {"IC_mean": float(ic.mean()) if n else np.nan, "IC_pos_months": int((ic > 0).sum()),
            "IC_n_months": n, "IC_tstat": float(t)}


def label_days(index, periods, month_labels):
    """Rattache chaque jour de détention (entry, exit] au libellé du mois de signal correspondant."""
    out = pd.Series(index=index, dtype=object)
    for _, r in periods.iterrows():
        out[(index > r["entry_date"]) & (index <= r["exit_date"])] = month_labels[r["Date"]]
    if out.isna().any():
        raise ValueError("Des jours de détention ne sont rattachés à aucune période.")
    return out
