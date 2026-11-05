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
