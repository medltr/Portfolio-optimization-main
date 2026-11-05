"""Étude de sensibilité aux graines du LSTM + comparaison à des références.

Compare, sur la même période de test et avec les mêmes contraintes
(10 % max par action, 50 % Gagnants / 50 % Stables) :
  1. LSTM + max Sharpe (stratégie actuelle), graines 0 à 4 ;
  2. Minimum de variance ;
  3. Équipondéré (a) version du pipeline (1/N) et (b) 50/50 par rôle ;
  4. SPY.

Aucun fichier existant n'est modifié : les données viennent du cache,
aucun appel réseau n'est fait (retry_missing=False) et les sorties vont dans
experiments/output/seed_study_<horodatage>/.

Usage (depuis n'importe quel dossier) :
    .venv/Scripts/python.exe Portfolio-optimization-main/experiments/seed_study.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
import lstm  # noqa: E402
from backtest import (holding_prices, month_period, perf_metrics,  # noqa: E402
                      portfolio_log_returns, run_backtest_lstm, get_spy_returns)
from data import (add_daily_features, add_rolling_betas, download_prices,  # noqa: E402
                  filter_liquidity, get_sp500_tickers, load_fama_french,
                  resample_monthly, trading_sessions)
from lstm import make_lstm_dataset, predict_test, split_dataset, train_lstm  # noqa: E402
from strategy import (apply_roles_all_months, build_universe_by_month,  # noqa: E402
                      get_clusters_one_month)

SEEDS = [0, 1, 2, 3, 4]
ALLOC_W, ALLOC_S = 0.5, 0.5
OUT_ROOT = Path(__file__).resolve().parent / "output"


# --------------------------------------------------------------------------
# 1. Préparation (copie de pipeline.prepare_data, sans écrire preparation.json)
# --------------------------------------------------------------------------
def require_cache():
    needed = [config.CACHE_DIR / "sp500.csv",
              config.CACHE_DIR / f"daily_{config.start_date}_{config.end_date}.csv.gz",
              config.FF_MONTHLY_PATH]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Cache incomplet, aucun téléchargement ne sera tenté : " + ", ".join(missing))


def prepare():
    require_cache()
    np.random.seed(config.RANDOM_SEED)
    symbols = get_sp500_tickers(refresh=False)
    daily = download_prices(symbols, refresh=False)
    cutoff = pd.Timestamp(config.end_date) - pd.Timedelta(days=1)
    sessions = trading_sessions(config.start_date, cutoff + pd.DateOffset(months=3))
    observed_sessions = sessions[sessions <= cutoff]
    prices_daily = daily["Adj Close"].unstack("Ticker").reindex(observed_sessions)
    prices_daily.index.name = "Date"
    stocks = daily[daily.index.get_level_values("Ticker").isin(symbols)]

    print("Indicateurs, mois complets et liquidité", flush=True)
    df = add_daily_features(stocks)
    monthly = resample_monthly(df, sessions, cutoff)
    df1 = filter_liquidity(monthly)
    print("Facteurs mensuels et RollingOLS", flush=True)
    df1 = add_rolling_betas(df1, load_fama_french())

    print("Clustering, rôles et sélection", flush=True)
    features = [c for c in df1.columns if c not in config.CLUSTER_DROP]
    df_feat = df1[features].copy()
    df_clustered = df_feat.groupby("Date", group_keys=False).apply(get_clusters_one_month)
    df_roles = apply_roles_all_months(df_clustered)
    universes = build_universe_by_month(df_roles)

    print("Séquences LSTM et découpage temporel", flush=True)
    ds, ml_features = make_lstm_dataset(df_roles, universes, prices_daily, sessions)
    train, val, test, split = split_dataset(ds)
    if len(features) != 14 or len(ml_features) != 18:
        raise ValueError(f"Features inattendues : clustering={len(features)}, LSTM={len(ml_features)}.")
    print(f"{len(ds)} séquences ; Train {len(train)}, Validation {len(val)}, Test {len(test)}.", flush=True)
    test_dates = pd.to_datetime(split["test_dates"])
    return df_roles, universes, prices_daily, sessions, cutoff, train, val, test, test_dates


# --------------------------------------------------------------------------
# 2. LSTM déterministe
# --------------------------------------------------------------------------
def enable_determinism():
    """Active le déterminisme TF. Toute erreur est remontée, jamais contournée."""
    import tensorflow as tf
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # signalé explicitement à l'utilisateur
        raise RuntimeError(f"Impossible d'activer le déterminisme TensorFlow : {exc!r}") from exc
    return tf.__version__


def train_and_predict(seed, train, val, test):
    import tensorflow as tf
    lstm.RANDOM_SEED = seed  # lu par build_lstm ; changement en mémoire uniquement
    np.random.seed(seed)
    try:
        model, scaler, history = train_lstm(train, val)
        preds = predict_test(model, scaler, test)
    except tf.errors.UnimplementedError as exc:
        raise RuntimeError(f"Une opération n'a pas d'implémentation déterministe (graine {seed}) : {exc}") from exc
    return preds, history


def ic_by_month(preds):
    ev = preds.dropna(subset=["observed"])
    return ev.groupby(level="Date").apply(
        lambda g: g["pred_return_next"].corr(g["observed"], method="spearman")).rename("IC")


def directional_accuracy(preds):
    ev = preds.dropna(subset=["observed"])
    return float((np.sign(ev["pred_return_next"]) == np.sign(ev["observed"])).mean())


# --------------------------------------------------------------------------
# 3. Références sur exactement les mêmes actions que la stratégie LSTM
# --------------------------------------------------------------------------
def min_variance_weights(px, winners, stables, max_weight=config.MAX_WEIGHT):
    """Même logique que backtest.optimize_with_predicted_mu, avec min_volatility()."""
    from cvxpy.error import SolverError
    from pypfopt import EfficientFrontier, risk_models
    from pypfopt.exceptions import OptimizationError

    px = px.dropna(axis=1, how="any")
    winners = [t for t in winners if t in px.columns]
    stables = [t for t in stables if t in px.columns]
    if set(winners) & set(stables) or set(winners + stables) != set(px.columns):
        raise ValueError("Chaque action doit appartenir à un seul des deux rôles retenus.")
    if not (len(winners) * max_weight >= ALLOC_W - 1e-12 and len(stables) * max_weight >= ALLOC_S - 1e-12):
        w = pd.Series(dtype=float)
        w.attrs.update(method="cash", reason="Contraintes 50/50 et poids maximal irréalisables")
        return w
    S = risk_models.sample_cov(px, frequency=252) / 12
    ef = EfficientFrontier(None, S, weight_bounds=(0.0, max_weight), solver="SCS")
    idx = list(px.columns)
    w_idx = [idx.index(t) for t in winners]
    s_idx = [idx.index(t) for t in stables]
    ef.add_constraint(lambda w: sum(w[i] for i in w_idx) == ALLOC_W)
    ef.add_constraint(lambda w: sum(w[i] for i in s_idx) == ALLOC_S)
    try:
        ef.min_volatility()
        w = pd.Series(ef.weights, index=px.columns, dtype=float)
        tol = 2e-5
        if (np.isfinite(w).all() and w.min() >= -tol and w.max() <= max_weight + tol
                and abs(w.sum() - 1) <= tol
                and abs(w.reindex(winners).sum() - ALLOC_W) <= tol
                and abs(w.reindex(stables).sum() - ALLOC_S) <= tol):
            w = w.clip(lower=0)
            w = w / w.sum()
            w.attrs.update(method="optimized", reason="")
            return w
        reason = "Poids retournés hors tolérance numérique"
    except (OptimizationError, SolverError, ValueError) as exc:
        reason = str(exc)
    return group_equal_weights(winners, stables, method="group_equal_weight", reason=reason)


def group_equal_weights(winners, stables, method="group_equal_weight", reason=""):
    w = pd.Series(0.0, index=list(winners) + list(stables))
    w.loc[winners] = ALLOC_W / len(winners)
    w.loc[stables] = ALLOC_S / len(stables)
    w.attrs.update(method=method, reason=reason)
    return w


def run_references(allocations, universes, prices_daily, sessions):
    """Min-variance et équipondéré 50/50 sur les actions retenues chaque mois par la stratégie LSTM."""
    minvar_daily, ew50_daily, log = [], [], []
    alloc = allocations[allocations["Ticker"] != "CASH"]
    for d, g in alloc.groupby("Date", sort=True):
        tickers = g["Ticker"].tolist()
        winners = [t for t in universes[d]["winners"] if t in tickers]
        stables = [t for t in universes[d]["stables"] if t in tickers]
        start_hold, end_hold = g["entry_date"].iloc[0], g["exit_date"].iloc[0]
        start_lb = d + pd.offsets.MonthBegin(1) - pd.DateOffset(months=config.LOOKBACK_MONTHS_DAILY)
        lb_dates = sessions[(sessions >= start_lb) & (sessions <= d)]
        px_lb = prices_daily.reindex(index=lb_dates, columns=tickers)

        w_mv = min_variance_weights(px_lb, winners, stables)
        feasible = (len(winners) * config.MAX_WEIGHT >= ALLOC_W - 1e-12
                    and len(stables) * config.MAX_WEIGHT >= ALLOC_S - 1e-12)
        if feasible:
            w_ew = group_equal_weights(winners, stables, method="equal_50_50")
        else:
            w_ew = pd.Series(dtype=float)
            w_ew.attrs.update(method="cash", reason="Contraintes 50/50 et poids maximal irréalisables")

        hold_dates = sessions[(sessions >= start_hold) & (sessions <= end_hold)]
        px_hold = holding_prices(prices_daily, tickers, hold_dates, retry_missing=False)
        minvar_daily.append(portfolio_log_returns(px_hold, w_mv).rename("MinVariance"))
        ew50_daily.append(portfolio_log_returns(px_hold, w_ew).rename("EqualWeight_50_50"))
        log.append({"Date": d, "n": len(tickers), "n_winners": len(winners), "n_stables": len(stables),
                    "minvar_method": w_mv.attrs["method"], "minvar_reason": w_mv.attrs["reason"],
                    "ew50_method": w_ew.attrs["method"],
                    "minvar_max_weight": float(w_mv.max()) if not w_mv.empty else 0.0,
                    "ew50_max_weight": float(w_ew.max()) if not w_ew.empty else 0.0})
    return (pd.concat(minvar_daily).sort_index(), pd.concat(ew50_daily).sort_index(), pd.DataFrame(log))


# --------------------------------------------------------------------------
# 4. Programme principal
# --------------------------------------------------------------------------
def main():
    out = OUT_ROOT / ("seed_study_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    print(f"Sorties : {out}", flush=True)

    tf_version = enable_determinism()
    print(f"Déterminisme TensorFlow {tf_version} activé.", flush=True)

    df_roles, universes, prices_daily, sessions, cutoff, train, val, test, test_dates = prepare()

    # --- Contrôle du déterminisme : graine 0 entraînée deux fois ---
    print("\n=== Contrôle du déterminisme : graine 0, entraînement n°1 ===", flush=True)
    p0a, h0a = train_and_predict(0, train, val, test)
    print("\n=== Contrôle du déterminisme : graine 0, entraînement n°2 ===", flush=True)
    p0b, h0b = train_and_predict(0, train, val, test)
    pa, pb = p0a["pred_return_next"].to_numpy(), p0b["pred_return_next"].to_numpy()
    determinism = {
        "predictions_identical": bool(p0a.index.equals(p0b.index) and np.array_equal(pa, pb)),
        "max_abs_diff_predictions": float(np.max(np.abs(pa - pb))),
        "history_identical": bool(h0a.equals(h0b)),
        "epochs_run_1": len(h0a), "epochs_run_2": len(h0b),
    }
    print(f"Déterminisme : {determinism}", flush=True)

    # --- Graines 0 à 4 ---
    seed_rows, ic_rows, daily = [], [], {}
    allocations0 = None
    for seed in SEEDS:
        if seed == 0:
            preds, history = p0a, h0a
        else:
            print(f"\n=== Graine {seed} ===", flush=True)
            preds, history = train_and_predict(seed, train, val, test)
        preds.to_csv(out / f"predictions_seed{seed}.csv")
        strat, base_a, allocations, periods = run_backtest_lstm(
            df_roles, preds, universes, prices_daily, test_dates, sessions, cutoff, retry_missing=False)
        name = f"LSTM_MaxSharpe_seed{seed}"
        daily[name] = strat.rename(name)
        if seed == 0:
            allocations0, periods0 = allocations, periods
            daily["EqualWeight_pipeline_1N"] = base_a.rename("EqualWeight_pipeline_1N")
        else:
            sel0 = allocations0[["Date", "Ticker"]].reset_index(drop=True)
            if not sel0.equals(allocations[["Date", "Ticker"]].reset_index(drop=True)):
                raise ValueError(f"Les actions éligibles diffèrent entre la graine 0 et la graine {seed}.")
        allocations.to_csv(out / f"allocations_seed{seed}.csv", index=False)
        ic = ic_by_month(preds)
        for d, v in ic.items():
            ic_rows.append({"seed": seed, "Date": d, "IC": v})
        m = perf_metrics(strat)
        seed_rows.append({"Portfolio": name, "seed": seed, **m.to_dict(),
                          "IC_mean": float(ic.mean()), "IC_pos_months": int((ic > 0).sum()),
                          "IC_n_months": int(ic.notna().sum()),
                          "DirAccuracy": directional_accuracy(preds),
                          "epochs": len(history),
                          "n_fallback_months": int((periods["method"] != "optimized").sum())})

    # --- Références ---
    print("\n=== Références : minimum de variance, équipondéré 50/50, SPY ===", flush=True)
    minvar, ew50, ref_log = run_references(allocations0, universes, prices_daily, sessions)
    ref_log.to_csv(out / "references_by_month.csv", index=False)
    spy = get_spy_returns(prices_daily, periods0["entry_date"].min(), periods0["exit_date"].max(),
                          sessions, retry_missing=False)
    daily.update({"MinVariance": minvar, "EqualWeight_50_50": ew50, "SPY": spy})

    returns = pd.concat(daily.values(), axis=1)
    ref_index = returns.index
    for c, s in daily.items():
        if not s.index.equals(ref_index):
            raise ValueError(f"Dates désalignées pour {c}.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Rendements non finis.")
    returns.to_csv(out / "daily_returns.csv", index_label="Date")

    # --- Tableau final ---
    seeds_df = pd.DataFrame(seed_rows)
    metric_cols = ["AnnReturn", "AnnVol", "Sharpe", "MaxDrawdown", "IC_mean", "IC_pos_months", "DirAccuracy"]
    agg = pd.DataFrame([
        {"Portfolio": "LSTM_MaxSharpe_mean", **seeds_df[metric_cols].mean().to_dict()},
        {"Portfolio": "LSTM_MaxSharpe_std", **seeds_df[metric_cols].std(ddof=1).to_dict()},
    ])
    refs = pd.DataFrame([{"Portfolio": c, **perf_metrics(returns[c]).to_dict()}
                         for c in ["MinVariance", "EqualWeight_50_50", "EqualWeight_pipeline_1N", "SPY"]])
    summary = pd.concat([seeds_df, agg, refs], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(ic_rows).pivot(index="Date", columns="seed", values="IC").to_csv(out / "ic_by_month.csv")

    info = {"determinism": determinism, "tensorflow": tf_version, "seeds": SEEDS,
            "test_signal_dates": [str(d.date()) for d in periods0["Date"]],
            "holding_start": str(returns.index.min().date()), "holding_end": str(returns.index.max().date()),
            "n_trading_days": len(returns), "max_weight": config.MAX_WEIGHT,
            "alloc_winners": ALLOC_W, "alloc_stables": ALLOC_S, "retry_missing": False}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Résumé ===")
    print(summary.drop(columns=["seed"]).set_index("Portfolio").round(4).to_string())
    print(f"\nRésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
