import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from data import (get_sp500_tickers, download_prices, trading_sessions, add_daily_features,
                  resample_monthly, filter_liquidity, load_fama_french, add_rolling_betas)
from strategy import (get_clusters_one_month, apply_roles_all_months, build_universe_by_month,
                      factor_signals, FACTOR_SIGNALS, LSTM_SIGNALS, ALL_SIGNALS, GROUP_RULES)
from lstm import (make_lstm_dataset, split_dataset, train_lstm, predict_test, evaluate_test,
                  enable_determinism, add_relative_target, walk_forward_blocks, split_at, predict_ensemble)
from backtest import (run_backtest_lstm, get_spy_returns, perf_metrics, run_backtest, CONSTRUCTIONS,
                      ic_by_month, ic_stats, label_days)


def prepare_data(refresh=False, return_dataset=False):
    np.random.seed(config.RANDOM_SEED)
    print("Univers S&P 500 et prix journaliers", flush=True)
    symbols = get_sp500_tickers(refresh)
    daily = download_prices(symbols, refresh)
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
    fama_french = load_fama_french()
    df1 = add_rolling_betas(df1, fama_french)

    print("Clustering, rôles et sélection", flush=True)
    features = [c for c in df1.columns if c not in config.CLUSTER_DROP]
    df_feat = df1[features].copy()
    counts = df_feat.groupby("Date").size()
    if (counts < config.N_CLUSTERS).any():
        raise ValueError("Un mois contient moins de quatre actions pour KMeans.")
    df_clustered = df_feat.groupby("Date", group_keys=False).apply(get_clusters_one_month)
    df_roles = apply_roles_all_months(df_clustered)
    universes = build_universe_by_month(df_roles)

    print("Séquences LSTM et découpage temporel", flush=True)
    ds, ml_features = make_lstm_dataset(df_roles, universes, prices_daily, sessions)
    train, val, test, split = split_dataset(ds)
    if len(features) != 14 or len(ml_features) != 18:
        raise ValueError(f"Features inattendues : clustering={len(features)}, LSTM={len(ml_features)}.")
    summary = {"status": "prepared_not_trained", "n_sequences": len(ds),
               "sequence_shape": [config.SEQ_LEN, len(ml_features)],
               "split": split, "features": ml_features,
               "symbols_requested": len(symbols), "symbols_with_prices": stocks.index.get_level_values("Ticker").nunique(),
               "symbols_without_prices": sorted(set(symbols) - set(stocks.index.get_level_values("Ticker"))),
               "data_cutoff": str(cutoff.date()),
               "last_complete_feature_month": str(monthly.index.get_level_values("Date").max().date())}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "preparation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(ds)} séquences de {config.SEQ_LEN} mois × {len(ml_features)} variables ; "
          f"Train : {len(train)}, Validation : {len(val)}, Test : {len(test)}.", flush=True)
    if return_dataset:
        return df_roles, universes, prices_daily, sessions, cutoff, train, val, test, summary, ds
    return df_roles, universes, prices_daily, sessions, cutoff, train, val, test, summary


def save_results(df_roles, universes, returns, allocations, periods, predictions, history, metadata):
    path = config.RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True)
    returns.to_csv(path / "daily_returns.csv", index_label="Date")
    pd.DataFrame({c: perf_metrics(returns[c]) for c in returns}).T.to_csv(path / "metrics.csv", index_label="Portfolio")
    allocations.to_csv(path / "allocations.csv", index=False)
    periods.to_csv(path / "periods.csv", index=False)
    predictions.to_csv(path / "test_predictions.csv")
    history.to_csv(path / "training_history.csv")
    clusters = df_roles.copy()
    selected = {(d, t) for d, u in universes.items() for t in u["winners"] + u["stables"]}
    clusters["selected"] = [idx in selected for idx in clusters.index]
    clusters.to_csv(path / "clusters.csv")
    metadata = dict(metadata, status="complete")
    (path / "parameters.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path



def run_single_split():
    """Ancienne méthode : découpage unique 70/15/15, un LSTM (RANDOM_SEED), max Sharpe 50/50."""
    df_roles, universes, prices_daily, sessions, cutoff, train, val, test, summary = prepare_data()
    if config.DETERMINISTIC:
        enable_determinism()
    model, scaler, history = train_lstm(train, val)
    predictions = predict_test(model, scaler, test)
    test_dates = pd.to_datetime(summary["split"]["test_dates"])
    strategy, baseline, allocations, periods = run_backtest_lstm(df_roles, predictions, universes, prices_daily, test_dates, sessions, cutoff, retry_missing=True)
    spy = get_spy_returns(prices_daily, periods["entry_date"].min(), periods["exit_date"].max(), sessions, retry_missing=True)
    if not strategy.index.equals(spy.index):
        raise ValueError("Les dates SPY et portefeuille diffèrent.")
    returns = pd.concat([strategy, baseline, spy], axis=1)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Les rendements contiennent des valeurs manquantes.")
    for idx, row in predictions.iterrows():
        ticker = idx[1]
        entry_date = row["entry_date"]
        exit_date = row["exit_date"]

        if exit_date <= cutoff and entry_date in prices_daily.index and exit_date in prices_daily.index:
            p0 = prices_daily.at[entry_date, ticker]
            p1 = prices_daily.at[exit_date, ticker]
            observed = p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0 else np.nan
            predictions.at[idx, "observed"] = observed

    params = {name: getattr(config, name) for name in [
        "start_date", "end_date", "N_CLUSTERS", "TOP_FRAC",
        "MIN_NAMES", "MAX_NAMES", "LOOKBACK_MONTHS_DAILY",
        "MAX_WEIGHT", "SEQ_LEN", "LSTM_EPOCHS", "BATCH_SIZE",
        "LR", "RANDOM_SEED", "BETA_WINDOW", "FACTOR_LAG_MONTHS"
    ]}
    metadata = dict(summary, parameters=params, test_metrics=evaluate_test(predictions), eval_mode="single_split")
    path = save_results(df_roles, universes, returns, allocations, periods, predictions, history, metadata)
    print(f"Résultats enregistrés : {path.relative_to(config.ROOT)}")


# ---------------------------------------------------------------------------
# Mode walk-forward
# ---------------------------------------------------------------------------
SIGNAL_LABELS = {"lstm_raw": "LSTM brut", "lstm_relative": "LSTM relatif", "momentum": "Momentum 12-1",
                 "low_vol": "Faible volatilité", "mom_lowvol": "Momentum + faible vol.",
                 "reversal_1m": "Retournement 1 mois", "low_beta": "Faible bêta", "quality": "Qualité (RMW)"}
CONSTRUCTION_LABELS = {"equal_weight": "Équipondéré", "tilt": "Inclinaison", "max_sharpe": "Max Sharpe",
                       "min_variance": "Minimum de variance"}
RULE_LABELS = {"50_50": "50/50", "none": "sans contrainte de groupe", "proportional": "proportionnel",
               "30_70": "groupes 30–70 %"}
EW_50_50_KEY = "equal_weight_50_50"


def strategy_spec(signal, construction, rule, kind):
    uses_signal = construction in ("tilt", "max_sharpe")
    key = "_".join(([signal] if uses_signal else []) + [construction, rule])
    label = (f"{SIGNAL_LABELS[signal]} · " if uses_signal else "") + \
        f"{CONSTRUCTION_LABELS[construction]} · {RULE_LABELS[rule]}"
    return {"key": key, "label": label, "signal": signal if uses_signal else None,
            "construction": construction, "rule": rule, "kind": kind}


def validate_config():
    if config.SIGNAL not in ALL_SIGNALS:
        raise ValueError(f"SIGNAL inconnu : {config.SIGNAL} (choix : {ALL_SIGNALS})")
    if config.CONSTRUCTION not in CONSTRUCTIONS:
        raise ValueError(f"CONSTRUCTION inconnue : {config.CONSTRUCTION} (choix : {CONSTRUCTIONS})")
    if config.GROUP_RULE not in GROUP_RULES:
        raise ValueError(f"GROUP_RULE inconnue : {config.GROUP_RULE} (choix : {GROUP_RULES})")
    for s, c, r in config.COMPARISONS:
        if s not in ALL_SIGNALS or c not in CONSTRUCTIONS or r not in GROUP_RULES:
            raise ValueError(f"Comparaison invalide : {(s, c, r)}")
    if any(s not in ALL_SIGNALS for s in config.IC_SIGNALS):
        raise ValueError("IC_SIGNALS contient un signal inconnu.")
    if not config.SEEDS:
        raise ValueError("SEEDS ne peut pas être vide.")


def build_specs():
    specs = [strategy_spec(config.SIGNAL, config.CONSTRUCTION, config.GROUP_RULE, "main")]
    specs += [strategy_spec(s, c, r, "comparison") for s, c, r in config.COMPARISONS]
    specs += [strategy_spec(None, "equal_weight", "50_50", "reference"),
              strategy_spec(None, "equal_weight", "none", "reference"),
              strategy_spec(None, "min_variance", "50_50", "reference")]
    unique = {}
    for s in specs:
        unique.setdefault(s["key"], s)  # la stratégie principale garde son statut si elle est aussi une référence
    return list(unique.values())


def run_walk_forward():
    validate_config()
    df_roles, universes, prices_daily, sessions, cutoff, _, _, _, summary, ds = prepare_data(return_dataset=True)
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds = add_relative_target(ds)
    specs = build_specs()
    needed = list(dict.fromkeys(list(config.IC_SIGNALS) + [s["signal"] for s in specs if s["signal"]]))
    lstm_needed = [s for s in LSTM_SIGNALS if s in needed]
    tf_version = enable_determinism() if (config.DETERMINISTIC and lstm_needed) else None
    blocks = walk_forward_blocks(ds, config.WF_FIRST_TEST, config.WF_BLOCK_MONTHS, cutoff)
    print(f"Walk-forward : {sum(len(b) for b in blocks)} mois de test en {len(blocks)} blocs ; "
          f"LSTM : {lstm_needed or 'aucun'} ; graines : {config.SEEDS}", flush=True)

    parts, schedule, train_logs = [], [], []
    for b, block in enumerate(blocks, start=1):
        T = block[0]
        test = ds[ds["Date"].isin(block)].copy()
        if test["Date"].min() < T:
            raise ValueError("Fuite d'information : prévision datée avant la date de réentraînement.")
        part = (test.set_index(["Date", "Ticker"]).sort_index()[["entry_date", "exit_date", "y"]]
                .rename(columns={"y": "observed"}))
        info = {"block": b, "test_months": [str(block[0].date()), str(block[-1].date())], "n_test": len(test)}
        if lstm_needed:
            train, val, split_info = split_at(ds, T, config.WF_VAL_FRAC, config.WF_MIN_VAL_MONTHS)
            info.update(split_info)
            for sig in lstm_needed:
                print(f"Bloc {b} ({T.date()}) : {SIGNAL_LABELS[sig]}, {len(config.SEEDS)} graines, "
                      f"train {len(train)} / val {len(val)} / test {len(test)}", flush=True)
                preds, logs = predict_ensemble(train, val, test, config.SEEDS,
                                               target="y" if sig == "lstm_raw" else "y_rel")
                part[sig] = preds["ensemble"].reindex(part.index)
                for s in config.SEEDS:
                    part[f"{sig}_seed{s}"] = preds[f"seed{s}"].reindex(part.index)
                train_logs.append(logs.assign(block=b, signal=sig))
                info[f"epochs_{sig}"] = {int(k): int(v) for k, v in logs.groupby("seed").size().items()}
        parts.append(part)
        schedule.append(info)

    signals = pd.concat(parts).sort_index()
    fs = factor_signals(df_roles, prices_daily, signals.index)
    for c in FACTOR_SIGNALS:
        if c in needed:
            signals[c] = fs[c]
    missing = signals[needed].isna().sum()
    if missing.any():
        raise ValueError(f"Signaux manquants : {missing[missing > 0].to_dict()}")
    for sig in needed:  # rendement attendu pour max Sharpe : prévision LSTM, ou signal centré sur le mois
        signals[f"mu_{sig}"] = (signals[sig] if sig in LSTM_SIGNALS
                                else signals[sig] - signals[sig].groupby(level="Date").transform("mean"))

    test_dates = pd.DatetimeIndex([d for blk in blocks for d in blk])
    returns, allocs, periods_by = {}, {}, {}
    for spec in specs:
        col = {"max_sharpe": f"mu_{spec['signal']}", "tilt": spec["signal"]}.get(spec["construction"])
        r, alloc, per = run_backtest(df_roles, signals, universes, prices_daily, test_dates, sessions, cutoff,
                                     spec["construction"], spec["rule"], signal_col=col,
                                     strength=config.TILT_STRENGTH, name=spec["key"],
                                     retry_missing=config.RETRY_MISSING_PRICES)
        returns[spec["key"]], allocs[spec["key"]], periods_by[spec["key"]] = r, alloc.assign(strategy=spec["key"]), per.assign(strategy=spec["key"])
        print(f"{spec['label']} : {per['method'].value_counts().to_dict()}", flush=True)
    main_key = specs[0]["key"]
    universe = allocs[main_key][["Date", "Ticker"]].reset_index(drop=True)
    for k, a in allocs.items():
        if not a[["Date", "Ticker"]].reset_index(drop=True).equals(universe):
            raise ValueError(f"Univers différent pour {k}.")
    main_periods = periods_by[main_key]
    spy = get_spy_returns(prices_daily, main_periods["entry_date"].min(), main_periods["exit_date"].max(),
                          sessions, retry_missing=config.RETRY_MISSING_PRICES)
    returns["SPY"] = spy
    returns = pd.concat(returns, axis=1)
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Rendements manquants ou non finis (dates désalignées ?).")

    selection_end = pd.Timestamp(config.SELECTION_END)
    month_period = {d: ("Sélection" if d <= selection_end else "Confirmation") for d in test_dates}
    month_block = {d: f"Bloc {b}" for b, blk in enumerate(blocks, start=1) for d in blk}
    day_period = label_days(returns.index, main_periods, month_period)
    day_block = label_days(returns.index, main_periods, month_block)
    method_counts = {k: p["method"].value_counts().to_dict() for k, p in periods_by.items()}
    metrics = pd.DataFrame({c: perf_metrics(returns[c]) for c in returns}).T
    for c in ["optimized", "group_equal_weight", "capped", "cash"]:
        metrics[f"n_{c}"] = [method_counts.get(k, {}).get(c, 0) for k in metrics.index]
    rows = []
    for labels in [day_period, day_block]:
        for p in labels.unique():
            for c in returns:
                rows.append({"Portfolio": c, "period": p, "n_days": int((labels == p).sum()),
                             **perf_metrics(returns[c][labels == p]).to_dict()})
    metrics_by_period = pd.DataFrame(rows)

    ic_cols = list(dict.fromkeys(needed))
    ic = pd.concat([ic_by_month(signals, c) for c in ic_cols], axis=1)
    ic_rows = []
    for c in ic_cols:
        groups = {"Total": test_dates,
                  **{p: [d for d in test_dates if month_period[d] == p] for p in ["Sélection", "Confirmation"]},
                  **{f"Bloc {b}": list(blk) for b, blk in enumerate(blocks, start=1)}}
        for p, months in groups.items():
            ic_rows.append({"signal": c, "period": p, **ic_stats(ic[c].reindex(months))})
    ic_summary = pd.DataFrame(ic_rows)

    ref = allocs.get(EW_50_50_KEY)
    allocations = allocs[main_key].rename(columns={"weight": "weight_opt"}).drop(columns="strategy")
    allocations["weight_eq"] = (ref.set_index(["Date", "Ticker"])["weight"].reindex(
        pd.MultiIndex.from_frame(allocations[["Date", "Ticker"]])).fillna(0.0).to_numpy() if ref is not None else np.nan)
    strategies = [dict(s) for s in specs] + [{"key": "SPY", "label": "SPY", "signal": None,
                                              "construction": None, "rule": None, "kind": "reference"}]
    params = {name: getattr(config, name) for name in [
        "start_date", "end_date", "N_CLUSTERS", "TOP_FRAC", "MIN_NAMES", "MAX_NAMES", "LOOKBACK_MONTHS_DAILY",
        "MAX_WEIGHT", "SEQ_LEN", "LSTM_EPOCHS", "BATCH_SIZE", "LR", "BETA_WINDOW", "FACTOR_LAG_MONTHS",
        "EVAL_MODE", "WF_FIRST_TEST", "WF_BLOCK_MONTHS", "WF_VAL_FRAC", "WF_MIN_VAL_MONTHS", "SELECTION_END",
        "SEEDS", "DETERMINISTIC", "SIGNAL", "CONSTRUCTION", "GROUP_RULE", "TILT_STRENGTH", "IC_SIGNALS"]}
    params["COMPARISONS"] = [list(c) for c in config.COMPARISONS]
    metadata = dict(summary, status="complete", eval_mode="walk_forward", parameters=params, main=main_key,
                    strategies=strategies, schedule=schedule, tensorflow=tf_version,
                    test_dates=[str(d.date()) for d in test_dates],
                    holding_start=str(returns.index.min().date()), holding_end=str(returns.index.max().date()))

    path = config.RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True)
    returns.to_csv(path / "daily_returns.csv", index_label="Date")
    metrics.to_csv(path / "metrics.csv", index_label="Portfolio")
    metrics_by_period.to_csv(path / "metrics_by_period.csv", index=False)
    allocations.to_csv(path / "allocations.csv", index=False)
    pd.concat(allocs.values()).to_csv(path / "allocations_all.csv", index=False)
    main_periods.drop(columns="strategy").to_csv(path / "periods.csv", index=False)
    pd.concat(periods_by.values()).to_csv(path / "periods_all.csv", index=False)
    signals.to_csv(path / "signals.csv")
    ic.to_csv(path / "ic_by_month.csv", index_label="Date")
    ic_summary.to_csv(path / "ic_summary.csv", index=False)
    if train_logs:
        pd.concat(train_logs, ignore_index=True).to_csv(path / "training_history.csv", index=False)
    clusters = df_roles.copy()
    selected = {(d, t) for d, u in universes.items() for t in u["winners"] + u["stables"]}
    clusters["selected"] = [idx in selected for idx in clusters.index]
    clusters.to_csv(path / "clusters.csv")
    (path / "parameters.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    pd.set_option("display.width", 200)
    table = metrics[["AnnReturn", "AnnVol", "Sharpe", "MaxDrawdown"]].copy()
    table.index = [next(s["label"] for s in strategies if s["key"] == k) for k in table.index]
    print("\n" + table.round(4).to_string())
    print(f"Résultats enregistrés : {path.relative_to(config.ROOT)}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline de recherche de portefeuille.")
    parser.add_argument("--legacy", action="store_true",
                        help="Ancienne méthode : découpage unique, un LSTM, max Sharpe 50/50.")
    parser.add_argument("--mode", choices=["walk_forward", "single_split"], default=None,
                        help="Remplace EVAL_MODE de config.py pour ce lancement.")
    args = parser.parse_args(argv)
    mode = "single_split" if (args.legacy or config.LEGACY_MODE) else (args.mode or config.EVAL_MODE)
    if mode == "single_split":
        run_single_split()
    elif mode == "walk_forward":
        run_walk_forward()
    else:
        raise ValueError(f"EVAL_MODE inconnu : {mode}")


if __name__ == "__main__":
    main()

