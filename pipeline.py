import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from data import (get_sp500_tickers, download_prices, trading_sessions, add_daily_features,
                  resample_monthly, filter_liquidity, load_fama_french, add_rolling_betas)
from strategy import get_clusters_one_month, apply_roles_all_months, build_universe_by_month
from lstm import make_lstm_dataset, split_dataset, train_lstm, predict_test, evaluate_test
from backtest import run_backtest_lstm, get_spy_returns, perf_metrics


def prepare_data(refresh=False):
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



def main():
    df_roles, universes, prices_daily, sessions, cutoff, train, val, test, summary = prepare_data()
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
    metadata = dict(summary, parameters=params, test_metrics=evaluate_test(predictions))
    path = save_results(df_roles, universes, returns, allocations, periods, predictions, history, metadata)
    print(f"Résultats enregistrés : {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()

