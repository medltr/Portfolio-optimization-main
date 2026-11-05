"""Walk-forward du LSTM avec fenêtre croissante et ensemble de 5 graines.

Tous les `--block` mois (6 par défaut), à la date de signal T :
  - entraînement + validation (arrêt anticipé) sur les seuls exemples dont la
    cible est entièrement connue à T (exit_date <= T), validation = ~20 % des
    derniers mois (au moins 4), avec purge comme dans lstm.split_dataset ;
  - 5 LSTM (graines 0 à 4), hyperparamètres de config.py inchangés ;
  - prévision des `--block` mois de signaux suivants (moyenne des 5 modèles).

Compare sur toute la période : LSTM ensemble + max Sharpe, LSTM graine 0 seule,
minimum de variance, équipondéré 50/50, équipondéré 1/N du pipeline et SPY.

Aucun fichier existant n'est modifié ; cache uniquement ; sorties dans
experiments/output/walk_forward_<horodatage>/.

Usage :
    .venv/Scripts/python.exe Portfolio-optimization-main/experiments/walk_forward.py
    .venv/Scripts/python.exe Portfolio-optimization-main/experiments/walk_forward.py --first-test 2021-07-31
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_study as ss  # noqa: E402  (ajoute aussi la racine du projet au sys.path)

import config  # noqa: E402
from backtest import get_spy_returns, perf_metrics, run_backtest_lstm  # noqa: E402
from lstm import make_lstm_dataset  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
VAL_FRAC, MIN_VAL_MONTHS = 0.20, 4
OUT_ROOT = Path(__file__).resolve().parent / "output"


# --------------------------------------------------------------------------
# Calendrier et découpages sans fuite
# --------------------------------------------------------------------------
def build_schedule(ds, first_test, block, cutoff):
    dates = pd.DatetimeIndex(sorted(ds["Date"].unique()))
    if first_test not in dates:
        raise ValueError(f"{first_test.date()} n'est pas une date de signal du dataset.")
    # signaux de test dont la détention se termine avant la date limite des données
    last_exit = ds.groupby("Date")["exit_date"].max()
    test_dates = dates[(dates >= first_test) & (last_exit.reindex(dates).to_numpy() <= cutoff)]
    expected = pd.period_range(test_dates[0], test_dates[-1], freq="M")
    if not test_dates.to_period("M").equals(expected):
        raise ValueError("Les mois de test ne sont pas consécutifs.")
    return [test_dates[i:i + block] for i in range(0, len(test_dates), block)]


def split_at(ds, T):
    """Entraînement / validation avec toutes les données dont la cible est connue à T."""
    avail = ds[(ds["exit_date"] <= T) & np.isfinite(ds["y"])].copy()
    months = pd.DatetimeIndex(sorted(avail["Date"].unique()))
    n_val = max(MIN_VAL_MONTHS, int(round(VAL_FRAC * len(months))))
    if len(months) - n_val < 6:
        raise ValueError(f"Trop peu de mois d'entraînement avant {T.date()}.")
    val_months, train_months = months[-n_val:], months[:-n_val]
    train = avail[avail["Date"].isin(train_months) & (avail["exit_date"] <= val_months[0])].copy()
    val = avail[avail["Date"].isin(val_months)].copy()
    # contrôles stricts de l'ordre du temps
    assert train["exit_date"].max() <= val["Date"].min(), "Fuite : cible d'entraînement après le début de la validation."
    assert val["exit_date"].max() <= T, "Fuite : cible de validation après la date de réentraînement."
    info = {"T": str(T.date()), "n_train": len(train), "n_val": len(val),
            "train_months": [str(train["Date"].min().date()), str(train["Date"].max().date())],
            "val_months": [str(val_months[0].date()), str(val_months[-1].date())],
            "n_train_months": int(train["Date"].nunique()), "n_val_months": len(val_months),
            "last_train_target": str(train["exit_date"].max().date()),
            "last_val_target": str(val["exit_date"].max().date())}
    return train, val, info


# --------------------------------------------------------------------------
# Indicateurs
# --------------------------------------------------------------------------
def ic_series(preds, col):
    ev = preds.dropna(subset=["observed"])
    return ev.groupby(level="Date").apply(
        lambda g: g[col].corr(g["observed"], method="spearman")).rename(col)


def ic_stats(ic):
    ic = ic.dropna()
    n = len(ic)
    t = ic.mean() / ic.std(ddof=1) * np.sqrt(n) if n > 1 and ic.std(ddof=1) > 0 else np.nan
    return {"IC_mean": float(ic.mean()), "IC_pos_months": int((ic > 0).sum()),
            "IC_n_months": n, "IC_tstat": float(t)}


def fallback_stats(periods, method_col="method", reason_col="reason"):
    if periods is None or periods.empty:
        return {"n_fallback_months": np.nan, "n_cash_months": np.nan, "fallback_detail": ""}
    bad = periods[periods[method_col] != "optimized"]
    detail = "; ".join(f"{pd.Timestamp(r['Date']).date()} [{r[method_col]}] {r[reason_col]}"
                       for _, r in bad.iterrows())
    return {"n_fallback_months": int((periods[method_col] == "group_equal_weight").sum()),
            "n_cash_months": int((periods[method_col] == "cash").sum()),
            "fallback_detail": detail}


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test", default="2022-01-31", help="Première date de signal de test (fin de mois).")
    ap.add_argument("--block", type=int, default=6, help="Nombre de mois entre deux réentraînements.")
    args = ap.parse_args()
    first_test = pd.Timestamp(args.first_test)

    out = OUT_ROOT / ("walk_forward_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    print(f"Sorties : {out}", flush=True)
    tf_version = ss.enable_determinism()
    print(f"Déterminisme TensorFlow {tf_version} activé. Hyperparamètres : "
          f"SEQ_LEN={config.SEQ_LEN}, EPOCHS={config.LSTM_EPOCHS}, BATCH={config.BATCH_SIZE}, LR={config.LR}", flush=True)

    df_roles, universes, prices_daily, sessions, cutoff, *_ = ss.prepare()
    ds, _ = make_lstm_dataset(df_roles, universes, prices_daily, sessions)
    ds["Date"] = pd.to_datetime(ds["Date"])
    blocks = build_schedule(ds, first_test, args.block, cutoff)
    print(f"Dataset : signaux {ds['Date'].min().date()} → {ds['Date'].max().date()} ; "
          f"test {blocks[0][0].date()} → {blocks[-1][-1].date()} en {len(blocks)} blocs.", flush=True)

    # --- Walk-forward ---
    all_preds, schedule = [], []
    for b, block_dates in enumerate(blocks, start=1):
        T = block_dates[0]
        train, val, info = split_at(ds, T)
        test = ds[ds["Date"].isin(block_dates)].copy()
        assert test["Date"].min() >= T
        info.update(block=b, test_months=[str(block_dates[0].date()), str(block_dates[-1].date())],
                    n_test=len(test), epochs={})
        print(f"\n=== Bloc {b} : réentraînement à {T.date()} | train {info['train_months']} ({len(train)}) "
              f"| val {info['val_months']} ({len(val)}) | test {info['test_months']} ({len(test)}) ===", flush=True)
        block_pred = None
        for seed in SEEDS:
            print(f"--- Bloc {b}, graine {seed} ---", flush=True)
            p, hist = ss.train_and_predict(seed, train, val, test)
            info["epochs"][seed] = len(hist)
            if block_pred is None:
                block_pred = p[["entry_date", "exit_date", "observed"]].copy()
            elif not p.index.equals(block_pred.index):
                raise ValueError("Index de prévisions différent entre graines.")
            block_pred[f"pred_seed{seed}"] = p["pred_return_next"]
        all_preds.append(block_pred)
        schedule.append(info)

    preds = pd.concat(all_preds).sort_index()
    seed_cols = [f"pred_seed{s}" for s in SEEDS]
    preds["pred_ensemble"] = preds[seed_cols].mean(axis=1)
    preds["seed_dispersion"] = preds[seed_cols].std(axis=1, ddof=1)
    preds.to_csv(out / "predictions.csv")
    test_dates = pd.DatetimeIndex([d for blk in blocks for d in blk])

    # --- Backtests ---
    variants = {"LSTM_Ensemble_MaxSharpe": "pred_ensemble", "LSTM_Seed0_MaxSharpe": "pred_seed0"}
    daily, periods_by, allocs_by = {}, {}, {}
    for name, col in variants.items():
        pdf = preds[["entry_date", "exit_date", "observed"]].assign(pred_return_next=preds[col])
        strat, base_1n, allocations, periods = run_backtest_lstm(
            df_roles, pdf, universes, prices_daily, test_dates, sessions, cutoff, retry_missing=False)
        daily[name] = strat.rename(name)
        periods_by[name], allocs_by[name] = periods, allocations
        allocations.to_csv(out / f"allocations_{name}.csv", index=False)
        periods.to_csv(out / f"periods_{name}.csv", index=False)
        if name == "LSTM_Ensemble_MaxSharpe":
            daily["EqualWeight_1N"] = base_1n.rename("EqualWeight_1N")
    a, c = (allocs_by[k][["Date", "Ticker"]].reset_index(drop=True) for k in variants)
    if not a.equals(c):
        raise ValueError("Les actions éligibles diffèrent entre l'ensemble et la graine 0.")

    print("\n=== Références : minimum de variance, équipondéré 50/50, SPY ===", flush=True)
    ref_alloc = allocs_by["LSTM_Ensemble_MaxSharpe"]
    minvar, ew50, ref_log = ss.run_references(ref_alloc, universes, prices_daily, sessions)
    ref_log.to_csv(out / "references_by_month.csv", index=False)
    periods_ref = periods_by["LSTM_Ensemble_MaxSharpe"]
    spy = get_spy_returns(prices_daily, periods_ref["entry_date"].min(), periods_ref["exit_date"].max(),
                          sessions, retry_missing=False)
    daily.update({"MinVariance": minvar.rename("MinVariance"),
                  "EqualWeight_50_50": ew50.rename("EqualWeight_50_50"), "SPY": spy})
    returns = pd.concat(daily.values(), axis=1)
    for k, s in daily.items():
        if not s.index.equals(returns.index):
            raise ValueError(f"Dates désalignées pour {k}.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Rendements non finis.")
    returns.to_csv(out / "daily_returns.csv", index_label="Date")

    # --- IC ---
    ic = pd.concat([ic_series(preds, c) for c in ["pred_ensemble"] + seed_cols], axis=1)
    ic.to_csv(out / "ic_by_month.csv")
    ic_col = {"LSTM_Ensemble_MaxSharpe": "pred_ensemble", "LSTM_Seed0_MaxSharpe": "pred_seed0"}
    minvar_periods = ref_log.rename(columns={"minvar_method": "method", "minvar_reason": "reason"})

    # --- Blocs (pour chaque mois, jours de détention dans (entry, exit]) ---
    day_block = pd.Series(index=returns.index, dtype="float")
    month_block = {d: b for b, blk in enumerate(blocks, start=1) for d in blk}
    for _, r in periods_ref.iterrows():
        mask = (returns.index > r["entry_date"]) & (returns.index <= r["exit_date"])
        day_block[mask] = month_block[r["Date"]]
    if day_block.isna().any():
        raise ValueError("Des jours de détention ne sont rattachés à aucun bloc.")

    def row(name, rets, block=None):
        months = test_dates if block is None else blocks[block - 1]
        r = {"Portfolio": name, "block": "all" if block is None else block,
             "months": f"{months[0].date()}→{months[-1].date()}", "n_months": len(months),
             **perf_metrics(rets).to_dict()}
        if name in ic_col:
            r.update(ic_stats(ic[ic_col[name]].reindex(months)))
            per = periods_by[name]
            r.update(fallback_stats(per[per["Date"].isin(months)]))
        elif name == "MinVariance":
            r.update(fallback_stats(minvar_periods[minvar_periods["Date"].isin(months)]))
        return r

    names = list(variants) + ["MinVariance", "EqualWeight_50_50", "EqualWeight_1N", "SPY"]
    summary = pd.DataFrame([row(n, returns[n]) for n in names])
    by_block = pd.DataFrame([row(n, returns[n][day_block == b], b)
                             for b in range(1, len(blocks) + 1) for n in names])
    summary.to_csv(out / "summary.csv", index=False)
    by_block.to_csv(out / "summary_by_block.csv", index=False)

    disp = {"mean_seed_dispersion": float(preds["seed_dispersion"].mean()),
            "std_ensemble_predictions": float(preds["pred_ensemble"].std()),
            "mean_std_single_seed_predictions": float(preds[seed_cols].std().mean()),
            "ic_mean_by_seed": {c: float(ic[c].mean()) for c in seed_cols}}
    info = {"tensorflow": tf_version, "first_test": str(first_test.date()), "block_months": args.block,
            "seeds": SEEDS, "val_frac": VAL_FRAC, "min_val_months": MIN_VAL_MONTHS,
            "lstm_hyperparameters": {"SEQ_LEN": config.SEQ_LEN, "LSTM_EPOCHS": config.LSTM_EPOCHS,
                                     "BATCH_SIZE": config.BATCH_SIZE, "LR": config.LR},
            "max_weight": config.MAX_WEIGHT, "alloc_winners": ss.ALLOC_W, "alloc_stables": ss.ALLOC_S,
            "holding_start": str(returns.index.min().date()), "holding_end": str(returns.index.max().date()),
            "n_trading_days": len(returns), "n_test_months": len(test_dates),
            "schedule": schedule, "prediction_dispersion": disp, "retry_missing": False}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 120)
    cols = ["AnnReturn", "AnnVol", "Sharpe", "MaxDrawdown", "IC_mean", "IC_pos_months", "IC_n_months",
            "IC_tstat", "n_fallback_months", "n_cash_months"]
    print("\n=== Résumé (toute la période) ===")
    print(summary.set_index("Portfolio")[cols].round(4).to_string())
    print("\n=== Par bloc ===")
    print(by_block.set_index(["block", "Portfolio"])[cols].round(4).to_string())
    print("\n=== Replis de l'optimisation ===")
    for _, r in summary.iterrows():
        if isinstance(r.get("fallback_detail"), str) and r["fallback_detail"]:
            print(f"{r['Portfolio']} : {r['fallback_detail']}")
    print(f"\nDispersion entre graines : {disp}")
    print(f"\nRésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
