"""Étape 2 : utilisation des signaux dans un portefeuille.

Lit la dernière sortie de signal_study.py et applique la règle fixée à l'avance :
on teste (b) LSTM à cible relative, et le signal dont la statistique t d'IC est la
plus élevée sur 19 mois (s'il est différent de (b)).

Pour chaque signal testé :
  (i)  max Sharpe comme dans le projet (backtest.run_backtest_lstm) ;
  (ii) inclinaison prudente autour de l'équipondéré 50/50 :
       w = (0,5 / n_groupe) * (1 + 0,5 * z), z = rang dans le groupe ramené à [-1, 1],
       plafond de 10 % avec redistribution dans le groupe.
Références : LSTM actuel + max Sharpe (rappel), min-variance, équipondéré 50/50, 1/N, SPY.

Usage : .venv/Scripts/python.exe Portfolio-optimization-main/experiments/portfolio_study.py [--signals-dir DIR]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_study as ss  # noqa: E402
import walk_forward as wf  # noqa: E402

import config  # noqa: E402
from backtest import (get_spy_returns, holding_prices, perf_metrics,  # noqa: E402
                      portfolio_log_returns, run_backtest_lstm)

MAIN_SIGNALS = ["a_lstm_brut", "b_lstm_rel", "c_momentum", "d_ridge_rel"]
TILT_K = 0.5
BLOCK = 6
OUT_ROOT = Path(__file__).resolve().parent / "output"


# --------------------------------------------------------------------------
def load_signals(signals_dir):
    if signals_dir is None:
        runs = sorted(OUT_ROOT.glob("signal_study_*/signal_summary.csv"))
        if not runs:
            raise FileNotFoundError("Aucune sortie de signal_study.py trouvée.")
        signals_dir = runs[-1].parent
    signals_dir = Path(signals_dir)
    sig = pd.read_csv(signals_dir / "signals.csv", parse_dates=["Date", "entry_date", "exit_date"])
    sig = sig.set_index(["Date", "Ticker"]).sort_index()
    summ = pd.read_csv(signals_dir / "signal_summary.csv", dtype={"block": str})
    overall = summ[(summ["block"] == "all") & summ["signal"].isin(MAIN_SIGNALS)].set_index("signal")
    best = overall["IC_tstat"].idxmax()
    return signals_dir, sig, overall, best


def mu_for(sig, name):
    s = sig[name]
    if name == "c_momentum":  # pas une prévision de rendement : on retire la moyenne du mois (règle fixée)
        s = s - s.groupby(level="Date").transform("mean")
    return s


def cap_group(w, cap):
    w = w.copy()
    capped = False
    for _ in range(100):
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


def tilt_weights(signal, winners, stables, max_weight=config.MAX_WEIGHT):
    if not (len(winners) * max_weight >= ss.ALLOC_W - 1e-12 and len(stables) * max_weight >= ss.ALLOC_S - 1e-12):
        w = pd.Series(dtype=float)
        w.attrs.update(method="cash", reason="Contraintes 50/50 et poids maximal irréalisables")
        return w
    parts, any_cap = [], False
    for group, alloc in [(winners, ss.ALLOC_W), (stables, ss.ALLOC_S)]:
        s = signal.reindex(group)
        if s.isna().any():
            raise ValueError("Signal manquant pour une action de l'univers.")
        n = len(group)
        z = 2 * (s.rank(method="average") - 1) / (n - 1) - 1 if n > 1 else s * 0.0
        w = alloc / n * (1 + TILT_K * z)
        w, capped = cap_group(w, max_weight)
        any_cap |= capped
        parts.append(w)
    w = pd.concat(parts)
    assert abs(w.sum() - 1) < 1e-9 and w.max() <= max_weight + 1e-9 and w.min() >= 0
    w.attrs.update(method="tilt_capped" if any_cap else "tilt", reason="")
    return w


def run_tilt(name, signal, allocations, universes, prices_daily, sessions):
    daily, log = [], []
    alloc = allocations[allocations["Ticker"] != "CASH"]
    for d, g in alloc.groupby("Date", sort=True):
        tickers = g["Ticker"].tolist()
        winners = [t for t in universes[d]["winners"] if t in tickers]
        stables = [t for t in universes[d]["stables"] if t in tickers]
        w = tilt_weights(signal.xs(d, level="Date"), winners, stables)
        hold_dates = sessions[(sessions >= g["entry_date"].iloc[0]) & (sessions <= g["exit_date"].iloc[0])]
        px_hold = holding_prices(prices_daily, tickers, hold_dates, retry_missing=False)
        daily.append(portfolio_log_returns(px_hold, w).rename(name))
        log.append({"Date": d, "method": w.attrs["method"], "reason": w.attrs["reason"],
                    "max_weight": float(w.max()) if not w.empty else 0.0,
                    "min_weight": float(w.min()) if not w.empty else 0.0})
    return pd.concat(daily).sort_index(), pd.DataFrame(log)


def method_stats(log):
    bad = log[~log["method"].isin(["optimized", "tilt", "tilt_capped", "equal_50_50"])]
    return {"n_fallback_months": int((log["method"] == "group_equal_weight").sum()),
            "n_cash_months": int((log["method"] == "cash").sum()),
            "n_capped_months": int((log["method"] == "tilt_capped").sum()),
            "fallback_detail": "; ".join(f"{pd.Timestamp(r['Date']).date()} [{r['method']}] {r['reason']}"
                                         for _, r in bad.iterrows())}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", default=None)
    args = ap.parse_args()
    signals_dir, sig, overall, best = load_signals(args.signals_dir)
    to_test = ["b_lstm_rel"] + ([best] if best != "b_lstm_rel" else [])
    print(f"Signaux lus dans {signals_dir}\nt de l'IC : {overall['IC_tstat'].round(3).to_dict()}\n"
          f"Meilleur signal (règle fixée) : {best} ; signaux testés : {to_test}", flush=True)

    out = OUT_ROOT / ("portfolio_study_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    df_roles, universes, prices_daily, sessions, cutoff, *_ = ss.prepare()

    test_dates = pd.DatetimeIndex(sorted(sig.index.get_level_values("Date").unique()))
    if not test_dates.to_period("M").equals(pd.period_range(test_dates[0], test_dates[-1], freq="M")):
        raise ValueError("Mois de test non consécutifs.")
    blocks = [test_dates[i:i + BLOCK] for i in range(0, len(test_dates), BLOCK)]
    print(f"Test : {test_dates[0].date()} → {test_dates[-1].date()} ({len(test_dates)} mois, {len(blocks)} blocs)", flush=True)

    daily, logs, ic_cols = {}, {}, {}
    ref_alloc = ref_periods = None
    ms_signals = list(dict.fromkeys(to_test + ["a_lstm_brut"]))
    for s in ms_signals:
        pdf = sig[["entry_date", "exit_date", "observed"]].assign(pred_return_next=mu_for(sig, s))
        strat, base_1n, allocations, periods = run_backtest_lstm(
            df_roles, pdf, universes, prices_daily, test_dates, sessions, cutoff, retry_missing=False)
        name = f"{s}_MaxSharpe"
        daily[name] = strat.rename(name)
        logs[name] = periods[["Date", "method", "reason"]]
        ic_cols[name] = s
        allocations.to_csv(out / f"allocations_{name}.csv", index=False)
        if ref_alloc is None:
            ref_alloc, ref_periods = allocations, periods
            daily["EqualWeight_1N"] = base_1n.rename("EqualWeight_1N")
        elif not allocations[["Date", "Ticker"]].reset_index(drop=True).equals(
                ref_alloc[["Date", "Ticker"]].reset_index(drop=True)):
            raise ValueError(f"Univers différent pour {s}.")
        print(f"{name} : {logs[name]['method'].value_counts().to_dict()}", flush=True)

    for s in to_test:
        name = f"{s}_Tilt"
        r, log = run_tilt(name, sig[s], ref_alloc, universes, prices_daily, sessions)
        daily[name], logs[name], ic_cols[name] = r, log, s
        log.to_csv(out / f"weights_log_{name}.csv", index=False)
        print(f"{name} : {log['method'].value_counts().to_dict()}", flush=True)

    minvar, ew50, ref_log = ss.run_references(ref_alloc, universes, prices_daily, sessions)
    logs["MinVariance"] = ref_log.rename(columns={"minvar_method": "method", "minvar_reason": "reason"})[["Date", "method", "reason"]]
    spy = get_spy_returns(prices_daily, ref_periods["entry_date"].min(), ref_periods["exit_date"].max(),
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

    # jours -> blocs
    month_block = {d: b for b, blk in enumerate(blocks, start=1) for d in blk}
    day_block = pd.Series(np.nan, index=returns.index)
    for _, r in ref_periods.iterrows():
        day_block[(returns.index > r["entry_date"]) & (returns.index <= r["exit_date"])] = month_block[r["Date"]]
    if day_block.isna().any():
        raise ValueError("Jours de détention sans bloc.")

    ic = {s: wf.ic_series(sig, s) for s in set(ic_cols.values())}

    def row(name, block=None):
        months = test_dates if block is None else blocks[block - 1]
        rets = returns[name] if block is None else returns[name][day_block == block]
        r = {"Portfolio": name, "block": "all" if block is None else block,
             "months": f"{months[0].date()}→{months[-1].date()}", "n_months": len(months),
             **perf_metrics(rets).to_dict()}
        if name in ic_cols:
            r.update({"signal": ic_cols[name], **wf.ic_stats(ic[ic_cols[name]].reindex(months))})
        if name in logs:
            lg = logs[name]
            r.update(method_stats(lg[lg["Date"].isin(months)]))
        return r

    names = list(daily)
    order = [n for n in names if n.endswith(("_MaxSharpe", "_Tilt"))] + \
            ["MinVariance", "EqualWeight_50_50", "EqualWeight_1N", "SPY"]
    summary = pd.DataFrame([row(n) for n in order])
    by_block = pd.DataFrame([row(n, b) for b in range(1, len(blocks) + 1) for n in order])
    summary.to_csv(out / "summary.csv", index=False)
    by_block.to_csv(out / "summary_by_block.csv", index=False)
    info = {"signals_dir": str(signals_dir), "best_signal_by_tstat": best, "tested_signals": to_test,
            "ic_tstat": overall["IC_tstat"].to_dict(), "tilt_k": TILT_K, "max_weight": config.MAX_WEIGHT,
            "holding_start": str(returns.index.min().date()), "holding_end": str(returns.index.max().date()),
            "n_trading_days": len(returns), "retry_missing": False}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["AnnReturn", "AnnVol", "Sharpe", "MaxDrawdown", "IC_mean", "n_fallback_months", "n_cash_months", "n_capped_months"]
    print("\n=== Résumé (19 mois) ===")
    print(summary.set_index("Portfolio")[cols].round(4).to_string())
    print("\n=== Sharpe par bloc ===")
    print(by_block.pivot(index="Portfolio", columns="block", values="Sharpe").reindex(order).round(3).to_string())
    for _, r in summary.iterrows():
        if isinstance(r.get("fallback_detail"), str) and r["fallback_detail"]:
            print(f"Replis {r['Portfolio']} : {r['fallback_detail']}")
    print(f"\nRésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
