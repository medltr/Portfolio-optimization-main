"""Phase A : exploration avec période de sélection (signaux 2022) et de confirmation (signaux 2023).

A1 : IC de nouveaux signaux (retournement 1 mois, faible volatilité, faible bêta, qualité,
     momentum + faible volatilité) comparés au momentum 12-1 et au LSTM relatif.
A2 : règles de répartition entre rôles (50/50, aucune, proportionnelle, 30-70 %) avec
     l'équipondéré et l'inclinaison vers le meilleur signal choisi sur la sélection.

Les règles de choix sont fixées dans JOURNAL.md avant le calcul. Rien n'est réajusté sur
la confirmation. Aucun nouvel entraînement : les prévisions LSTM relatives sont relues
depuis la dernière sortie de signal_study.py (déterministe, vérifiée).

Usage : .venv/Scripts/python.exe Portfolio-optimization-main/experiments/exploration.py
"""
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
from backtest import get_spy_returns, holding_prices, perf_metrics, portfolio_log_returns, run_backtest_lstm  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "output"
SIGNALS = ["momentum", "lstm_relative", "reversal_1m", "low_vol", "low_beta", "quality", "mom_lowvol"]
RULES = ["50_50", "none", "proportional", "30_70"]
TILT_K = 0.5
SEL_YEAR, CONF_YEAR = 2022, 2023


# --------------------------------------------------------------------------
# Règles de répartition et poids
# --------------------------------------------------------------------------
def group_totals(n_w, n_s, rule):
    """Part de chaque groupe (Gagnants, Stables), ou None si aucune contrainte de groupe."""
    n = n_w + n_s
    if rule == "50_50":
        return 0.5, 0.5
    if rule == "proportional":
        return n_w / n, n_s / n
    if rule == "30_70":
        tw = min(max(n_w / n, 0.3), 0.7)
        return tw, 1 - tw
    if rule == "none":
        return None
    raise ValueError(f"Règle inconnue : {rule}")


def cap_redistribute(w, cap):
    w = w.copy()
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
            raise ValueError("Plafond irréalisable.")
        w[free] += excess * w[free] / w[free].sum()
    return w, capped


def tilt_group(signal, tickers, total, k, cap):
    n = len(tickers)
    if signal is None or k == 0 or n == 1:
        w = pd.Series(total / n, index=tickers)
    else:
        s = signal.reindex(tickers)
        if s.isna().any():
            raise ValueError("Signal manquant pour une action de l'univers.")
        z = 2 * (s.rank(method="average") - 1) / (n - 1) - 1
        w = total / n * (1 + k * z)
    return cap_redistribute(w, cap)


def rule_weights(winners, stables, rule, signal=None, k=TILT_K, cap=config.MAX_WEIGHT):
    """Équipondéré (signal=None ou k=0) ou inclinaison, sous une règle de répartition."""
    winners, stables = list(winners), list(stables)
    totals = group_totals(len(winners), len(stables), rule)
    groups = [(winners + stables, 1.0)] if totals is None else [(winners, totals[0]), (stables, totals[1])]
    feasible = all((len(g) * cap >= t - 1e-12) and (t <= 1e-12 or len(g) > 0) for g, t in groups)
    if not feasible:
        w = pd.Series(dtype=float)
        w.attrs.update(method="cash", reason="Règle de répartition et plafond irréalisables")
        return w
    parts, any_cap = [], False
    for g, t in groups:
        if t <= 1e-12 or not g:
            continue
        w, capped = tilt_group(signal, g, t, k, cap)
        parts.append(w)
        any_cap |= capped
    w = pd.concat(parts)
    assert abs(w.sum() - 1) < 1e-9 and w.max() <= cap + 1e-9 and w.min() >= 0
    w.attrs.update(method="capped" if any_cap else "ok", reason="")
    return w


# --------------------------------------------------------------------------
# Signaux
# --------------------------------------------------------------------------
def load_signal_study():
    runs = sorted(OUT_ROOT.glob("signal_study_*/signals.csv"))
    if not runs:
        raise FileNotFoundError("Sortie de signal_study.py introuvable.")
    sig = pd.read_csv(runs[-1], parse_dates=["Date", "entry_date", "exit_date"]).set_index(["Date", "Ticker"]).sort_index()
    return runs[-1].parent, sig


def low_vol_signal(prices_daily, index):
    lr = np.log(prices_daily.where(prices_daily > 0)).diff()
    vol = lr.rolling(252, min_periods=200).std()
    out = []
    for d, t in index:
        pos = vol.index.searchsorted(d, side="right") - 1  # dernière séance <= date du signal
        out.append(-vol.iloc[pos][t] if pos >= 0 and t in vol.columns else np.nan)
    return pd.Series(out, index=index, name="low_vol")


def build_signals(sig, df_roles, prices_daily):
    idx = sig.index
    roles = df_roles.reindex(idx)
    S = pd.DataFrame(index=idx)
    S["momentum"] = sig["c_momentum"]
    S["lstm_relative"] = sig["b_lstm_rel"]
    S["reversal_1m"] = -roles["return_1m"]
    S["low_vol"] = low_vol_signal(prices_daily, idx)
    S["low_beta"] = -roles["Mkt-RF"]
    S["quality"] = roles["RMW"]
    ranks = S[["momentum", "low_vol"]].groupby(level="Date").rank(pct=True)
    S["mom_lowvol"] = ranks.mean(axis=1)
    if S.isna().any().any():
        raise ValueError(f"Signaux manquants : {S.isna().sum()[S.isna().sum() > 0].to_dict()}")
    return S


# --------------------------------------------------------------------------
def main():
    out = OUT_ROOT / ("exploration_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    print(f"Sorties : {out}", flush=True)
    sig_dir, sig = load_signal_study()
    df_roles, universes, prices_daily, sessions, cutoff, *_ = ss.prepare()
    S = build_signals(sig, df_roles, prices_daily)
    S["observed"] = sig["observed"]
    S.to_csv(out / "signals.csv")

    dates = pd.DatetimeIndex(sorted(S.index.get_level_values("Date").unique()))
    periods_def = {"selection": dates[dates.year == SEL_YEAR], "confirmation": dates[dates.year == CONF_YEAR], "total": dates}
    if len(periods_def["selection"]) != 12 or len(periods_def["confirmation"]) != 7:
        raise ValueError("Découpage sélection/confirmation inattendu.")

    # ---------------- A1 : IC ----------------
    ic = pd.concat([wf.ic_series(S, c) for c in SIGNALS], axis=1)
    ic.to_csv(out / "ic_by_month.csv")
    rows = [{"signal": c, "period": p, **wf.ic_stats(ic[c].reindex(m))} for c in SIGNALS for p, m in periods_def.items()]
    ic_summary = pd.DataFrame(rows)
    ic_summary.to_csv(out / "ic_summary.csv", index=False)
    sel_t = ic_summary[ic_summary.period == "selection"].set_index("signal")["IC_tstat"]
    best = sel_t.idxmax()
    best_conf_ic = float(ic_summary[(ic_summary.signal == best) & (ic_summary.period == "confirmation")]["IC_mean"].iloc[0])
    best_confirmed = best_conf_ic > 0
    print(f"\nMeilleur signal sur la sélection : {best} (t={sel_t[best]:.2f}) ; IC confirmation {best_conf_ic:+.4f} "
          f"→ {'confirmé' if best_confirmed else 'NON confirmé'}", flush=True)

    # ---------------- A2 : règles de répartition ----------------
    # même univers que les études précédentes (actions retenues par le backtest du projet)
    mu = S["momentum"] - S["momentum"].groupby(level="Date").transform("mean")
    pdf = sig[["entry_date", "exit_date", "observed"]].assign(pred_return_next=mu)
    _, _, allocations, periods = run_backtest_lstm(df_roles, pdf, universes, prices_daily, dates, sessions, cutoff, retry_missing=False)
    alloc = allocations[allocations["Ticker"] != "CASH"]

    daily, logs = {}, []
    for construction in ["EW", "Tilt"]:
        for rule in RULES:
            name = f"{construction}_{rule}"
            parts = []
            for d, g in alloc.groupby("Date", sort=True):
                tickers = g["Ticker"].tolist()
                winners = [t for t in universes[d]["winners"] if t in tickers]
                stables = [t for t in universes[d]["stables"] if t in tickers]
                signal = S[best].xs(d, level="Date") if construction == "Tilt" else None
                w = rule_weights(winners, stables, rule, signal=signal)
                hold = sessions[(sessions >= g["entry_date"].iloc[0]) & (sessions <= g["exit_date"].iloc[0])]
                px = holding_prices(prices_daily, tickers, hold, retry_missing=False)
                parts.append(portfolio_log_returns(px, w).rename(name))
                logs.append({"portfolio": name, "Date": d, "method": w.attrs["method"], "n_winners": len(winners),
                             "n_stables": len(stables), "w_winners": float(w.reindex(winners).sum()) if not w.empty else 0.0,
                             "max_weight": float(w.max()) if not w.empty else 0.0})
            daily[name] = pd.concat(parts).sort_index()

    minvar, ew50_ref, _ = ss.run_references(allocations, universes, prices_daily, sessions)
    if (daily["EW_50_50"] - ew50_ref).abs().max() > 1e-12:
        raise ValueError("L'équipondéré 50/50 recalculé diffère de la référence des études précédentes.")
    spy = get_spy_returns(prices_daily, periods["entry_date"].min(), periods["exit_date"].max(), sessions, retry_missing=False)
    daily.update({"MinVariance_50_50": minvar.rename("MinVariance_50_50"), "SPY": spy})
    returns = pd.concat(daily.values(), axis=1)
    returns.columns = list(daily)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Rendements non finis.")
    returns.to_csv(out / "daily_returns.csv", index_label="Date")
    pd.DataFrame(logs).to_csv(out / "weights_log.csv", index=False)

    # jours -> période (sélection / confirmation) via les dates d'entrée et de sortie
    day_period = pd.Series(index=returns.index, dtype=object)
    for _, r in periods.iterrows():
        day_period[(returns.index > r["entry_date"]) & (returns.index <= r["exit_date"])] = (
            "selection" if r["Date"].year == SEL_YEAR else "confirmation")
    if day_period.isna().any():
        raise ValueError("Jours de détention sans période.")
    rows = []
    for name in returns:
        for p in ["selection", "confirmation", "total"]:
            r = returns[name] if p == "total" else returns[name][day_period == p]
            rows.append({"portfolio": name, "period": p, "n_days": len(r), **perf_metrics(r).to_dict()})
    a2 = pd.DataFrame(rows)
    a2.to_csv(out / "a2_summary.csv", index=False)

    # ---------------- Décision (règle fixée à l'avance) ----------------
    candidates = [f"{c}_{r}" for c in ["EW", "Tilt"] for r in RULES]
    sh = a2.pivot(index="portfolio", columns="period", values="Sharpe")
    c_star = sh.loc[candidates, "selection"].idxmax()
    beats_ref = bool(sh.loc[c_star, "confirmation"] > sh.loc["EW_50_50", "confirmation"])
    uses_signal = c_star.startswith("Tilt")
    confirmed = beats_ref and (best_confirmed if uses_signal else True) and c_star != "EW_50_50"
    construction, rule = c_star.split("_", 1)
    default = ({"construction": "tilt" if uses_signal else "equal_weight", "group_rule": rule,
                "signal": best if uses_signal else None} if confirmed
               else {"construction": "equal_weight", "group_rule": "50_50", "signal": None})
    decision = {"signals_source": str(sig_dir), "best_signal_selection": best, "best_signal_tstat_selection": float(sel_t[best]),
                "best_signal_ic_confirmation": best_conf_ic, "best_signal_confirmed": bool(best_confirmed),
                "c_star": c_star, "c_star_sharpe_selection": float(sh.loc[c_star, "selection"]),
                "c_star_sharpe_confirmation": float(sh.loc[c_star, "confirmation"]),
                "ew50_sharpe_confirmation": float(sh.loc["EW_50_50", "confirmation"]),
                "c_star_beats_ew50_on_confirmation": beats_ref, "c_star_confirmed": bool(confirmed),
                "default_for_phase_B": default}
    (out / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n=== A1 : IC (moyenne / mois > 0 / t) ===")
    print(ic_summary.pivot(index="signal", columns="period", values="IC_mean").reindex(SIGNALS)
          [["selection", "confirmation", "total"]].round(4).to_string())
    print(ic_summary.pivot(index="signal", columns="period", values="IC_tstat").reindex(SIGNALS)
          [["selection", "confirmation", "total"]].round(2).to_string())
    print(ic_summary.pivot(index="signal", columns="period", values="IC_pos_months").reindex(SIGNALS)
          [["selection", "confirmation", "total"]].to_string())
    print("\n=== A2 : Sharpe ===")
    print(sh[["selection", "confirmation", "total"]].round(3).to_string())
    lg = pd.DataFrame(logs)
    print("\nMois plafonnés / en liquidités :", lg.groupby("portfolio")["method"].value_counts().unstack(fill_value=0).to_dict())
    print(f"\nDécision : {json.dumps(decision, indent=1, ensure_ascii=False)}")
    print(f"\nRésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
