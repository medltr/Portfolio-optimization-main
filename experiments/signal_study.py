"""Étape 1 : qualité du signal (IC), sans construire de portefeuille.

Même calendrier walk-forward que walk_forward.py. Signaux comparés :
  a_lstm_brut  : LSTM, cible = rendement brut (modèle actuel), ensemble 5 graines
  b_lstm_rel   : LSTM, cible = rendement - moyenne du mois, ensemble 5 graines
  c_momentum   : momentum 12-1 mois (aucun entraînement)
  d_ridge_rel  : Ridge, cible relative, alpha choisi sur la validation

Usage : .venv/Scripts/python.exe Portfolio-optimization-main/experiments/signal_study.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_study as ss  # noqa: E402
import walk_forward as wf  # noqa: E402

import config  # noqa: E402
from lstm import make_lstm_dataset  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
FIRST_TEST, BLOCK = pd.Timestamp("2022-01-31"), 6
RIDGE_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
MAIN_SIGNALS = ["a_lstm_brut", "b_lstm_rel", "c_momentum", "d_ridge_rel"]
OUT_ROOT = Path(__file__).resolve().parent / "output"


def add_relative_target(ds):
    ds = ds.copy()
    # la moyenne du mois n'utilise que les cibles de ce mois, toutes connues à la même date de sortie
    if ds.groupby("Date")["exit_date"].nunique().max() != 1:
        raise ValueError("Dates de sortie différentes dans un même mois : cible relative ambiguë.")
    ds["y_rel"] = ds["y"] - ds.groupby("Date")["y"].transform("mean")
    return ds


def momentum_12_1(df_roles):
    r12, r1 = df_roles["return_12m"], df_roles["return_1m"]
    return ((1 + r12) ** 12 / (1 + r1) - 1).rename("c_momentum")


def flat(df):
    X = np.stack(df["X"].values)
    return X.reshape(len(X), -1)


def ridge_predict(train, val, test):
    Xtr, Xva, Xte = flat(train), flat(val), flat(test)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    ytr, yva = train["y_rel"].to_numpy(), val["y_rel"].to_numpy()
    scores = {}
    models = {}
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        scores[a] = float(np.mean((m.predict(Xva) - yva) ** 2))
        models[a] = m
    best = min(scores, key=scores.get)
    return models[best].predict(Xte), best, scores


def lstm_ensemble(train, val, test, target, tag, epochs_log):
    tr = train.assign(y=train[target])
    va = val.assign(y=val[target])
    out = {}
    for seed in SEEDS:
        print(f"--- {tag}, graine {seed} ---", flush=True)
        p, hist = ss.train_and_predict(seed, tr, va, test)  # test garde le rendement brut comme 'observed'
        epochs_log[f"{tag}_seed{seed}"] = len(hist)
        out[seed] = p
    idx = out[0].index
    for p in out.values():
        if not p.index.equals(idx):
            raise ValueError("Index de prévisions différent entre graines.")
    seeds = pd.DataFrame({f"{tag}_seed{s}": out[s]["pred_return_next"] for s in SEEDS})
    seeds[tag] = seeds.mean(axis=1)
    return out[0][["entry_date", "exit_date", "observed"]], seeds


def latest_walk_forward_predictions():
    runs = sorted(OUT_ROOT.glob("walk_forward_*/predictions.csv"))
    if not runs:
        return None, None
    p = pd.read_csv(runs[-1], parse_dates=["Date"]).set_index(["Date", "Ticker"]).sort_index()
    return runs[-1], p


def main():
    out = OUT_ROOT / ("signal_study_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    print(f"Sorties : {out}", flush=True)
    tf_version = ss.enable_determinism()

    df_roles, universes, prices_daily, sessions, cutoff, *_ = ss.prepare()
    ds, _ = make_lstm_dataset(df_roles, universes, prices_daily, sessions)
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds = add_relative_target(ds)
    blocks = wf.build_schedule(ds, FIRST_TEST, BLOCK, cutoff)
    mom = momentum_12_1(df_roles)

    parts, schedule, epochs_log = [], [], {}
    for b, block_dates in enumerate(blocks, start=1):
        T = block_dates[0]
        train, val, info = wf.split_at(ds, T)
        test = ds[ds["Date"].isin(block_dates)].copy()
        assert test["Date"].min() >= T
        print(f"\n=== Bloc {b} : T={T.date()} | train {info['train_months']} ({len(train)}) "
              f"| val {info['val_months']} ({len(val)}) | test {len(test)} ===", flush=True)
        base, a = lstm_ensemble(train, val, test, "y", "a_lstm_brut", epochs_log)
        _, bb = lstm_ensemble(train, val, test, "y_rel", "b_lstm_rel", epochs_log)
        ridge_pred, alpha, ridge_scores = ridge_predict(train, val, test)
        d = pd.Series(ridge_pred, index=pd.MultiIndex.from_frame(test[["Date", "Ticker"]]), name="d_ridge_rel")
        part = base.join(a).join(bb).join(d.sort_index())
        part["c_momentum"] = mom.reindex(part.index).to_numpy()
        if part[MAIN_SIGNALS].isna().any().any():
            raise ValueError(f"Signal manquant dans le bloc {b} : {part[MAIN_SIGNALS].isna().sum().to_dict()}")
        parts.append(part)
        info.update(block=b, ridge_alpha=alpha, ridge_val_mse=ridge_scores)
        schedule.append(info)

    sig = pd.concat(parts).sort_index()
    sig.to_csv(out / "signals.csv")

    # Contrôle de reproductibilité de (a) avec le walk-forward précédent
    wf_path, wf_pred = latest_walk_forward_predictions()
    repro = {"walk_forward_file": str(wf_path) if wf_path else None}
    if wf_pred is not None:
        common = sig.index.intersection(wf_pred.index)
        diff = np.abs(sig.loc[common, "a_lstm_brut"].to_numpy() - wf_pred.loc[common, "pred_ensemble"].to_numpy())
        repro.update(n_common=len(common), n_signal_rows=len(sig), max_abs_diff=float(diff.max()),
                     identical=bool(len(common) == len(sig) and diff.max() == 0.0))
    print(f"\nReproductibilité de (a) : {repro}", flush=True)

    # IC
    cols = MAIN_SIGNALS + [c for c in sig.columns if "_seed" in c]
    ic = pd.concat([wf.ic_series(sig, c) for c in cols], axis=1)
    ic.to_csv(out / "ic_by_month.csv")
    rows = []
    for c in cols:
        rows.append({"signal": c, "block": "all", **wf.ic_stats(ic[c])})
        for b, blk in enumerate(blocks, start=1):
            rows.append({"signal": c, "block": b, **wf.ic_stats(ic[c].reindex(blk))})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "signal_summary.csv", index=False)

    overall = summary[(summary["block"] == "all") & summary["signal"].isin(MAIN_SIGNALS)].set_index("signal")
    best = overall["IC_tstat"].idxmax()
    n_obs = sig.dropna(subset=["observed"]).groupby(level="Date").size()
    info = {"tensorflow": tf_version, "first_test": str(FIRST_TEST.date()), "block_months": BLOCK,
            "seeds": SEEDS, "lstm_hyperparameters": {"SEQ_LEN": config.SEQ_LEN, "LSTM_EPOCHS": config.LSTM_EPOCHS,
                                                     "BATCH_SIZE": config.BATCH_SIZE, "LR": config.LR},
            "ridge_alphas_grid": RIDGE_ALPHAS, "schedule": schedule, "epochs": epochs_log,
            "reproducibility_a": repro, "best_signal_by_tstat": best,
            "stocks_per_month": {"min": int(n_obs.min()), "max": int(n_obs.max()), "mean": float(n_obs.mean())},
            "signal_correlations_spearman": sig[MAIN_SIGNALS].corr(method="spearman").round(4).to_dict()}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n=== IC sur 19 mois ===")
    print(overall.round(4).to_string())
    print("\n=== IC moyen par bloc ===")
    print(summary[summary["signal"].isin(MAIN_SIGNALS)].pivot(index="signal", columns="block", values="IC_mean").round(4).to_string())
    print(f"\nAlpha Ridge retenus : {[s['ridge_alpha'] for s in schedule]}")
    print(f"Meilleur signal (t le plus élevé) : {best}")
    print(f"\nRésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
