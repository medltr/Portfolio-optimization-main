"""Tests automatiques : règles de portefeuille, optimiseurs et calendrier walk-forward.

Lancer depuis le dossier du projet :
    python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backtest import month_period, optimize_weights, optimize_with_predicted_mu, rule_weights  # noqa: E402
from lstm import add_relative_target, split_at, walk_forward_blocks  # noqa: E402
from strategy import GROUP_RULES, group_bounds, low_volatility  # noqa: E402

CAP = config.MAX_WEIGHT
TOL = 1e-6
# Tolérance numérique du solveur SCS pour les portefeuilles optimisés (max Sharpe, min variance).
# L'étape d'origine « clip des poids négatifs puis renormalisation » laisse des écarts de l'ordre de
# 1e-5 sur le plafond et 1e-4 sur la part d'un groupe (déjà présents dans le projet d'origine).
SOLVER_CAP_TOL = 2e-5      # = WEIGHT_TOL utilisé par la validation du projet
SOLVER_SHARE_TOL = 5e-4


def synthetic_prices(tickers, n_days=260, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    vols = np.linspace(0.008, 0.03, len(tickers))
    rets = rng.normal(0.0004, vols, size=(n_days, len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)


def check_rule(test, w, winners, stables, rule, cap_tol=TOL, share_tol=TOL):
    """Somme à 100 %, plafond de 10 %, poids positifs et règle de répartition respectée."""
    test.assertAlmostEqual(w.sum(), 1.0, delta=TOL)
    test.assertLessEqual(w.max(), CAP + cap_tol)
    test.assertGreaterEqual(w.min(), -TOL)
    bounds = group_bounds(len(winners), len(stables), rule)
    if bounds is not None:
        for names, (lo, hi) in zip([winners, stables], bounds):
            share = w.reindex(names).fillna(0).sum()
            test.assertGreaterEqual(share, lo - share_tol, f"{rule}: part {share:.6f} < {lo}")
            test.assertLessEqual(share, hi + share_tol, f"{rule}: part {share:.6f} > {hi}")


class TestRuleWeights(unittest.TestCase):
    """Équipondéré et inclinaison sous chaque règle de répartition."""

    def test_sum_cap_and_group_rule(self):
        rng = np.random.default_rng(1)
        for n_w, n_s in [(9, 32), (14, 32), (21, 22), (8, 26), (30, 10)]:
            winners = [f"W{i}" for i in range(n_w)]
            stables = [f"S{i}" for i in range(n_s)]
            signal = pd.Series(rng.normal(size=n_w + n_s), index=winners + stables)
            for rule in GROUP_RULES:
                for strength in [0.0, 0.5, 1.0]:
                    with self.subTest(n_w=n_w, n_s=n_s, rule=rule, strength=strength):
                        w = rule_weights(winners, stables, rule, signal=signal, strength=strength)
                        if w.attrs["method"] == "cash":
                            # seul cas admis : la règle est irréalisable avec le plafond
                            totals = group_bounds(n_w, n_s, rule)
                            self.assertTrue(totals is not None and (n_w * CAP < totals[0][0] or n_s * CAP < totals[1][0]))
                            continue
                        check_rule(self, w, winners, stables, rule)

    def test_infeasible_rule_goes_to_cash(self):
        w = rule_weights(["W1", "W2", "W3"], [f"S{i}" for i in range(20)], "50_50")
        self.assertTrue(w.empty)
        self.assertEqual(w.attrs["method"], "cash")

    def test_tilt_prefers_higher_signal(self):
        winners, stables = [f"W{i}" for i in range(10)], [f"S{i}" for i in range(10)]
        signal = pd.Series(np.arange(20, dtype=float), index=winners + stables)
        w = rule_weights(winners, stables, "50_50", signal=signal, strength=0.5)
        self.assertGreater(w["W9"], w["W0"])
        self.assertAlmostEqual(w["W9"] / w["W0"], 3.0, places=6)  # 1,5x contre 0,5x


class TestOptimizers(unittest.TestCase):
    """Max Sharpe et minimum de variance sous chaque règle."""

    def setUp(self):
        self.winners = [f"W{i}" for i in range(12)]
        self.stables = [f"S{i}" for i in range(28)]
        self.px = synthetic_prices(self.winners + self.stables)
        rng = np.random.default_rng(3)
        self.mu = pd.Series(rng.normal(0.01, 0.02, size=40), index=self.px.columns)

    def test_constraints(self):
        for construction in ["max_sharpe", "min_variance"]:
            for rule in GROUP_RULES:
                with self.subTest(construction=construction, rule=rule):
                    w = optimize_weights(self.px, self.mu, self.winners, self.stables, construction, rule)
                    self.assertIn(w.attrs["method"], ["optimized", "group_equal_weight"])
                    check_rule(self, w, self.winners, self.stables, rule, SOLVER_CAP_TOL, SOLVER_SHARE_TOL)

    def test_matches_legacy_optimizer(self):
        """max_sharpe + 50/50 reproduit exactement l'ancienne fonction du projet."""
        new = optimize_weights(self.px, self.mu, self.winners, self.stables, "max_sharpe", "50_50")
        old = optimize_with_predicted_mu(self.px, self.mu, winners=self.winners, stables=self.stables)
        self.assertEqual(new.attrs["method"], old.attrs["method"])
        pd.testing.assert_series_equal(new, old.reindex(new.index), check_names=False)


class TestWalkForwardCalendar(unittest.TestCase):
    """Absence de fuite d'information dans le calendrier et les cibles."""

    @classmethod
    def setUpClass(cls):
        sessions = pd.bdate_range("2019-01-01", "2024-03-29")
        dates = pd.date_range("2019-06-30", "2023-12-31", freq="ME")
        rng = np.random.default_rng(7)
        rows = []
        for d in dates:
            entry, exit_date = month_period(d, sessions)
            for t in range(30):
                rows.append({"Date": d, "Ticker": f"T{t}", "X": np.zeros((2, 2), dtype=np.float32),
                             "y": rng.normal(0.01, 0.05), "entry_date": entry, "exit_date": exit_date})
        cls.ds = add_relative_target(pd.DataFrame(rows))
        cls.cutoff = pd.Timestamp("2023-09-26")

    def test_blocks_consecutive_and_observable(self):
        blocks = walk_forward_blocks(self.ds, "2022-01-31", 6, self.cutoff)
        months = pd.DatetimeIndex([d for b in blocks for d in b])
        self.assertTrue(months.to_period("M").equals(pd.period_range(months[0], months[-1], freq="M")))
        last_exit = self.ds.groupby("Date")["exit_date"].max()
        self.assertTrue((last_exit.reindex(months) <= self.cutoff).all())
        self.assertTrue(all(len(b) <= 6 for b in blocks))

    def test_no_leakage_at_each_retraining(self):
        for block in walk_forward_blocks(self.ds, "2022-01-31", 6, self.cutoff):
            T = block[0]
            train, val, _ = split_at(self.ds, T, 0.2, 4)
            with self.subTest(T=T):
                self.assertLessEqual(val["exit_date"].max(), T)               # cibles de validation connues à T
                self.assertLessEqual(train["exit_date"].max(), val["Date"].min())  # purge avant la validation
                self.assertLess(train["Date"].max(), val["Date"].min())
                self.assertTrue((pd.DatetimeIndex(block) >= T).all())          # on ne prédit qu'à partir de T

    def test_relative_target_is_monthly_demeaned(self):
        means = self.ds.groupby("Date")["y_rel"].mean()
        self.assertLess(means.abs().max(), 1e-12)

    def test_low_volatility_uses_only_past(self):
        px = synthetic_prices(["A", "B", "C"], n_days=400)
        d = px.index[300]
        idx = pd.MultiIndex.from_product([[d], ["A", "B", "C"]], names=["Date", "Ticker"])
        before = low_volatility(px, idx)
        shocked = px.copy()
        shocked.iloc[301:] *= np.linspace(1, 5, len(shocked) - 301)[:, None]  # le futur change fortement
        after = low_volatility(shocked, idx)
        pd.testing.assert_series_equal(before, after)


class TestLatestResults(unittest.TestCase):
    """Contrôle des derniers résultats walk-forward enregistrés (ignoré s'il n'y en a pas)."""

    def setUp(self):
        runs = []
        for p in sorted(config.RESULTS_DIR.glob("*/parameters.json"), reverse=True):
            meta = json.loads(p.read_text(encoding="utf-8"))
            if meta.get("eval_mode") == "walk_forward" and meta.get("status") == "complete":
                runs.append((p.parent, meta))
        if not runs:
            self.skipTest("Aucun résultat walk-forward.")
        self.path, self.meta = runs[0]

    def test_all_strategies_respect_rules(self):
        alloc = pd.read_csv(self.path / "allocations_all.csv")
        methods = pd.read_csv(self.path / "periods_all.csv").set_index(["strategy", "Date"])["method"]
        rules = {s["key"]: s["rule"] for s in self.meta["strategies"] if s["key"] != "SPY"}
        for (key, d), g in alloc.groupby(["strategy", "Date"]):
            w = g.set_index("Ticker")["weight"]
            with self.subTest(strategy=key, date=d):
                if "CASH" in w.index:
                    self.assertAlmostEqual(w.sum(), 1.0, delta=TOL)
                    continue
                winners = g.loc[g.role == "Gagnants dynamiques", "Ticker"].tolist()
                stables = g.loc[g.role == "Stables de qualité", "Ticker"].tolist()
                solver = methods[(key, d)] == "optimized"
                check_rule(self, w, winners, stables, rules[key],
                           SOLVER_CAP_TOL if solver else TOL, SOLVER_SHARE_TOL if solver else TOL)

    def test_schedule_has_no_leakage(self):
        for b in self.meta["schedule"]:
            if "last_val_target" in b:
                with self.subTest(block=b["block"]):
                    T = pd.Timestamp(b["T"])
                    self.assertLessEqual(pd.Timestamp(b["last_val_target"]), T)
                    self.assertLessEqual(pd.Timestamp(b["last_train_target"]), pd.Timestamp(b["val_months"][0]))
                    self.assertGreaterEqual(pd.Timestamp(b["test_months"][0]), T)


if __name__ == "__main__":
    unittest.main()
