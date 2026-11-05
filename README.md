# Quantitative Portfolio Research

An end-to-end quantitative investment research project combining **KMeans clustering, LSTM-based return forecasting, factor signals and constrained portfolio construction**, evaluated with a **walk-forward** protocol.

The project explores whether Machine Learning and Deep Learning can support stock selection and portfolio allocation within the S&P 500 universe. An interactive **Streamlit dashboard** compares every strategy against fair benchmarks (equal weight 50/50, 1/N, minimum variance, SPY), period by period, and reports the information coefficient (IC) of each signal.

## Overview

1. **Data Collection & Feature Engineering:** historical S&P 500 prices (cached), technical indicators, returns and rolling Fama-French factor exposures.
2. **Stock Clustering & Selection:** monthly KMeans; two roles are kept, *Gagnants dynamiques* (winners) and *Stables de qualité* (quality stables).
3. **Signals:** an LSTM (raw or relative-return target, ensemble of seeds) or simple factor signals (12-1 momentum, low volatility, short-term reversal, low beta, quality, momentum + low volatility).
4. **Portfolio Construction:** equal weight, signal tilt, max Sharpe or minimum variance, with a 10 % cap per stock and a configurable split between the two roles (50/50, none, proportional, 30–70 %).
5. **Walk-forward Evaluation:** models are retrained every 6 months on all information available at that date (expanding window, purged validation, deterministic training). The 19 test months are split into a *selection* period (2022 signals) and a *confirmation* period (2023 signals).

## Key findings (see `experiments/`)

- The LSTM does not rank stocks better than chance; its IC is negative over 19 months, with either target.
- No signal or role-split rule chosen on 2022 is confirmed on 2023.
- The default configuration is therefore the most robust one: **equal weight, 50/50 between roles**. The original method (LSTM + max Sharpe) remains available.
- Full reports, written for beginners: `experiments/RAPPORT_FINAL.md`, `experiments/RAPPORT_EXPLORATION.md`, `experiments/RAPPORT_INTEGRATION.md`.

## Technologies

**Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · PyPortfolioOpt · Statsmodels · Streamlit · Plotly · yfinance**

## Installation & Usage

Requires Python 3.13.

```bash
git clone <repository-url>
cd Quantitative_Portfolio
pip install -r requirements.txt
```

Run the pipeline (walk-forward, configuration from `config.py`):

```bash
python pipeline.py
```

Run the original method (single 70/15/15 split, one LSTM, max Sharpe 50/50):

```bash
python pipeline.py --legacy
```

Launch the dashboard (French, read-only: it never trains a model nor downloads data):

```bash
streamlit run app.py
```

The dashboard has 7 pages (Vue d'ensemble, Simulateur, Comparaison, Par période, Portefeuille, Modèle et signal, Méthode et conclusions), sidebar filters (result folder, strategies, period, amount invested), KPI cards compared with SPY, a "?" help on every indicator and chart, CSV export of every table, and custom light/dark themes (menu ⋮ → Settings). Theme: `.streamlit/config.toml`; style: `assets/style.css`. Direct links to a page: `?page=apercu|simulateur|comparaison|periodes|portefeuille|signal|methode`. Older result folders are still displayed.

Run the automated tests:

```bash
python -m unittest discover -s tests -v
```

## Configuration (`config.py`)

| Option | Values | Default |
|---|---|---|
| `EVAL_MODE` | `"walk_forward"`, `"single_split"` | `"walk_forward"` |
| `LEGACY_MODE` | `True` forces the original method | `False` |
| `WF_FIRST_TEST`, `WF_BLOCK_MONTHS` | first test signal, months between retrainings | `"2022-01-31"`, `6` |
| `SELECTION_END` | end of the selection period | `"2022-12-31"` |
| `SEEDS` | seeds of the LSTM ensemble | `[0, 1, 2, 3, 4]` |
| `DETERMINISTIC` | reproducible TensorFlow training | `True` |
| `SIGNAL` | `lstm_raw`, `lstm_relative`, `momentum`, `low_vol`, `mom_lowvol`, `reversal_1m`, `low_beta`, `quality` | `"momentum"` (unused by equal weight) |
| `CONSTRUCTION` | `equal_weight`, `tilt`, `max_sharpe`, `min_variance` | `"equal_weight"` |
| `GROUP_RULE` | `50_50`, `none`, `proportional`, `30_70` | `"50_50"` |
| `TILT_STRENGTH` | tilt from (1 − k)× to (1 + k)× the equal weight | `0.5` |
| `IC_SIGNALS`, `COMPARISONS` | signals and strategies shown in the dashboard | see file |

LSTM hyperparameters (`SEQ_LEN`, `LSTM_EPOCHS`, `BATCH_SIZE`, `LR`) are unchanged.

## Limitations

Current S&P 500 constituents (survivorship bias), short test period (19 months), no transaction costs, zero risk-free rate in the Sharpe ratio.
