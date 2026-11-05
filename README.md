# Quantitative Portfolio Research

An end-to-end quantitative investment research project combining **KMeans clustering, LSTM-based return forecasting, and constrained Markowitz portfolio optimization**.

The project explores how Machine Learning and Deep Learning can support stock selection and portfolio allocation within the S&P 500 universe. An interactive **Streamlit dashboard** provides portfolio performance analysis, allocation insights, model diagnostics, and comparisons against an equal-weighted portfolio and the SPY benchmark.

## Overview

The investment pipeline follows five main steps:

1. **Data Collection & Feature Engineering:** Retrieve historical S&P 500 stock data and compute technical indicators, returns, and Fama-French factor exposures.
2. **Stock Clustering & Selection:** Apply KMeans to group stocks by their financial characteristics and select candidates based on momentum, risk, and quality indicators.
3. **Return Forecasting:** Train an LSTM neural network to predict future monthly stock returns using historical financial features.
4. **Portfolio Optimization:** Apply constrained Markowitz optimization to determine portfolio weights using predicted returns and historical risk estimates.
5. **Backtesting & Evaluation:** Evaluate the strategy out of sample and compare its performance with an equal-weighted portfolio and SPY.

## Technologies

**Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · PyPortfolioOpt · Statsmodels · Streamlit · Plotly · yfinance**

## Streamlit Dashboard

The interactive dashboard includes:

- **Portfolio Performance:** Cumulative returns, annualized performance, volatility, Sharpe Ratio, maximum drawdown, and benchmark comparisons.
- **Portfolio Allocation:** Monthly portfolio weights, selected stocks, and allocation breakdowns.
- **LSTM Diagnostics:** Training curves, forecasting errors, and observed vs. predicted returns.
- **Clustering & Stock Selection:** Cluster visualization, stock roles, and selection analysis.
- **Methodology:** Model parameters, investment approach, and key assumptions.

## Installation & Usage

Requires Python 3.13.

```bash
git clone <repository-url>
cd Quantitative_Portfolio
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python pipeline.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

The pipeline generates the results used by the dashboard.