import os
from datetime import datetime, timezone
from io import BytesIO, StringIO
from zipfile import ZipFile
import yfinance as yf

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from config import (BETA_WINDOW, CACHE_DIR, DATA_DIR, FACTOR_LAG_MONTHS,
                    FACTORS, FF_MONTHLY_PATH, FF_START, FF_URL, WIKI_URL,
                    start_date, end_date)


def trading_sessions(start, end):
    import pandas_market_calendars as mcal
    return mcal.get_calendar("NYSE").valid_days(start, end).tz_localize(None)


def get_sp500_tickers(refresh=False):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "sp500.csv"
    if path.exists() and not refresh:
        sp500 = pd.read_csv(path)
    else:
        response = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
        response.raise_for_status()
        sp500 = pd.read_html(StringIO(response.text))[0]
        sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-", regex=False)
        sp500.to_csv(path, index=False)
    return sp500["Symbol"].unique().tolist()


def download_prices(symbols, refresh=False):
    CACHE_DIR.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR / "yfinance"))
    path = CACHE_DIR / f"daily_{start_date}_{end_date}.csv.gz"
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["Date"], index_col=["Date", "Ticker"])
        return df.sort_index()
    tickers = list(dict.fromkeys(symbols + ["SPY"]))
    raw = yf.download(tickers, start=start_date, end=end_date,
                      auto_adjust=False, progress=False, threads=8)
    if raw.empty:
        raise ValueError("Aucun prix téléchargé. Vérifiez la connexion et Yahoo Finance.")
    raw.columns.names = ["Price", "Ticker"]
    df = raw.stack(level="Ticker", future_stack=True).dropna(how="all")
    df.index.names = ["Date", "Ticker"]
    df.columns.name = None
    df = df.sort_index()
    if "Adj Close" not in df:
        raise ValueError("Les cours ajustés sont absents du téléchargement.")
    df.to_csv(path, compression="gzip")
    return df


def compute_atr(stock_data):
    import pandas_ta as ta
    if len(stock_data) < 14:
        return pd.Series(np.nan, index=stock_data.index)
    return ta.atr(high=stock_data["High"], low=stock_data["Low"],
                  close=stock_data["Close"], length=14)


def add_daily_features(df):
    (CACHE_DIR / "numba").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
    import pandas_ta as ta
    df = df.dropna(subset=["High", "Low", "Close"]).copy()
    df["GK"] = np.sqrt(np.maximum(
        0, 0.5 * np.log(df["High"] / df["Low"])**2
        - (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"])**2))
    df["dollar_volume"] = df["Volume"] * df["Adj Close"] / 1e6
    df["rsi"] = df.groupby("Ticker")["Adj Close"].transform(
        lambda x: ta.rsi(close=x, length=20))
    df["atr"] = df.groupby(level="Ticker", group_keys=False).apply(compute_atr)
    return df


def resample_monthly(df, sessions, cutoff):
    df1 = df.sort_index().reset_index().set_index("Date")
    monthly = df1.groupby("Ticker").resample("ME").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
        "Adj Close": "last", "Volume": "sum", "dollar_volume": "sum",
        "GK": "mean", "rsi": "last", "atr": "last"})
    monthly = monthly.reset_index().set_index(["Date", "Ticker"]).sort_index()
    last_sessions = pd.Series(sessions, index=sessions).resample("ME").last()
    dates = monthly.index.get_level_values("Date")
    actual_dates = dates.map(last_sessions)
    daily_index = pd.MultiIndex.from_arrays(
        [actual_dates, monthly.index.get_level_values("Ticker")], names=["Date", "Ticker"])
    for col in ["Adj Close", "Close", "rsi", "atr"]:
        monthly[col] = df[col].reindex(daily_index).to_numpy()
    complete = actual_dates <= pd.Timestamp(cutoff)
    monthly = monthly[complete].copy()
    monthly["atr_pct"] = monthly["atr"] / monthly["Adj Close"]
    return monthly


def calculate_returns(g):
    for i in [1, 2, 3, 6, 9, 12]:
        g[f"return_{i}m"] = g["Adj Close"].pct_change(i, fill_method=None).add(1).pow(1/i).sub(1)
    return g


def filter_liquidity(monthly):
    df1 = monthly.groupby(level="Ticker", group_keys=False).apply(calculate_returns)
    df1["dollar_volume"] = (df1["dollar_volume"].unstack("Ticker")
                            .rolling(5*12, min_periods=12).mean().stack())
    df1["classement"] = df1.groupby("Date")["dollar_volume"].rank(ascending=False)
    return df1[df1["classement"] < 150].drop(columns=["dollar_volume", "classement"]).dropna()


def load_fama_french():
    if not FF_MONTHLY_PATH.exists():
        response = requests.get(FF_URL, timeout=60)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            text = archive.read(archive.namelist()[0]).decode("utf-8-sig")
        rows = []
        for line in text.splitlines():
            parts = line.split(",")
            if len(parts) == 7 and parts[0].strip().isdigit() and len(parts[0].strip()) == 6:
                rows.append([parts[0].strip()] + [float(v) / 100 for v in parts[1:]])
        if not rows:
            raise ValueError("Le fichier officiel ne contient pas les six facteurs mensuels attendus.")
        ff = pd.DataFrame(rows, columns=["Date"] + FACTORS + ["RF"])
        ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m") + pd.offsets.MonthEnd(0)
        DATA_DIR.mkdir(exist_ok=True)
        with FF_MONTHLY_PATH.open("w", encoding="utf-8", newline="") as f:
            f.write(f"# Source: {FF_URL}\n")
            f.write(f"# Retrieved UTC: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("# Monthly official factors; decimal returns; revised historical vintage.\n")
            ff.to_csv(f, index=False)
    ff = pd.read_csv(FF_MONTHLY_PATH, comment="#", parse_dates=["Date"]).set_index("Date")
    if ff.index.has_duplicates or ff[FACTORS + ["RF"]].isna().any().any():
        raise ValueError("Dates dupliquées ou facteurs Fama-French manquants.")
    return ff.sort_index().loc[FF_START:]


def add_rolling_betas(df1, fama_french):
    out = []
    for ticker, g in df1.groupby(level="Ticker"):
        g = g.droplevel("Ticker").sort_index()
        first = max(g.index.min(), pd.Timestamp(FF_START) + pd.offsets.MonthEnd(0))
        grid = pd.date_range(first, g.index.max(), freq="ME", name="Date")
        if len(grid) < BETA_WINDOW:
            continue
        ff = fama_french.reindex(grid)
        y = g["return_1m"].reindex(grid) - ff["RF"]
        fit = RollingOLS(y, sm.add_constant(ff[FACTORS]), window=BETA_WINDOW,
                         min_nobs=len(FACTORS) + 2, missing="drop").fit(params_only=True)
        betas = fit.params.drop(columns="const").shift(FACTOR_LAG_MONTHS)
        betas["Ticker"] = ticker
        out.append(betas.reset_index().set_index(["Date", "Ticker"]))
    if not out:
        raise ValueError("Historique insuffisant pour les betas glissants sur 24 mois.")
    return df1.join(pd.concat(out)).replace([np.inf, -np.inf], np.nan).dropna()
