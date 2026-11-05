import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import SEQ_LEN, LSTM_EPOCHS, BATCH_SIZE, LR, RANDOM_SEED, ML_DROP
from backtest import month_period


def make_lstm_dataset(df_roles, universes, prices_daily, sessions, seq_len=SEQ_LEN):
    ml_features = [c for c in df_roles.columns if c not in ML_DROP]
    rows = []
    dates = df_roles.index.get_level_values("Date").unique().sort_values()
    histories = {t: g.droplevel("Ticker").sort_index()
                 for t, g in df_roles.groupby(level="Ticker")}
    for d in dates:
        sel = list(dict.fromkeys(universes[d]["winners"] + universes[d]["stables"]))
        entry, exit_date = month_period(d, sessions)
        grid = pd.date_range(end=d, periods=seq_len, freq="ME")
        for t in sel:
            g = histories[t]
            window = g.reindex(grid)[ml_features]
            if not np.isfinite(window.to_numpy()).all():
                continue
            y = np.nan
            if t in prices_daily and entry in prices_daily.index and exit_date in prices_daily.index:
                p0, p1 = prices_daily.at[entry, t], prices_daily.at[exit_date, t]
                if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0:
                    y = p1 / p0 - 1
            rows.append({"Date": d, "Ticker": t, "X": window.values.astype(np.float32),
                         "y": float(y), "entry_date": entry, "exit_date": exit_date})
    if not rows:
        raise ValueError("Dataset LSTM vide : aucune séquence de 12 mois consécutifs avec features complètes.")
    return pd.DataFrame(rows), ml_features


def split_dataset(ds):
    ds = ds.copy()
    ds["Date"] = pd.to_datetime(ds["Date"])
    dates_sorted = pd.DatetimeIndex(ds["Date"].sort_values().unique())
    cut1 = int(len(dates_sorted) * 0.70)
    cut2 = int(len(dates_sorted) * 0.85)
    if not 0 < cut1 < cut2 < len(dates_sorted):
        raise ValueError("Nombre de dates insuffisant pour le split chronologique 70/15/15.")
    train_dates, val_dates, test_dates = dates_sorted[:cut1], dates_sorted[cut1:cut2], dates_sorted[cut2:]
    train = ds[ds["Date"].isin(train_dates)].copy()
    val = ds[ds["Date"].isin(val_dates)].copy()
    test = ds[ds["Date"].isin(test_dates)].copy()
    n_train, n_val = len(train), len(val)
    train = train[(train["exit_date"] <= val_dates[0]) & np.isfinite(train["y"])].copy()
    val = val[(val["exit_date"] <= test_dates[0]) & np.isfinite(val["y"])].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Un ensemble est vide après contrôle des dates des cibles. "
                         "Les hyperparamètres n'ont pas été modifiés automatiquement.")
    info = {"train_dates": train_dates.strftime("%Y-%m-%d").tolist(),
            "validation_dates": val_dates.strftime("%Y-%m-%d").tolist(),
            "test_dates": test_dates.strftime("%Y-%m-%d").tolist(),
            "n_train": len(train), "n_validation": len(val), "n_test": len(test),
            "removed_train": n_train-len(train), "removed_validation": n_val-len(val),
            "last_train_target": str(train["exit_date"].max().date()),
            "last_validation_target": str(val["exit_date"].max().date())}
    return train, val, test, info


def build_lstm(n_features):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mse")
    return model


def train_lstm(train, val):
    from tensorflow.keras import callbacks
    X_train = np.stack(train["X"].values)
    y_train = train["y"].values.astype(np.float32)
    X_val = np.stack(val["X"].values)
    y_val = val["y"].values.astype(np.float32)
    n_features = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    X_train_s = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    model = build_lstm(n_features)
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    hist = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val),
                     epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=cb)
    history = pd.DataFrame(hist.history)
    history.index = pd.RangeIndex(1, len(history)+1, name="epoch")
    return model, scaler, history


def predict_test(model, scaler, test):
    X_test = np.stack(test["X"].values)
    n_features = X_test.shape[-1]
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    pred_df = test[["Date", "Ticker", "entry_date", "exit_date", "y"]].copy()
    pred_df = pred_df.rename(columns={"y": "observed"})
    pred_df["pred_return_next"] = model.predict(X_test_s, verbose=0).reshape(-1)
    if not np.isfinite(pred_df["pred_return_next"]).all():
        raise ValueError("Le LSTM a produit des prévisions non finies ; évaluation interrompue.")
    return pred_df.set_index(["Date", "Ticker"]).sort_index()


def evaluate_test(pred_df):
    evaluated = pred_df.dropna(subset=["observed", "pred_return_next"])
    if evaluated.empty:
        raise ValueError("Aucune cible Test observable pour l'évaluation prédictive.")
    error = evaluated["pred_return_next"] - evaluated["observed"]
    return {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error))),
            "directional_accuracy": float((np.sign(evaluated["pred_return_next"])
                                            == np.sign(evaluated["observed"])).mean()),
            "n_evaluated": len(evaluated), "n_unobserved": len(pred_df)-len(evaluated)}
