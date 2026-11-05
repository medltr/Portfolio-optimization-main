import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtest import perf_metrics
from config import RESULTS_DIR

NAMES = {"Strategy_LSTM_Opt": "LSTM + Markowitz",
         "Baseline_EqualWeight": "Équipondération", "SPY": "SPY"}
COLORS = {"LSTM + Markowitz": "#137e83", "Équipondération": "#8291ae", "SPY": "#d69a3f"}
ROLE_COLORS = {"Gagnants dynamiques": "#137e83", "Stables de qualité": "#5275cf",
               "Sous pression": "#ce6673", "Volatils neutres": "#b49459", "Liquidités": "#8291ae"}


@st.cache_data(show_spinner=False)
def read_result(folder, modified):
    path = Path(folder)
    metadata = json.loads((path / "parameters.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Calcul incomplet.")
    result = {"metadata": metadata, "path": path}
    for name in ["daily_returns", "allocations", "periods", "test_predictions", "training_history", "clusters"]:
        frame = pd.read_csv(path / f"{name}.csv")
        if frame.empty:
            raise ValueError(f"Résultats absents : {name}.")
        for column in ["Date", "entry_date", "exit_date"]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column])
        result[name] = frame
    returns = result["daily_returns"].set_index("Date")
    if (returns.empty or returns.index.has_duplicates or not returns.index.is_monotonic_increasing
            or set(returns.columns) != set(NAMES) or not np.isfinite(returns.to_numpy()).all()):
        raise ValueError("Rendements comparatifs invalides.")
    result["daily_returns"] = returns
    test_dates = pd.to_datetime(metadata["split"]["test_dates"])
    for name in ["periods", "allocations", "test_predictions"]:
        if not result[name]["Date"].isin(test_dates).all():
            raise ValueError("Prédiction ou allocation extérieure à Test.")
    return result


def latest_result():
    failures = []
    for parameters in sorted(RESULTS_DIR.glob("*/parameters.json"), reverse=True):
        try:
            return read_result(str(parameters.parent), parameters.stat().st_mtime_ns), failures
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures.append(f"{parameters.parent.name} : {exc}")
    return None, failures


def chart(fig, height=360):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=12, r=16, t=24, b=16),
                      font=dict(family="Arial, sans-serif", color="#263650", size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8edf4", zerolinecolor="#d6deeb")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def overview(result):
    st.subheader("Analyse et performance du portefeuille")
    returns = result["daily_returns"]
    selected = st.date_input("Période analysée", value=(returns.index.min().date(), returns.index.max().date()),
                             min_value=returns.index.min().date(), max_value=returns.index.max().date())
    if len(selected) != 2:
        st.info("Sélectionnez une date de début et de fin.")
        return
    returns = returns.loc[str(selected[0]):str(selected[1])]
    if returns.empty:
        st.info("Aucune séance commune dans cette période.")
        return
    metrics = pd.DataFrame({c: perf_metrics(returns[c]) for c in returns}).T
    cumulative = np.exp(returns.sum()) - 1
    s = metrics.loc["Strategy_LSTM_Opt"]
    columns = st.columns(5)
    for column, label, value in zip(columns,
            ["Rendement cumulé", "Rendement annualisé", "Volatilité annualisée", "Sharpe", "Drawdown maximal"],
            [f"{cumulative['Strategy_LSTM_Opt']:.2%}", f"{s.AnnReturn:.2%}", f"{s.AnnVol:.2%}", f"{s.Sharpe:.2f}", f"{s.MaxDrawdown:.2%}"]):
        column.metric(label, value)
    st.caption(f"Hors échantillon · {len(returns)} séances communes · "
               f"{returns.index.min():%d/%m/%Y} — {returns.index.max():%d/%m/%Y}")
    equity = np.exp(returns.cumsum()).rename(columns=NAMES)
    st.markdown("**Évolution du capital**")
    chart(px.line(equity, color_discrete_map=COLORS,
                  labels={"value": "Valeur cumulée · base 1", "variable": "", "Date": ""}), 400)
    left, right = st.columns(2)
    with left:
        st.markdown("**Drawdown**")
        drawdown = equity / equity.cummax() - 1
        fig = px.line(drawdown, color_discrete_map=COLORS, labels={"value": "Drawdown", "variable": "", "Date": ""})
        fig.update_yaxes(tickformat=".0%")
        chart(fig, 300)
    with right:
        st.markdown("**Rendements mensuels**")
        monthly = np.expm1(returns.resample("ME").sum()).rename(columns=NAMES)
        fig = px.bar(monthly, barmode="group", color_discrete_map=COLORS,
                     labels={"value": "Rendement", "variable": "", "Date": ""})
        fig.update_yaxes(tickformat=".0%")
        chart(fig, 300)
    table = metrics.copy()
    table.insert(0, "Cumulative", cumulative)
    table.index = table.index.map(NAMES)
    table = table.rename(columns={"Cumulative": "Cumulé", "AnnReturn": "Annualisé", "AnnVol": "Volatilité",
                                  "MaxDrawdown": "Drawdown maximal"})
    st.dataframe(table.style.format({c: "{:.2%}" for c in table.columns if c != "Sharpe"} | {"Sharpe": "{:.2f}"}),
                 width="stretch")
    st.caption("Les mois aux extrémités du filtre peuvent être partiels. Le Sharpe suit la convention des log-rendements annualisés, RF nul.")


def allocation(result):
    st.subheader("Construction et allocation du portefeuille")
    allocations = result["allocations"]
    dates = sorted(allocations["Date"].unique(), reverse=True)
    month = st.selectbox("Mois du signal", dates, format_func=lambda d: pd.Timestamp(d).strftime("%m/%Y"), key="allocation_month")
    weights = allocations[allocations["Date"] == month].copy()
    period = result["periods"].set_index("Date").loc[month]
    method = {"optimized": "Allocation Markowitz", "group_equal_weight": "Allocation de secours par rôle", "cash": "Allocation en liquidités"}[period["method"]]
    st.caption(f"{method} · Entrée à la clôture du {period.entry_date:%d/%m/%Y} · Sortie à la clôture du {period.exit_date:%d/%m/%Y}")
    held = weights[(weights.Ticker != "CASH") & (weights.weight_opt > 1e-8)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions en actions", len(held))
    c2.metric("Poids minimum · actions", f"{held.weight_opt.min():.2%}" if len(held) else "—")
    c3.metric("Poids maximum · actions", f"{held.weight_opt.max():.2%}" if len(held) else "—")
    c4.metric("Somme des poids", f"{weights.weight_opt.sum():.2%}")
    if period["method"] == "cash":
        st.info("Les contraintes d'allocation ne sont pas réalisables sur l'univers disponible. La stratégie détient des liquidités non rémunérées pendant cette période.")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Poids à l'entrée**")
        plot = weights.sort_values("weight_opt", ascending=False).rename(
            columns={"weight_opt": "LSTM + Markowitz", "weight_eq": "Équipondération"})
        fig = px.bar(plot, x="Ticker", y=["LSTM + Markowitz", "Équipondération"], barmode="group",
                     color_discrete_map=COLORS, labels={"value": "Poids", "variable": ""})
        fig.update_yaxes(tickformat=".0%")
        chart(fig)
    with right:
        st.markdown("**Répartition par rôle**")
        roles = weights.groupby("role", as_index=False)["weight_opt"].sum()
        roles = roles[roles.weight_opt > 0]
        chart(px.pie(roles, names="role", values="weight_opt", color="role", hole=.65, color_discrete_map=ROLE_COLORS))
    st.dataframe(weights[["Ticker", "role", "weight_opt", "weight_eq"]].rename(
        columns={"Ticker": "Action", "role": "Rôle", "weight_opt": "LSTM + Markowitz", "weight_eq": "Équipondération"})
        .style.format({"LSTM + Markowitz": "{:.2%}", "Équipondération": "{:.2%}"}), hide_index=True, width="stretch")
    selected = result["clusters"]
    with st.expander("Actions sélectionnées au signal, avant contrôle de disponibilité"):
        st.dataframe(selected[(selected.Date == month) & selected.selected][["Ticker", "role", "TrendScore", "QualityScore"]],
                     hide_index=True, width="stretch")


def lstm_view(result):
    st.subheader("Prévision des rendements avec LSTM")
    meta = result["metadata"]
    scores = meta["test_metrics"]
    st.caption(f"{len(meta['features'])} variables · {meta['n_sequences']:,} séquences avant purge · "
               f"{scores['n_evaluated']} cibles Test observables")
    a, b, c, d = st.columns(4)
    a.metric("MSE · Test", f"{scores['mse']:.6f}")
    b.metric("MAE · Test", f"{scores['mae']:.4f}")
    c.metric("Direction correcte · Test", f"{scores['directional_accuracy']:.1%}")
    d.metric("Époques exécutées", len(result["training_history"]))
    st.markdown("**Masking → LSTM 64 → Dropout 0,2 → LSTM 32 → Dropout 0,2 → Dense 32 ReLU → Dense 1**")
    history = result["training_history"].set_index("epoch").rename(columns={"loss": "Train", "val_loss": "Validation"})
    chart(px.line(history, color_discrete_map={"Train": "#137e83", "Validation": "#d69a3f"},
                  labels={"epoch": "Époque", "value": "Erreur quadratique moyenne", "variable": ""}), 290)
    predictions = result["test_predictions"].dropna(subset=["observed", "pred_return_next"]).copy()
    ticker = st.selectbox("Action à analyser", ["Toutes les actions"] + sorted(predictions.Ticker.unique()), key="prediction_ticker")
    if ticker != "Toutes les actions":
        predictions = predictions[predictions.Ticker == ticker]
    left, right = st.columns(2)
    with left:
        st.markdown("**Observations et prévisions**")
        fig = px.scatter(predictions, x="observed", y="pred_return_next", hover_data=["Ticker", "Date"],
                         color_discrete_sequence=["#137e83"], opacity=.65,
                         labels={"observed": "Rendement observé", "pred_return_next": "Rendement prévu"})
        lo = predictions[["observed", "pred_return_next"]].min().min()
        hi = predictions[["observed", "pred_return_next"]].max().max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#8291ae", dash="dash"), name="Prévision exacte"))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        chart(fig, 330)
    with right:
        st.markdown("**Distribution des erreurs**")
        predictions["error"] = predictions.pred_return_next - predictions.observed
        fig = px.histogram(predictions, x="error", nbins=30, color_discrete_sequence=["#5275cf"],
                           labels={"error": "Prévu − observé", "count": "Observations"})
        fig.update_xaxes(tickformat=".0%")
        chart(fig, 330)
    means = predictions.groupby("Date")[["observed", "pred_return_next"]].mean().rename(
        columns={"observed": "Observé", "pred_return_next": "Prévu"})
    st.markdown("**Rendements moyens par date de signal**")
    fig = px.line(means, markers=True, color_discrete_map={"Observé": "#137e83", "Prévu": "#d69a3f"},
                  labels={"value": "Rendement", "variable": "", "Date": "Signal"})
    fig.update_yaxes(tickformat=".0%")
    chart(fig, 270)
    st.caption("Les indicateurs portent sur toutes les cibles Test observables. Le filtre action concerne les diagnostics graphiques. "
               "Direction correcte : égalité des signes du rendement prévu et observé, zéro traité comme un signe distinct.")


def clustering(result):
    st.subheader("Segmentation et sélection des actions")
    clusters = result["clusters"]
    dates = sorted(clusters.Date.unique(), reverse=True)
    month = st.selectbox("Mois à analyser", dates, format_func=lambda d: pd.Timestamp(d).strftime("%m/%Y"), key="cluster_month")
    monthly = clusters[clusters.Date == month]
    profiles = monthly.groupby(["cluster", "role"], as_index=False).agg(
        Actions=("Ticker", "size"), Retenues=("selected", "sum"),
        TrendScore=("TrendScore", "first"), RiskScore=("RiskScore", "first"),
        QualityScore=("QualityScore", "first"), StressScore=("StressScore", "first"))
    cards = st.columns(4)
    for column, row in zip(cards, profiles.itertuples()):
        column.metric(row.role, int(row.Actions))
        column.caption(f"Cluster {row.cluster} · {int(row.Retenues)} retenues")
    selected_roles = st.multiselect("Rôles affichés", list(ROLE_COLORS)[:-1], default=list(ROLE_COLORS)[:-1])
    filtered = monthly[monthly.role.isin(selected_roles)]
    if filtered.empty:
        st.info("Sélectionnez au moins un rôle pour afficher les actions.")
        return
    fig = px.scatter(filtered, x="return_12m", y="atr_pct", color="role", symbol="selected",
                     hover_name="Ticker", hover_data=["cluster", "rsi"], color_discrete_map=ROLE_COLORS,
                     labels={"return_12m": "Momentum 12 mois · moyenne mensuelle", "atr_pct": "ATR / prix ajusté", "role": "", "selected": "Retenue"})
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    chart(fig, 420)
    st.dataframe(profiles[profiles.role.isin(selected_roles)].rename(columns={"cluster": "Cluster", "role": "Rôle"})
                 .style.format({c: "{:.4f}" for c in ["TrendScore", "RiskScore", "QualityScore", "StressScore"]}),
                 hide_index=True, width="stretch")
    with st.expander("Actions, scores et sélection", expanded=True):
        st.dataframe(filtered[["Ticker", "cluster", "role", "selected", "TrendScore", "QualityScore", "return_12m", "atr_pct"]],
                     hide_index=True, width="stretch")
    st.caption("Les scores sont communs aux actions d'un même cluster. La sélection retient 50 % des actions de chaque rôle éligible, avec un minimum de 10 et un maximum de 60, selon la disponibilité.")


def methodology(result):
    st.subheader("Méthodologie et paramètres")
    st.markdown("La stratégie combine des indicateurs techniques, des expositions Fama-French, une segmentation KMeans, "
                "une prévision LSTM et une allocation Markowitz. Les résultats financiers reposent uniquement sur les signaux Test.")
    st.markdown("**Données → indicateurs → facteurs et betas → clusters et rôles → sélection → LSTM → allocation → backtest**")
    st.markdown("- **Sélection :** quatre clusters KMeans ; actions des groupes Gagnants dynamiques et Stables de qualité.\n"
                "- **Apprentissage :** dates uniques réparties à 70/15/15, purge des cibles aux frontières, normalisation ajustée sur Train.\n"
                "- **Exécution :** clôture de la première séance après le signal ; positions conservées jusqu'au rééquilibrage suivant.\n"
                "- **Allocation :** covariance sur douze mois, poids maximal de 10 %, deux groupes à 50 % à l'entrée.\n"
                "- **Secours :** équipondération par rôle si réalisable, sinon liquidités non rémunérées.\n"
                "- **Évaluation :** variation exacte de la valeur du portefeuille, puis log-rendements et dates communes avec SPY.")
    with st.expander("Conventions et limites", expanded=True):
        st.markdown("Les facteurs mensuels officiels sont décalés de **deux mois**. Ce délai est une hypothèse conservatrice : "
                    "les dates historiques de publication et les révisions ne sont pas vérifiées. L'univers actuel du S&P 500 "
                    "introduit un biais de survivance. Les résultats excluent frais, slippage et impact de marché.")
        st.markdown("Le Sharpe affiché utilise la moyenne annualisée des **log-rendements**, une volatilité avec `ddof=0` et RF nul. "
                    "Le taux RF observé intervient uniquement dans la régression factorielle. Les jours en liquidités restent dans les statistiques.")
        st.code("AnnReturn = exp(mean(r) * 252) - 1\nAnnVol = std(r, ddof=0) * sqrt(252)\nSharpe = mean(r) * 252 / (AnnVol + 1e-12)\nequity = exp(cumsum(r))\nMaxDrawdown = min(equity / cummax(equity) - 1)", language="python")
        st.markdown("[Source des facteurs : Kenneth French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)")
    meta = result["metadata"]
    with st.expander("Paramètres et découpage temporel"):
        st.dataframe(pd.DataFrame(list(meta["parameters"].items()), columns=["Paramètre", "Valeur"]).astype(str), hide_index=True, width="stretch")
        split = meta["split"]
        st.dataframe(pd.DataFrame([
            {"Ensemble": label, "Premier signal": split[f"{name}_dates"][0],
             "Dernier signal": split[f"{name}_dates"][-1], "Séquences retenues": split[f"n_{name}"]}
            for name, label in [("train", "Train"), ("validation", "Validation"), ("test", "Test")]
        ]), hide_index=True, width="stretch")
        st.caption("Les plages portent sur les signaux candidats. Les cibles indisponibles ou dépassant la frontière suivante sont retirées de Train et Validation.")
    st.markdown("**Télécharger les résultats**")
    columns = st.columns(4)
    for column, name, label in zip(columns,
            ["daily_returns.csv", "allocations.csv", "test_predictions.csv", "parameters.json"],
            ["Rendements", "Allocations", "Prédictions Test", "Paramètres"]):
        column.download_button(label, data=(result["path"] / name).read_bytes(), file_name=name,
                               mime="application/json" if name.endswith("json") else "text/csv")


def main():
    st.set_page_config(page_title="Quantitative Portfolio", page_icon="📈", layout="wide")
    st.markdown("""<style>
        .stApp {background:#f7f9fc;color:#20314d;}
        .block-container {max-width:1500px;padding-top:2.2rem;}
        [data-testid="stMetric"] {background:white;border:1px solid #e5ebf3;border-radius:12px;padding:16px;}
        [data-testid="stMetricValue"] {font-size:1.7rem;color:#137e83;}
        [data-testid="stSidebar"] {background:#edf2f8;}
        .stTabs [data-baseweb="tab-list"] {gap:24px;border-bottom:1px solid #dce4ee;}
        .stTabs [data-baseweb="tab"] {font-size:15px;}
        </style>""", unsafe_allow_html=True)
    st.title("Portfolio Analytics")
    st.caption("Segmentation des actions, prévision des rendements et allocation de portefeuille.")
    result, failures = latest_result()
    with st.sidebar:
        st.markdown("### Quantitative Portfolio")
        st.caption("Recherche académique\n\nLSTM · Markowitz · Fama-French")
        st.divider()
        if result:
            returns = result["daily_returns"]
            st.markdown("**Évaluation hors échantillon**")
            st.markdown(f"**{len(returns)}** séances comparables")
            st.caption(f"{result['periods'].entry_date.min():%d/%m/%Y} → {returns.index.max():%d/%m/%Y}")
        else:
            st.caption("Les résultats apparaîtront après l'exécution du pipeline.")
    if result is None:
        st.info("Aucun résultat complet disponible. Le pipeline prépare les données, entraîne le modèle et enregistre les résultats pour le dashboard.")
        st.code("python pipeline.py --prepare-only\npython pipeline.py --train", language="bash")
        if failures:
            with st.expander("Résultats non chargeables"):
                st.write("\n".join(failures))
        st.stop()
    tabs = st.tabs(["Vue d'ensemble", "Allocation du portefeuille", "Modèle LSTM", "Clustering et sélection", "Méthodologie"])
    for tab, render in zip(tabs, [overview, allocation, lstm_view, clustering, methodology]):
        with tab:
            render(result)


if __name__ == "__main__":
    main()
