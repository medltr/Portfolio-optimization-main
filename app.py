"""Tableau de bord du projet (lecture seule des résultats déjà calculés).

Lancement : streamlit run app.py
L'interface ne lance aucun entraînement ni téléchargement : elle lit les dossiers de results/.
Thème : .streamlit/config.toml ; style : assets/style.css ; graphiques : modèle Plotly commun (plus bas).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtest import perf_metrics
from config import RESULTS_DIR

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Palette et noms
# ---------------------------------------------------------------------------
ACCENT = "#3B6FD4"
NEUTRAL = "#8A94A6"
GAIN, LOSS = "#1E9E5A", "#D64545"
FONT = "Inter, IBM Plex Sans, Segoe UI, system-ui, sans-serif"

# nom simple, description, couleur (aucun vert ni rouge : réservés aux gains et pertes)
FIXED = {
    "equal_weight_50_50": ("Répartition égale", "Même somme sur chaque action retenue, moitié Gagnants, moitié Stables.", "#3B6FD4"),
    "equal_weight_none": ("Répartition égale simple (1/N)", "Même somme sur chaque action retenue, sans quota par famille.", "#93B4EA"),
    "Baseline_EqualWeight": ("Répartition égale simple (1/N)", "Même somme sur chaque action retenue, sans quota par famille.", "#93B4EA"),
    "min_variance_50_50": ("Prudent", "Cherche chaque mois le mélange d'actions qui varie le moins.", "#1BA3B8"),
    "lstm_raw_max_sharpe_50_50": ("IA, méthode d'origine", "Un réseau de neurones prévoit les rendements ; un optimiseur concentre l'argent sur les meilleures prévisions.", "#8E5CC4"),
    "Strategy_LSTM_Opt": ("IA, méthode d'origine", "Un réseau de neurones prévoit les rendements ; un optimiseur concentre l'argent sur les meilleures prévisions.", "#8E5CC4"),
    "lstm_relative_tilt_50_50": ("IA, inclinaison légère", "Répartition égale, légèrement penchée vers les actions préférées par le réseau de neurones.", "#C29BE3"),
    "momentum_max_sharpe_50_50": ("Tendance, concentré", "Concentre l'argent sur les actions qui ont le plus monté sur un an.", "#E07B28"),
    "momentum_tilt_50_50": ("Tendance, inclinaison légère", "Répartition égale, légèrement penchée vers les actions qui ont le plus monté sur un an.", "#D9A53B"),
    "SPY": ("Marché (SPY)", "Fonds qui suit les 500 plus grandes entreprises américaines : la référence.", NEUTRAL),
}
FALLBACK_COLORS = ["#4C5FD5", "#2B8CBE", "#A06CD5", "#C77C48", "#6B7FA3", "#B07AA1", "#5DA5DA", "#9C8A5B"]
SIGNALS = {
    "lstm_raw": ("IA (rendement)", "Le réseau de neurones prévoit le rendement de chaque action le mois suivant."),
    "lstm_relative": ("IA (relatif)", "Le réseau de neurones prévoit si une action fera mieux ou moins bien que les autres."),
    "momentum": ("Tendance 12 mois", "Préfère les actions qui ont le plus monté sur 12 mois, sans le dernier."),
    "low_vol": ("Actions calmes", "Préfère les actions dont le prix varie le moins."),
    "mom_lowvol": ("Tendance et calme", "Moyenne des deux classements précédents."),
    "reversal_1m": ("Rebond", "Préfère les actions qui ont baissé le mois dernier."),
    "low_beta": ("Peu sensible au marché", "Préfère les actions qui suivent peu les mouvements du marché."),
    "quality": ("Entreprises rentables", "Préfère les actions qui se comportent comme des entreprises très rentables."),
}
SIGNAL_WORDS = {"lstm": "IA", "momentum": "Tendance", "low_vol": "Actions calmes", "mom_lowvol": "Tendance et calme",
                "reversal_1m": "Rebond", "low_beta": "Peu sensible au marché", "quality": "Entreprises rentables"}
ROLE_COLORS = {"Gagnants dynamiques": "#3B6FD4", "Stables de qualité": "#8E5CC4", "Sous pression": NEUTRAL,
               "Volatils neutres": "#C3CAD5", "Liquidités": "#5B6472"}
METHOD_LABELS = {"optimized": "poids calculés par l'optimiseur", "group_equal_weight": "optimiseur en échec, répartition égale de secours",
                 "cash": "aucune répartition possible, argent non investi", "ok": "poids fixés par la règle de répartition",
                 "capped": "poids fixés par la règle, plafond de 10 % appliqué"}
YEAR_LABEL = {"2022": "2022 · sélection", "2023": "2023 · confirmation"}

HELP = {
    "value": "Valeur atteinte par le montant investi à la fin de la période. Flèche : écart avec le marché (SPY).",
    "ret": "Gain moyen par an. Négatif : perte.",
    "vol": "Ampleur des variations sur un an : la mesure habituelle du risque. Moins de 12 % : faible ; 12 à 20 % : moyen ; plus de 20 % : élevé.",
    "sharpe": "Gain par unité de risque. Plus de 1 : très bon ; autour de 0 : rien gagné ; négatif : perte.",
    "dd": "Plus forte chute depuis un plus haut pendant la période.",
    "equity": "Valeur du montant investi, jour par jour. Cliquer sur la légende masque ou affiche une stratégie.",
    "ranking": "Valeur finale de chaque stratégie. Ligne pointillée : montant de départ.",
    "scatter": "Chaque point est une stratégie. Idéal : en haut (rendement élevé) et à gauche (risque faible).",
    "drawdown": "Écart avec le plus haut précédent. 0 % : au sommet.",
    "heatmap": "Rendement de chaque mois : vert pour un gain, rouge pour une perte.",
    "monthly_eur": "Gain ou perte de chaque mois, en euros.",
    "periods": "2022 : marché en baisse, période de sélection. 2023 : reprise, période de confirmation. Une stratégie robuste se classe bien dans les deux.",
    "rank": "Rang de chaque stratégie selon le Sharpe (1 = meilleure), en 2022 puis en 2023. Des lignes qui se croisent signifient que le classement change.",
    "treemap": "Chaque rectangle est une action ; sa surface est proportionnelle à son poids.",
    "weights": "Poids de chaque action ce mois-ci. Ligne pointillée : plafond de 10 %.",
    "roles_time": "Part de l'argent placée dans chaque famille d'actions, mois après mois.",
    "weights_time": "Poids des actions les plus utilisées, mois après mois.",
    "ic": "L'IC mesure chaque mois si le signal classe bien les actions : +1 classement parfait, 0 aucun lien, −1 classement inversé.",
    "tstat": "La statistique t indique si l'IC moyen est assez éloigné de zéro pour ne pas être dû au hasard (environ 2 ou plus). Lignes pointillées : ±2.",
    "ic_cum": "Somme des IC mois après mois. Une courbe qui monte régulièrement signale un classement utile.",
    "pred": "Chaque point est une action un mois donné. Un bon modèle placerait les points le long de la diagonale.",
    "training": "Erreur du modèle pendant l'apprentissage, sur l'entraînement et sur la validation.",
    "signals_table": "Qualité de chaque signal sur la période choisie. Verdict : signal net si t ≥ 2.",
}
GLOSSARY = {
    "Action": "Part d'entreprise, achetée et vendue en bourse.",
    "Rendement": "Variation de la valeur d'un placement, en pourcentage.",
    "SPY": "Fonds qui reproduit le S&P 500, l'indice des 500 plus grandes entreprises américaines.",
    "Portefeuille, poids": "Ensemble des actions détenues ; le poids est la part de l'argent placée sur chacune.",
    "Volatilité": "Ampleur des variations d'un placement sur un an ; mesure habituelle du risque.",
    "Ratio de Sharpe": "Rendement divisé par la volatilité : gain par unité de risque.",
    "Pire baisse (drawdown)": "Plus forte chute depuis un plus haut pendant la période.",
    "Gagnants dynamiques, Stables de qualité": "Les deux familles d'actions retenues chaque mois : forte hausse récente, ou entreprises rentables et peu agitées.",
    "Clustering": "Méthode qui range automatiquement les actions en groupes d'actions qui se ressemblent.",
    "Réseau de neurones (LSTM)": "Programme d'intelligence artificielle qui apprend à partir de séquences de données, ici 12 mois d'historique par action.",
    "Signal": "Note donnée à chaque action pour la classer de la plus prometteuse à la moins prometteuse.",
    "IC (coefficient d'information)": "Ressemblance entre le classement d'un signal et le classement réel des rendements du mois suivant.",
    "Statistique t": "Indique si un résultat est trop net pour être dû au hasard, au-delà d'environ 2.",
    "Walk-forward": "Test qui imite la réalité : à chaque date, seul le passé est utilisé, puis on teste sur la période suivante.",
    "Sélection, confirmation": "On choisit sur 2022 (sélection), puis on vérifie sans rien changer sur 2023 (confirmation).",
    "Biais de survie": "Erreur qui consiste à n'étudier que les entreprises encore présentes aujourd'hui, ce qui embellit le passé.",
    "Coûts de transaction": "Frais payés à chaque achat ou vente ; non comptés ici.",
}


# ---------------------------------------------------------------------------
# Formats de nombres (identiques partout) : 1 décimale pour les %, milliers séparés
# ---------------------------------------------------------------------------
NBSP = " "


def fr(x, digits=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:+,.{digits}f}" if sign else f"{x:,.{digits}f}"
    return s.replace(",", NBSP).replace(".", ",")


def eur(x, sign=False):
    return fr(x, 0, sign) + NBSP + "€"


def pct(x, sign=False):
    return fr(100 * x, 1, sign) + NBSP + "%"


def pts(x):
    return fr(100 * x, 1, True) + NBSP + "pts"


def day(d):
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def month(d):
    return pd.Timestamp(d).strftime("%m/%Y")


def risk_word(vol):
    return "faible" if vol < 0.12 else ("moyen" if vol < 0.20 else "élevé")


def sharpe_word(s):
    return "perte" if s < 0 else ("faible" if s < 0.5 else ("correct" if s < 1 else "bon"))


def ic_verdict(t):
    if pd.isna(t):
        return "trop peu de mois"
    return "signal net" if t >= 2 else ("faible indice" if t >= 1 else ("aucun lien" if t > -1 else "plutôt inversé"))


# ---------------------------------------------------------------------------
# Modèle Plotly commun (clair / sombre)
# ---------------------------------------------------------------------------
def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def template(dark=None):
    dark = is_dark() if dark is None else dark
    fg = "#E5E9F0" if dark else "#1F2937"
    muted = "#9AA5B4" if dark else "#6B7280"
    grid = "rgba(154,165,180,0.16)" if dark else "rgba(31,41,55,0.08)"
    axis = dict(showgrid=False, zeroline=False, showline=False, ticks="", tickfont=dict(color=muted, size=11),
                title=dict(font=dict(color=muted, size=11), standoff=8), automargin=True)
    return go.layout.Template(layout=dict(
        font=dict(family=FONT, size=12, color=fg), separators=", ", colorway=[ACCENT] + FALLBACK_COLORS,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=4, r=12, t=8, b=4),
        xaxis=axis, yaxis=dict(axis, showgrid=True, gridcolor=grid),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left", title=None,
                    font=dict(size=11, color=fg), bgcolor="rgba(0,0,0,0)", itemclick="toggle"),
        hoverlabel=dict(font=dict(family=FONT, size=12), namelength=-1),
        coloraxis=dict(colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=muted, size=10),
                                     title=dict(font=dict(color=muted, size=10)))),
        bargap=0.25, hovermode="closest"))


def time_axis(fig, fmt="%m/%Y"):
    fig.update_xaxes(tickformat=fmt, hoverformat="%d/%m/%Y")
    return fig


# Mise en forme commune à tous les graphiques de l'application. Les couleurs de texte et de grille sont
# laissées au thème Streamlit (theme="streamlit"), qui les adapte dans le navigateur au thème clair ou
# sombre réellement affiché ; tout le reste (police, marges, légende, formats, fonds) est fixé ici.
LAYOUT = dict(font=dict(family=FONT, size=12), separators=", ", margin=dict(l=4, r=12, t=8, b=4),
              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.25, hovermode="closest",
              legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left", title=None,
                          font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
              hoverlabel=dict(font=dict(family=FONT, size=12), namelength=-1))


def show(fig, height=360, key=None, margin=None):
    fig.update_layout(**LAYOUT)
    fig.update_layout(height=height, margin=margin or LAYOUT["margin"])
    for ax in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        is_x = ax.plotly_name.startswith("xaxis")
        if ax.showgrid is None:
            ax.showgrid = not is_x                     # grilles horizontales légères uniquement
        ax.update(zeroline=False, showline=False, ticks="", automargin=True, title_standoff=8,
                  tickfont=dict(size=11), title_font=dict(size=11))
    st.plotly_chart(fig, theme="streamlit", width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]})


def wrap(label, width=18):
    """Coupe un nom long en lignes d'au plus `width` caractères (mot à mot) pour les axes de catégories."""
    import textwrap
    return "<br>".join(textwrap.wrap(label, width)) or label


def diverging():
    return [[0, LOSS], [0.5, "rgba(138,148,166,0.12)"], [1, GAIN]]


# ---------------------------------------------------------------------------
# Lecture des résultats (aucun calcul de modèle)
# ---------------------------------------------------------------------------
def read_csv_dates(path):
    frame = pd.read_csv(path)
    for column in ["Date", "entry_date", "exit_date"]:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def friendly(key, spec=None):
    if key in FIXED:
        return FIXED[key][:2]
    if not spec:
        return key, ""
    c, r, s = spec.get("construction"), spec.get("rule"), spec.get("signal")
    rule = {"none": ", sans quota", "proportional": ", proportionnel", "30_70": ", 30–70 %"}.get(r, "")
    if c == "equal_weight":
        return f"Répartition égale{rule}", "Même somme sur chaque action retenue."
    if c == "min_variance":
        return f"Prudent{rule}", "Cherche chaque mois le mélange d'actions qui varie le moins."
    word = next((v for k, v in SIGNAL_WORDS.items() if s and s.startswith(k)), s or "Signal")
    if c == "max_sharpe":
        return f"{word}, concentré{rule}", "Concentre l'argent sur les actions les mieux notées."
    return f"{word}, inclinaison légère{rule}", "Répartition égale, penchée vers les actions les mieux notées."


def build_strategies(keys, specs, main):
    out, i = {}, 0
    for k in keys:
        name, desc = friendly(k, specs.get(k))
        if k in FIXED:
            color = FIXED[k][2]
        else:
            color, i = FALLBACK_COLORS[i % len(FALLBACK_COLORS)], i + 1
        out[k] = {"name": name, "desc": desc, "color": color, "default": k == main}
    return out


def load_walk_forward(path, meta):
    r = {"mode": "walk_forward", "meta": meta, "path": path}
    r["returns"] = read_csv_dates(path / "daily_returns.csv").set_index("Date")
    keys = [s["key"] for s in meta["strategies"]]
    if set(r["returns"].columns) != set(keys):
        raise ValueError("Colonnes de rendements inattendues.")
    r["periods"] = read_csv_dates(path / "periods.csv")
    alloc_all = path / "allocations_all.csv"
    if alloc_all.exists():
        a = read_csv_dates(alloc_all)
    else:
        a = read_csv_dates(path / "allocations.csv").rename(columns={"weight_opt": "weight"}).assign(strategy=meta["main"])
    r["allocations"] = a[["strategy", "Date", "entry_date", "exit_date", "Ticker", "role", "weight"]]
    pa = path / "periods_all.csv"
    r["periods_all"] = read_csv_dates(pa) if pa.exists() else r["periods"].assign(strategy=meta["main"])
    r["ic"] = read_csv_dates(path / "ic_by_month.csv").set_index("Date")
    r["signals"] = read_csv_dates(path / "signals.csv").set_index(["Date", "Ticker"])
    r["signal_cols"] = [c for c in r["ic"].columns if c in r["signals"].columns]
    th = path / "training_history.csv"
    r["training"] = read_csv_dates(th) if th.exists() else None
    r["clusters"] = read_csv_dates(path / "clusters.csv")
    r["main"] = meta["main"]
    r["strategies"] = build_strategies(keys, {s["key"]: s for s in meta["strategies"]}, meta["main"])
    r["blocks"] = [(pd.Timestamp(b["test_months"][0]), pd.Timestamp(b["test_months"][1])) for b in meta.get("schedule", [])]
    r["params"] = meta.get("parameters", {})
    return r


def load_legacy(path, meta):
    r = {"mode": "single_split", "meta": meta, "path": path}
    returns = read_csv_dates(path / "daily_returns.csv").set_index("Date")
    if set(returns.columns) != {"Strategy_LSTM_Opt", "Baseline_EqualWeight", "SPY"}:
        raise ValueError("Colonnes de rendements inattendues.")
    r["returns"] = returns[["Strategy_LSTM_Opt", "Baseline_EqualWeight", "SPY"]]
    r["periods"] = read_csv_dates(path / "periods.csv")
    a = read_csv_dates(path / "allocations.csv")
    r["allocations"] = pd.concat([a.assign(strategy="Strategy_LSTM_Opt", weight=a["weight_opt"]),
                                  a.assign(strategy="Baseline_EqualWeight", weight=a["weight_eq"])])[
        ["strategy", "Date", "entry_date", "exit_date", "Ticker", "role", "weight"]]
    r["periods_all"] = pd.concat([r["periods"].assign(strategy="Strategy_LSTM_Opt"),
                                  r["periods"].assign(strategy="Baseline_EqualWeight", method="ok", reason="")])
    sig = read_csv_dates(path / "test_predictions.csv").rename(columns={"pred_return_next": "lstm_raw"}).set_index(["Date", "Ticker"])
    r["signals"] = sig
    ev = sig.dropna(subset=["observed", "lstm_raw"])
    r["ic"] = ev.groupby(level="Date").apply(lambda g: g["lstm_raw"].corr(g["observed"], method="spearman")).to_frame("lstm_raw")
    r["signal_cols"] = ["lstm_raw"]
    r["training"] = read_csv_dates(path / "training_history.csv")
    r["clusters"] = read_csv_dates(path / "clusters.csv")
    r["main"] = "Strategy_LSTM_Opt"
    r["strategies"] = build_strategies(list(r["returns"].columns), {}, r["main"])
    r["blocks"] = []
    r["params"] = meta.get("parameters", {})
    return r


@st.cache_data(show_spinner=False)
def load_result(folder, modified):
    path = Path(folder)
    meta = json.loads((path / "parameters.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError("Calcul incomplet.")
    r = load_walk_forward(path, meta) if meta.get("eval_mode") == "walk_forward" else load_legacy(path, meta)
    ret = r["returns"]
    if ret.empty or ret.index.has_duplicates or not ret.index.is_monotonic_increasing or not np.isfinite(ret.to_numpy()).all():
        raise ValueError("Rendements invalides.")
    lab = pd.Series(np.nan, index=ret.index, dtype=object)       # année du signal de chaque jour détenu
    for _, p in r["periods"].iterrows():
        lab[(ret.index > p["entry_date"]) & (ret.index <= p["exit_date"])] = str(pd.Timestamp(p["Date"]).year)
    r["day_year"] = lab
    return r


def list_results():
    valid, failures = [], []
    for p in sorted(RESULTS_DIR.glob("*/parameters.json"), reverse=True):
        try:
            load_result(str(p.parent), p.stat().st_mtime_ns)
            valid.append(p.parent)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures.append(f"{p.parent.name} : {exc}")
    return valid, failures


# ---------------------------------------------------------------------------
# Calculs d'affichage
# ---------------------------------------------------------------------------
def metrics_for(ret):
    m = perf_metrics(ret)
    m["Final"] = float(np.exp(ret.sum()))
    return m


def ic_stats(ic):
    ic = pd.Series(ic, dtype=float).dropna()
    n = len(ic)
    sd = ic.std(ddof=1) if n > 1 else np.nan
    t = ic.mean() / sd * np.sqrt(n) if n > 1 and sd > 0 else np.nan
    return {"mean": ic.mean() if n else np.nan, "pos": int((ic > 0).sum()), "n": n, "t": t}


def name(ctx, k):
    return ctx["r"]["strategies"][k]["name"]


def color(ctx, k):
    return ctx["r"]["strategies"][k]["color"]


def line_chart(ctx, frame, y_title, hover, key, height=380):
    """Courbes par stratégie, couleurs fixes, SPY en pointillé."""
    fig = go.Figure()
    for k in frame:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[k], name=name(ctx, k), mode="lines",
                                 line=dict(color=color(ctx, k), width=2.2 if ctx["r"]["strategies"][k]["default"] else 1.6,
                                           dash="dot" if k == "SPY" else "solid"),
                                 hovertemplate=f"{name(ctx, k)}<br>%{{x|%d/%m/%Y}} · {hover}<extra></extra>"))
    fig.update_yaxes(title=y_title)
    show(time_axis(fig), height, key)


# ---------------------------------------------------------------------------
# Composants
# ---------------------------------------------------------------------------
def section(text, help_key=None):
    st.markdown(f"##### {text}", help=HELP.get(help_key) if help_key else None)


def note(text):
    st.markdown(f'<div class="pf-note">{text}</div>', unsafe_allow_html=True)


def table(df, filename, formats=None, key=None):
    """Tableau triable (valeurs numériques conservées) avec formats français et export CSV."""
    styled = df.style.format(formats or {}, na_rep="–")
    st.dataframe(styled, hide_index=True, width="stretch")
    st.download_button("Exporter en CSV", df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", key=key or f"dl_{filename}")


F_PCT = lambda v: pct(v / 100)  # noqa: E731  (colonnes stockées en %)
F_EUR = lambda v: eur(v)  # noqa: E731
F_DEC = lambda v: fr(v, 2)  # noqa: E731
F_SDEC = lambda v: fr(v, 2, True)  # noqa: E731


def kpi_row(ctx, key):
    ret, amount = ctx["returns"], ctx["amount"]
    m = metrics_for(ret[key])
    s = metrics_for(ret["SPY"]) if "SPY" in ret and key != "SPY" else None
    vs = (lambda text: f"{text} vs SPY") if s is not None else (lambda text: None)
    c = st.columns(5)
    c[0].metric("Valeur finale", eur(amount * m["Final"]), vs(eur(amount * (m["Final"] - s["Final"]), True)) if s is not None else None,
                help=HELP["value"], border=True)
    c[1].metric("Rendement annuel", pct(m["AnnReturn"]), vs(pts(m["AnnReturn"] - s["AnnReturn"])) if s is not None else None,
                help=HELP["ret"], border=True)
    c[2].metric(f"Volatilité · risque {risk_word(m['AnnVol'])}", pct(m["AnnVol"]),
                vs(pts(m["AnnVol"] - s["AnnVol"])) if s is not None else None, delta_color="inverse", help=HELP["vol"], border=True)
    c[3].metric(f"Sharpe · {sharpe_word(m['Sharpe'])}", fr(m["Sharpe"], 2),
                vs(fr(m["Sharpe"] - s["Sharpe"], 2, True)) if s is not None else None, help=HELP["sharpe"], border=True)
    c[4].metric("Pire baisse", pct(m["MaxDrawdown"]), vs(pts(m["MaxDrawdown"] - s["MaxDrawdown"])) if s is not None else None,
                help=HELP["dd"], border=True)


# ---------------------------------------------------------------------------
# Graphique des rangs (page « Par période »)
# ---------------------------------------------------------------------------
def rank_figure(ranks, labels, colors, years=("2022", "2023"), dark=None):
    """Une ligne par stratégie reliant son rang de la 1re à la 2e année ; noms lisibles des deux côtés.

    `ranks` : DataFrame (index = clé de stratégie, colonnes = années, valeurs = rang entier).
    L'axe horizontal utilise des positions 0 et 1 étiquetées par les années (et non des nombres 2022/2023).
    `dark` (True/False) n'est utilisé que pour exporter le graphique hors de l'application.
    """
    fig = go.Figure()
    n = len(ranks)
    for k in ranks.index:
        y0, y1 = int(ranks.loc[k, years[0]]), int(ranks.loc[k, years[1]])
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y0, y1], mode="lines+markers+text", name=labels[k],
            line=dict(color=colors[k], width=2.5), marker=dict(size=9, color=colors[k]),
            text=[f"{labels[k]}  {y0}", f"{y1}  {labels[k]}"], textposition=["middle left", "middle right"],
            textfont=dict(size=11, color=colors[k]), cliponaxis=False,
            hovertemplate=f"{labels[k]}<br>{years[0]} : rang {y0}<br>{years[1]} : rang {y1}<extra></extra>"))
    fig.update_xaxes(tickvals=[0, 1], ticktext=[YEAR_LABEL.get(y, y) for y in years], range=[-1.25, 2.25],
                     showgrid=False, side="top", tickfont=dict(size=12))
    fig.update_yaxes(autorange=False, range=[n + 0.6, 0.4], tickvals=list(range(1, n + 1)), title="Rang (1 = meilleure)",
                     showgrid=False)
    fig.update_layout(showlegend=False, margin=dict(l=4, r=4, t=36, b=4))
    if dark is not None:
        fig.update_layout(template=template(dark))
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(ctx):
    r = ctx["r"]
    main = r["main"]
    st.markdown(f"### {name(ctx, main)} · stratégie par défaut", help=r["strategies"][main]["desc"])
    note("Simulation sur données passées, sans frais. Ne constitue pas un conseil d'investissement.")
    kpi_row(ctx, main)
    left, right = st.columns([2, 1])
    with left:
        section(f"Valeur de {eur(ctx['amount'])} investis", "equity")
        line_chart(ctx, ctx["amount"] * np.exp(ctx["returns"][ctx["shown"]].cumsum()), "Valeur (€)", "%{y:,.0f} €", "ov_eq", 400)
    with right:
        section("Valeur finale par stratégie", "ranking")
        final = (ctx["amount"] * np.exp(ctx["returns"][ctx["shown"]].sum())).sort_values()
        fig = go.Figure(go.Bar(x=final.values, y=[name(ctx, k) for k in final.index], orientation="h",
                               marker_color=[color(ctx, k) for k in final.index],
                               text=[eur(v) for v in final.values], textposition="inside", insidetextanchor="end",
                               textfont=dict(color="white", size=11), hovertemplate="%{y}<br>%{x:,.0f} €<extra></extra>"))
        fig.add_vline(x=ctx["amount"], line_dash="dot", line_color=NEUTRAL, line_width=1)
        fig.update_xaxes(title="Valeur finale (€)", showgrid=True, gridcolor="rgba(128,140,155,0.12)")
        fig.update_yaxes(showgrid=False)
        show(fig, 400, "ov_rank")
    with st.expander("En savoir plus sur le projet"):
        st.markdown("Chaque mois, le projet retient des actions américaines parmi les plus échangées, les range en familles, "
                    "puis répartit l'argent selon plusieurs méthodes : à parts égales, de façon prudente, en suivant la tendance "
                    "ou avec un réseau de neurones. Chaque stratégie est simulée sur des données passées, sans information du futur, "
                    "et comparée au marché (SPY).")
        if r["mode"] == "single_split":
            st.markdown("Ce dossier est un ancien résultat (découpage unique, 6 mois de test) : trois stratégies seulement.")


def page_simulator(ctx):
    r = ctx["r"]
    full = ctx["returns"]
    keys = list(full.columns)
    options = [k for k in keys if k != "SPY"] or keys
    c1, c2, c3 = st.columns([1, 1, 2])
    amount = c1.number_input("Montant investi (€)", min_value=100, max_value=10_000_000, value=int(ctx["amount"]), step=500, key="sim_amount")
    start = c2.date_input("Date de départ", value=full.index.min().date(), min_value=full.index.min().date(),
                          max_value=(full.index.max() - pd.Timedelta(days=7)).date(), format="DD/MM/YYYY", key="sim_start")
    key = c3.selectbox("Stratégie", options, index=options.index(r["main"]) if r["main"] in options else 0,
                       format_func=lambda k: r["strategies"][k]["name"] if k in r["strategies"] else k, key="sim_strategy",
                       help=r["strategies"][r["main"]]["desc"])
    ret = full.loc[full.index >= pd.Timestamp(start)]
    if len(ret) < 5:
        st.info("Choisissez une date de départ plus ancienne.")
        return
    value = amount * np.exp(ret[key].cumsum())
    spy = amount * np.exp(ret["SPY"].cumsum()) if "SPY" in ret else None
    loss = value - value.cummax()
    trough = loss.idxmin()
    peak_date = value.loc[:trough].idxmax()
    monthly = np.expm1(ret[key].resample("ME").sum())
    wins = int((monthly > 0).sum())
    final = value.iloc[-1]
    m = metrics_for(ret[key])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valeur finale", eur(final), eur(final - amount, True), border=True,
              help=f"Valeur au {day(ret.index[-1])} de {eur(amount)} placés le {day(ret.index[0])}.")
    k2.metric("Marché (SPY) sur la même période", eur(spy.iloc[-1]) if spy is not None else "–",
              eur(spy.iloc[-1] - amount, True) if spy is not None else None, border=True, help="Même montant placé sur SPY.")
    k3.metric("Pire perte depuis un plus haut", eur(loss.min()), pct(value[trough] / value[peak_date] - 1), delta_color="off", border=True,
              help=f"Le {day(trough)}, le placement valait {eur(value[trough])}, contre {eur(value[peak_date])} au plus haut du {day(peak_date)}.")
    k4.metric(f"Mois gagnants · risque {risk_word(m['AnnVol'])}", f"{wins} sur {len(monthly)}", border=True,
              help=f"Meilleur mois : {pct(monthly.max(), True)} ; pire mois : {pct(monthly.min(), True)}. Volatilité : {pct(m['AnnVol'])}.")
    note(f"{eur(amount)} placés le {day(ret.index[0])} auraient valu {eur(final)} le {day(ret.index[-1])} "
         f"({pct(final / amount - 1, True)}). Pire moment : le {day(trough)}, {eur(-loss.min())} sous le plus haut du {day(peak_date)}.")
    left, right = st.columns([2, 1])
    with left:
        section("Valeur du placement, comparée au marché", "equity")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=value.index, y=value, name=name(ctx, key), line=dict(color=color(ctx, key), width=2.2),
                                 hovertemplate="%{x|%d/%m/%Y} · %{y:,.0f} €<extra></extra>"))
        if spy is not None:
            fig.add_trace(go.Scatter(x=spy.index, y=spy, name=name(ctx, "SPY"), line=dict(color=NEUTRAL, width=1.6, dash="dot"),
                                     hovertemplate="%{x|%d/%m/%Y} · %{y:,.0f} €<extra></extra>"))
        fig.add_hline(y=amount, line_dash="dot", line_color=NEUTRAL, line_width=1)
        fig.add_trace(go.Scatter(x=[trough], y=[value[trough]], mode="markers", name="Pire moment",
                                 marker=dict(size=9, color=LOSS, line=dict(width=1, color="white")),
                                 hovertemplate="Pire moment<br>%{x|%d/%m/%Y} · %{y:,.0f} €<extra></extra>"))
        fig.update_yaxes(title="Valeur (€)")
        show(time_axis(fig), 400, "sim_eq")
    with right:
        section("Gain ou perte de chaque mois", "monthly_eur")
        ends = value.resample("ME").last()
        gains = ends.diff()
        gains.iloc[0] = ends.iloc[0] - amount
        fig = go.Figure(go.Bar(x=gains.index, y=gains.values, marker_color=np.where(gains.values >= 0, GAIN, LOSS),
                               hovertemplate="%{x|%m/%Y} · %{y:+,.0f} €<extra></extra>"))
        fig.update_yaxes(title="€")
        show(time_axis(fig), 400, "sim_month")


def page_comparison(ctx):
    r = ctx["r"]
    ret = ctx["returns"][ctx["shown"]]
    rows = []
    for k in ret:
        m = metrics_for(ret[k])
        rows.append({"Stratégie": name(ctx, k), "Par défaut": "oui" if r["strategies"][k]["default"] else "",
                     "Valeur finale": ctx["amount"] * m["Final"], "Rendement annuel": 100 * m["AnnReturn"],
                     "Volatilité": 100 * m["AnnVol"], "Risque": risk_word(m["AnnVol"]), "Sharpe": m["Sharpe"],
                     "Pire baisse": 100 * m["MaxDrawdown"]})
    t = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    t.insert(0, "Rang", range(1, len(t) + 1))
    section("Synthèse des stratégies", "sharpe")
    table(t, "comparaison_strategies.csv", {"Valeur finale": F_EUR, "Rendement annuel": F_PCT, "Volatilité": F_PCT,
                                             "Sharpe": F_DEC, "Pire baisse": F_PCT}, key="cmp_dl")
    left, right = st.columns(2)
    with left:
        section("Rendement et risque", "scatter")
        fig = go.Figure()
        for k in ret:
            m = metrics_for(ret[k])
            fig.add_trace(go.Scatter(x=[100 * m["AnnVol"]], y=[100 * m["AnnReturn"]], mode="markers+text", name=name(ctx, k),
                                     marker=dict(size=13, color=color(ctx, k), line=dict(width=1, color="white")),
                                     text=[name(ctx, k)], textposition="top center", textfont=dict(size=10, color=color(ctx, k)),
                                     cliponaxis=False,
                                     hovertemplate=f"{name(ctx, k)}<br>Volatilité %{{x:.1f}} % · Rendement %{{y:.1f}} %<extra></extra>"))
        fig.add_hline(y=0, line_color=NEUTRAL, line_width=1, line_dash="dot")
        fig.update_layout(showlegend=False)
        vols = [100 * metrics_for(ret[k])["AnnVol"] for k in ret]
        span = max(max(vols) - min(vols), 1.0)
        fig.update_xaxes(title="Volatilité annuelle (%)", showgrid=True, gridcolor="rgba(128,140,155,0.12)",
                         range=[min(vols) - 0.25 * span, max(vols) + 0.25 * span])
        fig.update_yaxes(title="Rendement annuel (%)")
        show(fig, 380, "cmp_scatter")
    with right:
        section("Baisse depuis le plus haut", "drawdown")
        eq = np.exp(ret.cumsum())
        line_chart(ctx, 100 * (eq / eq.cummax() - 1), "Baisse (%)", "%{y:.1f} %", "cmp_dd", 380)
    section("Rendement mensuel", "heatmap")
    monthly = (np.expm1(ret.resample("ME").sum()) * 100).T
    monthly.index = [name(ctx, k) for k in monthly.index]
    monthly.columns = [month(d) for d in monthly.columns]
    lim = float(np.nanmax(np.abs(monthly.to_numpy()))) or 1.0
    # valeurs écrites en blanc dans les cases nettement colorées seulement : lisible en thème clair et sombre
    labels = [[fr(v, 1) if abs(v) >= 0.4 * lim else "" for v in row] for row in monthly.values]
    fig = go.Figure(go.Heatmap(z=monthly.values, x=monthly.columns, y=monthly.index, zmin=-lim, zmax=lim, colorscale=diverging(),
                               text=labels, texttemplate="%{text}",
                               textfont=dict(size=10, color="white"), xgap=2, ygap=2, colorbar=dict(title="%"),
                               hovertemplate="%{y} · %{x} · %{z:.1f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    show(fig, 80 + 34 * len(monthly), "cmp_heat")


def page_periods(ctx):
    r = ctx["r"]
    full, years = r["returns"][ctx["shown"]], r["day_year"]
    avail = [y for y in ["2022", "2023"] if (years == y).any()]
    if len(avail) < 2:
        st.info("Ce dossier ne couvre qu'une période (" + ", ".join(avail) + "). La comparaison 2022 / 2023 "
                "est disponible pour les résultats walk-forward.")
        return
    c1, c2 = st.columns([3, 2])
    metric = c1.radio("Indicateur", ["Sharpe", "AnnReturn", "AnnVol", "MaxDrawdown"], horizontal=True, key="per_metric",
                      format_func=lambda m: {"Sharpe": "Sharpe", "AnnReturn": "Rendement annuel", "AnnVol": "Volatilité",
                                             "MaxDrawdown": "Pire baisse"}.get(m, m))
    c2.caption("Comparaison fixe 2022 / 2023, indépendante du filtre de période.")
    rows = []
    for y in avail:
        for k in full:
            m = metrics_for(full[k][years == y])
            rows.append({"key": k, "Année": y, **m.to_dict()})
    data = pd.DataFrame(rows)
    sh = data.pivot(index="key", columns="Année", values="Sharpe")
    ranks = sh.rank(ascending=False, method="first").astype(int)
    left, right = st.columns(2)
    with left:
        section("2022 et 2023 par stratégie", "periods")
        scale, unit = (1, "") if metric == "Sharpe" else (100, " %")
        fig = go.Figure()
        for y, c in zip(avail, [NEUTRAL, ACCENT]):
            d = data[data["Année"] == y].set_index("key").reindex(full.columns)
            fig.add_trace(go.Bar(x=[wrap(name(ctx, k)) for k in d.index], y=d[metric] * scale, name=YEAR_LABEL[y], marker_color=c,
                                 hovertemplate=f"%{{x}} · {YEAR_LABEL[y]} : %{{y:.{2 if metric == 'Sharpe' else 1}f}}{unit}<extra></extra>"))
        fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
        fig.update_layout(barmode="group")
        fig.update_yaxes(title={"Sharpe": "Sharpe", "AnnReturn": "Rendement annuel (%)", "AnnVol": "Volatilité (%)",
                                "MaxDrawdown": "Pire baisse (%)"}[metric])
        fig.update_xaxes(tickangle=0)
        show(fig, 430, "per_bar")
    with right:
        section("Évolution du classement", "rank")
        fig = rank_figure(ranks, {k: name(ctx, k) for k in ranks.index}, {k: color(ctx, k) for k in ranks.index}, avail)
        show(fig, 430, "per_rank", margin=dict(l=4, r=4, t=36, b=4))
    best = ranks["2022"].idxmin()
    note(f"Première en 2022 : {name(ctx, best)}, {ranks.loc[best, '2023']}e sur {len(ranks)} en 2023. "
         f"Première en 2023 : {name(ctx, ranks['2023'].idxmin())}.")
    t = pd.DataFrame({"Stratégie": [name(ctx, k) for k in sh.index], "Sharpe 2022": sh["2022"].values, "Rang 2022": ranks["2022"].values,
                      "Sharpe 2023": sh["2023"].values, "Rang 2023": ranks["2023"].values})
    t["Places gagnées"] = t["Rang 2022"] - t["Rang 2023"]
    section("Classement détaillé")
    table(t.sort_values("Rang 2022"), "classement_2022_2023.csv",
          {"Sharpe 2022": F_DEC, "Sharpe 2023": F_DEC, "Places gagnées": lambda v: f"{v:+d}"}, key="per_dl")
    if r["blocks"]:
        with st.expander("Détail par semestre"):
            rows = []
            for i, (a, b) in enumerate(r["blocks"], start=1):
                months = r["periods"][(r["periods"]["Date"] >= a) & (r["periods"]["Date"] <= b)]
                mask = pd.Series(False, index=full.index)
                for _, p in months.iterrows():
                    mask |= (full.index > p["entry_date"]) & (full.index <= p["exit_date"])
                for k in full:
                    m = metrics_for(full[k][mask])
                    rows.append({"Semestre": f"{i} · {month(a)} – {month(b)}", "Stratégie": name(ctx, k),
                                 "Rendement annuel": 100 * m["AnnReturn"], "Sharpe": m["Sharpe"], "Pire baisse": 100 * m["MaxDrawdown"]})
            table(pd.DataFrame(rows), "resultats_par_semestre.csv",
                  {"Rendement annuel": F_PCT, "Sharpe": F_DEC, "Pire baisse": F_PCT}, key="per_blocks")
            st.caption("Le dernier semestre peut ne contenir qu'un mois : chiffres peu fiables.")


def page_portfolio(ctx):
    r = ctx["r"]
    alloc = r["allocations"]
    keys = [k for k in r["strategies"] if k in alloc["strategy"].unique()]
    c1, c2 = st.columns([2, 1])
    key = c1.selectbox("Stratégie", keys, index=keys.index(r["main"]) if r["main"] in keys else 0,
                       format_func=lambda k: r["strategies"][k]["name"] if k in r["strategies"] else k, key="pf_strategy")
    a = alloc[alloc["strategy"] == key]
    dates = sorted(a["Date"].unique(), reverse=True)
    sel = c2.selectbox("Mois du signal", dates, format_func=lambda d: d if isinstance(d, str) else month(d), key="pf_month")
    w = a[a["Date"] == sel]
    held = w[(w["Ticker"] != "CASH") & (w["weight"] > 1e-6)]
    per = r["periods_all"]
    per = per[(per["strategy"] == key) & (per["Date"] == sel)]
    method = METHOD_LABELS.get(per["method"].iloc[0], per["method"].iloc[0]) if len(per) else "–"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Actions détenues", f"{len(held)}", help="Nombre d'actions avec un poids non nul.", border=True)
    k2.metric("Plus gros poids", pct(held["weight"].max()) if len(held) else "–", help="Plafond : 10 %.", border=True)
    k3.metric("Part des Gagnants", pct(held.loc[held["role"] == "Gagnants dynamiques", "weight"].sum()),
              help="Actions en forte hausse récente.", border=True)
    k4.metric("Part des Stables", pct(held.loc[held["role"] == "Stables de qualité", "weight"].sum()),
              help="Actions d'entreprises rentables et peu agitées.", border=True)
    note(f"Achat le {day(w['entry_date'].iloc[0])}, revente le {day(w['exit_date'].iloc[0])} · {method}.")
    if (w["Ticker"] == "CASH").any():
        st.info("Aucune répartition respectant les règles n'était possible ce mois-là : argent non investi.")
    left, right = st.columns(2)
    with left:
        section("Répartition par famille et par action", "treemap")
        if len(held):
            fig = px.treemap(held.assign(Poids=held["weight"] * 100), path=[px.Constant("Portefeuille"), "role", "Ticker"],
                             values="Poids", color="role", color_discrete_map=ROLE_COLORS | {"(?)": "rgba(0,0,0,0)"})
            fig.update_traces(hovertemplate="%{label} · %{value:.1f} %<extra></extra>", texttemplate="%{label}<br>%{value:.1f} %",
                              marker=dict(line=dict(width=1, color="rgba(255,255,255,0.6)")), root_color="rgba(0,0,0,0)")
            show(fig, 420, "pf_tree")
    with right:
        section("Poids de chaque action", "weights")
        if len(held):
            hs = held.sort_values("weight", ascending=False)
            fig = go.Figure()
            for role, g in hs.groupby("role", sort=False):
                fig.add_trace(go.Bar(x=g["Ticker"], y=g["weight"] * 100, name=role, marker_color=ROLE_COLORS.get(role, NEUTRAL),
                                     hovertemplate="%{x} · %{y:.1f} %<extra></extra>"))
            fig.add_hline(y=10, line_dash="dot", line_color=NEUTRAL, line_width=1)
            fig.update_xaxes(categoryorder="total descending", tickangle=-60, tickfont=dict(size=9))
            fig.update_yaxes(title="Poids (%)")
            show(fig, 420, "pf_bar")
    left, right = st.columns(2)
    with left:
        section("Part de chaque famille dans le temps", "roles_time")
        roles = a[a["Ticker"] != "CASH"].groupby(["Date", "role"])["weight"].sum().unstack(fill_value=0) * 100
        fig = go.Figure()
        for role in roles:
            fig.add_trace(go.Scatter(x=roles.index, y=roles[role], name=role, stackgroup="one", mode="lines",
                                     line=dict(width=0.5, color=ROLE_COLORS.get(role, NEUTRAL)),
                                     hovertemplate=f"{role} · %{{x|%m/%Y}} · %{{y:.1f}} %<extra></extra>"))
        fig.update_yaxes(title="Part (%)", range=[0, 100])
        show(time_axis(fig), 360, "pf_roles")
    with right:
        section("Poids des 20 actions les plus utilisées", "weights_time")
        piv = a[a["Ticker"] != "CASH"].pivot_table(index="Ticker", columns="Date", values="weight", aggfunc="sum", fill_value=0) * 100
        top = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index[:20]]
        fig = go.Figure(go.Heatmap(z=top.values, x=[month(c) for c in top.columns], y=top.index, xgap=1, ygap=1,
                                   colorscale=[[0, "rgba(59,111,212,0.04)"], [1, ACCENT]], colorbar=dict(title="%"),
                                   hovertemplate="%{y} · %{x} · %{z:.1f} %<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))
        show(fig, 360, "pf_heat")
    section("Positions du mois")
    t = held.sort_values("weight", ascending=False).assign(Poids=lambda x: x["weight"] * 100)[["Ticker", "role", "Poids"]]
    table(t.rename(columns={"Ticker": "Action", "role": "Famille"}), f"portefeuille_{pd.Timestamp(sel):%Y_%m}.csv", {"Poids": F_PCT}, key="pf_dl")
    with st.expander("Sélection des actions candidates ce mois-là"):
        c = r["clusters"]
        c = c[c["Date"] == sel]
        if c.empty or "return_12m" not in c:
            st.caption("Données de sélection indisponibles pour ce mois.")
        else:
            st.caption("Les ~150 actions les plus échangées sont rangées en 4 familles ; les meilleures des familles "
                       "Gagnants dynamiques et Stables de qualité sont retenues (cercles pleins).")
            fig = go.Figure()
            for role, g in c.groupby("role"):
                for sel_flag, sym in [(True, "circle"), (False, "circle-open")]:
                    gg = g[g["selected"] == sel_flag]
                    if gg.empty:
                        continue
                    fig.add_trace(go.Scatter(x=gg["return_12m"] * 100, y=gg["atr_pct"] * 100, mode="markers",
                                             name=f"{role}{'' if sel_flag else ' (non retenue)'}", text=gg["Ticker"],
                                             marker=dict(symbol=sym, size=8, color=ROLE_COLORS.get(role, NEUTRAL)),
                                             hovertemplate="%{text}<br>Hausse mensuelle moyenne %{x:.1f} % · Agitation %{y:.1f} %<extra></extra>"))
            fig.update_xaxes(title="Hausse mensuelle moyenne sur 12 mois (%)")
            fig.update_yaxes(title="Agitation quotidienne du prix (%)")
            show(fig, 420, "pf_clusters")


def page_signal(ctx):
    r = ctx["r"]
    ic = r["ic"][r["signal_cols"]]
    months = ctx["months"]
    icp = ic[ic.index.isin(months)]
    if icp.empty:
        st.info("Aucun mois de signal dans la période choisie.")
        return
    sname = lambda s: SIGNALS.get(s, (s,))[0]  # noqa: E731
    rows = []
    for s in icp.columns:
        a = ic_stats(icp[s])
        rows.append({"Signal": sname(s), "IC moyen": a["mean"], "Mois IC > 0": a["pos"], "Mois": a["n"],
                     "Statistique t": a["t"], "Verdict": ic_verdict(a["t"]), "Principe": SIGNALS.get(s, ("", ""))[1]})
    section(f"Qualité des signaux · {ctx['period_label']}", "signals_table")
    table(pd.DataFrame(rows).sort_values("Statistique t", ascending=False), "qualite_des_signaux.csv",
          {"IC moyen": lambda v: fr(v, 3, True), "Statistique t": F_SDEC}, key="sig_dl")
    years = sorted({str(d.year) for d in ic.index})
    left, right = st.columns(2)
    with left:
        if len(years) > 1:
            section("Statistique t par période", "tstat")
            fig = go.Figure()
            for y, c in zip(years, [NEUTRAL, ACCENT]):
                fig.add_trace(go.Bar(x=[wrap(sname(s), 12) for s in ic.columns], y=[ic_stats(ic[s][ic.index.year == int(y)])["t"] for s in ic.columns],
                                     name=YEAR_LABEL.get(y, y), marker_color=c, hovertemplate="%{x} · t = %{y:.2f}<extra></extra>"))
            for yv in (2, -2):
                fig.add_hline(y=yv, line_dash="dot", line_color=NEUTRAL, line_width=1)
            fig.update_layout(barmode="group")
            fig.update_xaxes(tickangle=0, tickfont=dict(size=10))
            fig.update_yaxes(title="Statistique t")
            show(fig, 380, "sig_t")
        else:
            section("IC cumulé", "ic_cum")
            cum = icp.cumsum()
            fig = go.Figure([go.Scatter(x=cum.index, y=cum[s], name=sname(s), mode="lines+markers", line=dict(color=ACCENT))
                             for s in cum])
            show(time_axis(fig), 380, "sig_cum1")
    with right:
        sig = st.selectbox("Signal détaillé", list(ic.columns), format_func=sname, key="sig_choice")
        section("IC mois par mois", "ic")
        series = icp[sig]
        fig = go.Figure(go.Bar(x=series.index, y=series.values, marker_color=np.where(series.values >= 0, ACCENT, NEUTRAL),
                               hovertemplate="%{x|%m/%Y} · IC %{y:+.2f}<extra></extra>"))
        if len(years) > 1 and months.min() <= pd.Timestamp("2022-12-31") < months.max():
            fig.add_vline(x=pd.Timestamp("2023-01-15").timestamp() * 1000, line_dash="dot", line_color=NEUTRAL, line_width=1)
        fig.update_yaxes(title="IC", range=[-1, 1])
        show(time_axis(fig), 312, "sig_month")
    left, right = st.columns(2)
    with left:
        if len(years) > 1:
            section("IC cumulé de chaque signal", "ic_cum")
            cum = icp.cumsum()
            palette = [ACCENT] + FALLBACK_COLORS
            fig = go.Figure([go.Scatter(x=cum.index, y=cum[s], name=sname(s), mode="lines",
                                        line=dict(color=palette[i % len(palette)], width=2 if s == sig else 1.4),
                                        hovertemplate=f"{sname(s)} · %{{x|%m/%Y}} · %{{y:.2f}}<extra></extra>")
                             for i, s in enumerate(cum)])
            fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
            fig.update_yaxes(title="IC cumulé")
            show(time_axis(fig), 380, "sig_cum")
    with right:
        sg = r["signals"]
        sg = sg[sg.index.get_level_values("Date").isin(months)].dropna(subset=["observed", sig])
        if sig.startswith("lstm") and len(sg):
            section("Prévision et réalité", "pred")
            pts_ = sg.reset_index()
            fig = go.Figure(go.Scatter(x=pts_["observed"] * 100, y=pts_[sig] * 100, mode="markers", text=pts_["Ticker"],
                                       marker=dict(size=6, color=ACCENT, opacity=0.5), name="Actions",
                                       hovertemplate="%{text}<br>Réel %{x:.1f} % · Prévu %{y:.1f} %<extra></extra>"))
            lo = float(min(pts_["observed"].min(), pts_[sig].min()) * 100)
            hi = float(max(pts_["observed"].max(), pts_[sig].max()) * 100)
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dot", color=NEUTRAL, width=1),
                                     name="Prévision parfaite", hoverinfo="skip"))
            fig.update_xaxes(title="Rendement réel du mois suivant (%)", showgrid=True, gridcolor="rgba(128,140,155,0.12)")
            fig.update_yaxes(title="Rendement prévu (%)")
            show(fig, 380, "sig_scatter")
        elif len(years) > 1:
            st.caption("Le nuage prévision / réalité concerne les signaux d'IA ; les autres signaux sont des notes, pas des prévisions.")
    if r["training"] is not None and len(r["training"]):
        with st.expander("Apprentissage du réseau de neurones"):
            th = r["training"]
            if "block" in th:
                c1, c2, c3 = st.columns(3)
                s = c1.selectbox("Signal", sorted(th["signal"].unique()), format_func=sname, key="tr_sig")
                b = c2.selectbox("Réentraînement", sorted(th["block"].unique()), key="tr_block")
                seed = c3.selectbox("Graine", sorted(th["seed"].unique()), key="tr_seed")
                th = th[(th["signal"] == s) & (th["block"] == b) & (th["seed"] == seed)]
            section("Erreur pendant l'apprentissage", "training")
            fig = go.Figure([go.Scatter(x=th["epoch"], y=th[c], name=lbl, mode="lines+markers", line=dict(color=clr))
                             for c, lbl, clr in [("loss", "Entraînement", ACCENT), ("val_loss", "Validation", NEUTRAL)]])
            fig.update_xaxes(title="Époque", dtick=1)
            fig.update_yaxes(title="Erreur (MSE)")
            show(fig, 280, "tr_plot")


def page_method(ctx):
    r = ctx["r"]
    fg, edge = "#7C8798", "#9AA5B4"          # tons neutres lisibles en thème clair et sombre
    section("Étapes du projet")
    st.graphviz_chart(f"""
digraph G {{ rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
 node [shape=box, style="rounded", color="{ACCENT}", fontcolor="{fg}", fontname="Inter, Segoe UI, sans-serif", fontsize=11, penwidth=1.2];
 edge [color="{edge}", arrowsize=0.6];
 a [label="Prix des actions\\nS&P 500, 2015-2023"]; b [label="Indicateurs\\ntendance, risque, facteurs"];
 c [label="Familles d'actions\\nclustering"]; d [label="Sélection\\nGagnants et Stables"];
 e [label="Signal\\nIA, tendance, ..."]; f [label="Répartition\\nde l'argent"]; g [label="Simulation\\nmois par mois"];
 h [label="Comparaison\\nau marché (SPY)"];
 a -> b -> c -> d -> e -> f -> g -> h;
}}""", width="stretch")
    st.caption("À chaque date, seules les informations disponibles à cette date sont utilisées ; le portefeuille est conservé un mois.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            section("Ce qu'on a appris")
            st.markdown(
                "- Le réseau de neurones ne classe pas les actions mieux que le hasard : sur 19 mois, son IC est légèrement négatif.\n"
                "- Un bon résultat sur quelques mois peut être de la chance : selon le tirage aléatoire de départ, le Sharpe du même modèle allait de 0,02 à 1,52.\n"
                "- Ce qui fonctionne en 2022 ne fonctionne pas forcément en 2023 : les classements s'inversent.\n"
                "- Concentrer l'argent sur des prévisions incertaines amplifie les pertes ; la répartition égale est plus robuste.\n"
                "- Réglage par défaut : répartition égale 50/50, aucune alternative n'ayant été confirmée.")
    with right:
        with st.container(border=True):
            section("Limites")
            st.markdown(
                "- Biais de survie : liste actuelle du S&P 500 ; les entreprises sorties de l'indice manquent.\n"
                "- Période courte : 19 mois de test au plus (6 mois pour les anciens résultats).\n"
                "- Pas de coûts de transaction : les frais réduiraient les performances.\n"
                "- Pas de taux sans risque dans le Sharpe ; facteurs Fama-French supposés connus avec deux mois de retard.")
    with st.expander("Glossaire"):
        g = pd.DataFrame(list(GLOSSARY.items()), columns=["Terme", "Définition"])
        st.dataframe(g, hide_index=True, width="stretch")
    with st.expander("Paramètres techniques de ce dossier"):
        table(pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in r["params"].items()]), "parametres.csv", key="m_params")
        if r["meta"].get("schedule"):
            section("Calendrier des réentraînements")
            table(pd.DataFrame(r["meta"]["schedule"]).astype(str), "calendrier.csv", key="m_sched")
    with st.expander("Télécharger les résultats bruts"):
        files = [f for f in ["daily_returns.csv", "metrics.csv", "allocations.csv", "parameters.json"] if (r["path"] / f).exists()]
        for c, f in zip(st.columns(len(files)), files):
            c.download_button(f, (r["path"] / f).read_bytes(), file_name=f, key=f"raw_{f}",
                              mime="application/json" if f.endswith("json") else "text/csv")


PAGES = {"Vue d'ensemble": page_overview, "Simulateur": page_simulator, "Comparaison": page_comparison,
         "Par période": page_periods, "Portefeuille": page_portfolio, "Modèle et signal": page_signal,
         "Méthode et conclusions": page_method}
SLUGS = {"apercu": "Vue d'ensemble", "simulateur": "Simulateur", "comparaison": "Comparaison", "periodes": "Par période",
         "portefeuille": "Portefeuille", "signal": "Modèle et signal", "methode": "Méthode et conclusions"}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Recherche de portefeuille", layout="wide")
    css = ROOT / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    valid, failures = list_results()
    with st.sidebar:
        st.markdown("#### Filtres")
        if valid:
            def label(p):
                r = load_result(str(p), (p / "parameters.json").stat().st_mtime_ns)
                mode = "walk-forward" if r["mode"] == "walk_forward" else "ancienne méthode"
                return f"{p.name[6:8]}/{p.name[4:6]}/{p.name[:4]} {p.name[9:11]}:{p.name[11:13]} · {mode}"
            folder = st.selectbox("Dossier de résultats", [p.name for p in valid], key="folder",
                                  format_func=lambda n: label(RESULTS_DIR / n) if (RESULTS_DIR / n / "parameters.json").exists() else n,
                                  help="Chaque lancement du pipeline crée un dossier ; le plus récent est en haut.")
        if failures:
            with st.expander("Dossiers illisibles"):
                st.caption("\n".join(failures))
    if not valid:
        st.markdown('<div class="pf-header"><span class="pf-title">Recherche de portefeuille</span></div>', unsafe_allow_html=True)
        st.info("Aucun résultat complet. Lancez d'abord : python pipeline.py")
        st.stop()
    path = RESULTS_DIR / folder
    r = load_result(str(path), (path / "parameters.json").stat().st_mtime_ns)
    keys = list(r["returns"].columns)
    with st.sidebar:
        default = [k for k in keys if k in (r["main"], "equal_weight_50_50", "min_variance_50_50", "lstm_raw_max_sharpe_50_50",
                                            "momentum_tilt_50_50", "Baseline_EqualWeight", "Strategy_LSTM_Opt", "SPY")]
        shown = st.multiselect("Stratégies", keys, default=default, key=f"shown_{folder}",
                               format_func=lambda k: r["strategies"][k]["name"] if k in r["strategies"] else k,
                               help="Stratégies comparées sur toutes les pages. Le marché (SPY) reste toujours affiché.")
        if not shown:
            shown = [r["main"]]
        if "SPY" in keys and "SPY" not in shown:
            shown = shown + ["SPY"]
        years = [y for y in ["2022", "2023"] if (r["day_year"] == y).any()]
        options = ["Toute la période"] + [YEAR_LABEL[y] if len(years) > 1 else y for y in years] + ["Plage de dates"]
        period = st.radio("Période", options, key=f"period_{folder}", help=HELP["periods"])
        ret = r["returns"]
        if period == "Plage de dates":
            rng = st.date_input("Du … au …", value=(ret.index.min().date(), ret.index.max().date()),
                                min_value=ret.index.min().date(), max_value=ret.index.max().date(), format="DD/MM/YYYY", key=f"range_{folder}")
            if isinstance(rng, (tuple, list)) and len(rng) == 2:
                ret = ret.loc[str(rng[0]):str(rng[1])]
        elif period != "Toute la période":
            ret = ret[r["day_year"] == period[:4]]
        amount = st.number_input("Montant investi (€)", min_value=100, max_value=10_000_000, value=10_000, step=1_000, key="amount",
                                 help="Somme placée au début de la période.")
        st.divider()
        st.caption("Thème clair ou sombre : menu ⋮ > Settings. Simulation sur données passées, sans frais ; pas un conseil d'investissement.")
    mode = "Walk-forward, réentraînement semestriel" if r["mode"] == "walk_forward" else "Ancienne méthode, découpage unique"
    st.markdown(f'<div class="pf-header"><span class="pf-title">Recherche de portefeuille · S&amp;P 500</span>'
                f'<span class="pf-meta">{mode} · test du {day(r["returns"].index.min())} au {day(r["returns"].index.max())} · '
                f'{len(keys)} stratégies</span></div>', unsafe_allow_html=True)
    if len(ret) < 5:
        st.warning("La période choisie contient trop peu de jours de bourse.")
        st.stop()
    in_range = [p["Date"] for _, p in r["periods"].iterrows()
                if ((ret.index > p["entry_date"]) & (ret.index <= p["exit_date"])).any()]
    ctx = {"r": r, "returns": ret, "shown": [k for k in keys if k in shown], "amount": float(amount),
           "period_label": period, "months": pd.DatetimeIndex(in_range)}
    requested = SLUGS.get(st.query_params.get("page", ""), None)
    pages = list(PAGES)
    with st.container(key="nav"):
        page = st.radio("Page", pages, index=pages.index(requested) if requested else 0, horizontal=True, key="page",
                        label_visibility="collapsed")
    st.caption(f"{ctx['period_label']} · {day(ret.index.min())} – {day(ret.index.max())} · {len(ret)} jours de bourse · "
               f"montant de départ {eur(ctx['amount'])}")
    PAGES[page](ctx)


if __name__ == "__main__":
    main()
