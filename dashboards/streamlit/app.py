"""Pre-launch checker (DATA 230 Group 6, Streamlit dashboard, Keerat Khandpur).

A developer describes a planned game; the app scores it with the team's final
XGBoost model (results/ml/xgb_model.json), says how that score ranks against
real 2022+ games and how often real games with similar scores reached 50K
owners, explains the score with SHAP, and shows what changing one lever does.

Run from the repo root:  streamlit run dashboards/streamlit/app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Works on older Streamlit (e.g. Anaconda's) and newer: width="stretch" only exists in newer versions
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "results" / "ml"
HERE = Path(__file__).resolve().parent

UP, DOWN, GREY, INK = "#2a78d6", "#eb6834", "#8a8a8a", "#222222"
WINDOW = 1000            # "games scored like this" = the 1,000 test games nearest in score

st.set_page_config(page_title="Pre-launch checker", page_icon="🎮", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(ML / "xgb_model.json")
    return model


@st.cache_data
def load_tables():
    meta = json.loads((ML / "model_features.json").read_text())
    typical = json.loads((HERE / "typical_values.json").read_text())
    tp = pd.read_csv(ML / "test_predictions.csv", usecols=["prob_xgb", "actual_50k", "genres", "Price"])
    metrics = pd.read_csv(ML / "model_metrics.csv").set_index("Model")
    # Score -> rank and real hit rate, from the 47,596 held-out 2022+ games
    srt = tp.sort_values("prob_xgb").reset_index(drop=True)
    roll = srt["actual_50k"].rolling(WINDOW, center=True, min_periods=WINDOW // 2).mean()
    return meta, typical, tp, metrics, srt["prob_xgb"].to_numpy(), roll.to_numpy()


model = load_model()
meta, typical, tp, metrics, SORTED, ROLL = load_tables()
FEATURES, LABELS = meta["features"], meta["labels"]
T = typical["typical"]
BASE_RATE = float(tp["actual_50k"].mean())
GENRES = [f[6:] for f in FEATURES if f.startswith("genre_")]
CATS = [f[4:] for f in FEATURES if f.startswith("cat_")]


def rank_of(score):
    """Share of 2022+ games scoring below this one, and the real hit rate of games scored like it."""
    i = int(np.searchsorted(SORTED, score))
    i = min(max(i, 0), len(SORTED) - 1)
    return i / len(SORTED), float(ROLL[i])


# ---------------------------------------------------------------- inputs
st.sidebar.header("Describe the planned game")
st.sidebar.caption("Anything you leave alone is set to what a typical 2022+ Steam game has.")

with st.sidebar:
    genres = st.multiselect("Genres", GENRES, default=["Action", "Indie"])
    free = st.checkbox("Free to play", value=False)
    price = 0.0 if free else st.select_slider("Price (US $)", options=[0.99, 1.99, 2.99, 4.99, 7.99, 9.99, 12.99, 14.99, 19.99, 24.99, 29.99, 39.99, 49.99, 59.99, 69.99], value=14.99, format_func=lambda p: f"${p:.2f}")
    langs = st.slider("Supported languages", 1, 30, 5,
                      help="Interface or subtitle languages listed on the store page.")
    audio = st.slider("Fully voiced languages", 0, 30, 0)

    with st.expander("Publisher and developer"):
        self_pub = st.checkbox("Self-published (developer = publisher)", value=True)
        if self_pub:
            pub_prior = st.number_input("Your earlier Steam releases", 0, 500, 0)
            dev_prior = pub_prior
        else:
            pub_prior = st.number_input("Publisher's earlier Steam releases", 0, 500, 0)
            dev_prior = st.number_input("Developer's earlier Steam releases", 0, 500, 0)

    with st.expander("Steam features and platforms"):
        default_cats = ["Single-player", "Family Sharing", "Steam Achievements", "Steam Cloud"]
        cats = st.multiselect("Store categories", CATS, default=default_cats)
        achievements = st.number_input("Number of achievements", 0, 102, 20,
                                       help="Capped at 102, as in the cleaned data.")
        mac = st.checkbox("Mac", value=False)
        linux = st.checkbox("Linux", value=False)
        early = st.checkbox("Launches in Early Access", value=False)

    with st.expander("Store page and release"):
        words = st.number_input("Words in the store description", 0, 5000, int(round(np.expm1(T["log_about_words"]))))
        website = st.checkbox("Has a website", value=False)
        support_url = st.checkbox("Has a support page", value=False)
        mature = st.checkbox("Mature content notes", value=False)
        age = st.selectbox("Required age", [0, 13, 16, 17, 18], index=0)
        month = st.selectbox("Release month", list(range(1, 13)), index=int(T["release_month"]) - 1,
                             format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))


def build_row(price=price, langs=langs, pub_prior=pub_prior):
    """One row of the 88 model features, in the same units as 04_modeling.ipynb."""
    r = dict(T)
    is_free = price == 0
    dev = pub_prior if self_pub else dev_prior
    r.update({
        "log_price": np.log1p(price), "is_free": float(is_free), "price_outlier": float(price > 100),
        "Required age": float(age), "is_mature": float(mature), "has_content_notes": float(mature),
        "Windows": 1.0, "Mac": float(mac), "Linux": float(linux), "n_platforms": 1.0 + mac + linux,
        "log_languages": np.log1p(langs), "log_audio_langs": np.log1p(min(audio, langs)),
        "supports_english": 1.0, "log_achievements": np.log1p(achievements),
        "log_about_words": np.log1p(words), "release_month": float(month),
        "is_early_access": float(early), "self_published": float(self_pub),
        "log_pub_prior": np.log1p(pub_prior), "log_dev_prior": np.log1p(dev),
        "n_genres": float(len(genres)), "n_categories": float(len(cats)),
        "has_website": float(website), "has_support_url": float(support_url),
    })
    for g in GENRES:
        r[f"genre_{g}"] = float(g in genres or (g == "Free To Play" and is_free))
    for c in CATS:
        r[f"cat_{c}"] = float(c in cats)
    return r


X = pd.DataFrame([build_row()])[FEATURES].astype(float)
score = float(model.predict_proba(X)[:, 1][0])
pct, hit = rank_of(score)
top = max(1, round(100 * (1 - pct)))

# SHAP: exact TreeSHAP contributions (log-odds), grouped into levers a developer controls
contrib = model.get_booster().predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[0][:-1]
GROUPS = {
    "Languages": ["log_languages", "log_audio_langs", "supports_english"],
    "Price": ["log_price", "is_free", "price_outlier", "genre_Free To Play"],
    "Publisher track record": ["log_pub_prior", "log_dev_prior", "self_published"],
    "Genres": [f for f in FEATURES if f.startswith("genre_") and f != "genre_Free To Play"] + ["n_genres"],
    "Steam features": [f for f in FEATURES if f.startswith("cat_")] + ["n_categories", "log_achievements"],
    "Platforms": ["Windows", "Mac", "Linux", "n_platforms"],
    "Store page": ["log_about_words", "has_website", "has_support_url", "has_support_email"],
    "Age rating / mature": ["Required age", "is_mature", "has_content_notes"],
    "Release timing": ["release_month", "release_dayofweek"],
    "Early Access": ["is_early_access"],
}
idx = {f: i for i, f in enumerate(FEATURES)}
shap = pd.Series({g: float(sum(contrib[idx[f]] for f in fs)) for g, fs in GROUPS.items()})
shap_feat = pd.Series(contrib, index=FEATURES)

# ---------------------------------------------------------------- header + KPIs
st.title("Pre-launch checker")
st.markdown("**For a planned Steam game, how likely is 50K+ owners, and which choices move it?** "
            "Across our model, languages and the publisher's track record move the odds most; price barely does.")

k1, k2, k3 = st.columns(3)
k1.metric("Score rank among 2022+ games", f"Top {top}%",
          help="Where this game's model score falls among the 47,596 games released 2022 or later (held-out test set).")
k2.metric("Similar games that hit 50K", f"{hit:.0%}",
          help=f"Share of the {WINDOW:,} test games with the nearest scores that actually reached 50K owners. "
               "This is the honest 'chance': the raw model score is inflated by class weighting.")
best = shap.idxmax() if shap.max() > 0 else None
worst = shap.idxmin() if shap.min() < 0 else None
k3.metric("Biggest boost", f"{best or 'none'}")
k1.caption(f"Ranked against {len(tp):,} real games released 2022 or later")
k2.caption(f"{hit / BASE_RATE:.1f}× the {BASE_RATE:.0%} average for 2022+ games")
k3.caption(f"Biggest drag: {worst or 'none'}")

# ---------------------------------------------------------------- charts
left, right = st.columns(2, gap="large")

with left:
    s = shap.sort_values()
    s = s[s.abs() > 0.005]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h",
        marker_color=[UP if v > 0 else DOWN for v in s.values],
        text=[("raises" if v > 0 else "lowers") for v in s.values], textposition="outside",
        hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
    lim = max(0.3, float(s.abs().max()) * 1.6) if len(s) else 0.3
    fig.update_layout(
        title=dict(text=f"Why: <b>{best or 'nothing'}</b> lifts this game most", x=0, font_size=16),
        xaxis=dict(title="← lowers the score · raises it →  (SHAP, log-odds)",
                   range=[-lim, lim], zeroline=True, zerolinecolor=INK, showgrid=False),
        yaxis=dict(automargin=True), height=380, margin=dict(l=10, r=10, t=50, b=50),
        plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig, **STRETCH)
    st.caption("Each bar compares this game with the average 2022+ game, grouped into choices a developer controls.")
    with st.expander("Which single settings mattered most"):
        tops = shap_feat.reindex(shap_feat.abs().sort_values(ascending=False).index).head(6)
        st.dataframe(pd.DataFrame({"Setting": [LABELS.get(f, f) for f in tops.index],
                                   "Effect": ["raises" if v > 0 else "lowers" for v in tops.values],
                                   "SHAP (log-odds)": tops.round(3).values}),
                     hide_index=True, **STRETCH)

with right:
    lever = st.radio("What if I change…", ["Languages", "Price", "Publisher's earlier releases"], horizontal=True)
    if lever == "Languages":
        grid, cur, xt = list(range(1, 31)), langs, "Supported languages"
        rows = [build_row(langs=v) for v in grid]
    elif lever == "Price":
        grid = [0, 0.99, 2.99, 4.99, 7.99, 9.99, 14.99, 19.99, 24.99, 29.99, 39.99, 49.99, 59.99]
        cur, xt = price, "Price (US $, 0 = free to play)"
        rows = [build_row(price=v) for v in grid]
    else:
        grid, cur, xt = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100], pub_prior, "Publisher's earlier Steam releases"
        rows = [build_row(pub_prior=v) for v in grid]
    sc = model.predict_proba(pd.DataFrame(rows)[FEATURES].astype(float))[:, 1]
    hits = np.array([rank_of(v)[1] for v in sc])
    b = int(np.argmax(hits))
    best_txt = (f"${grid[b]:.2f}" if grid[b] else "free") if lever == "Price" else f"{grid[b]}"
    fig2 = go.Figure()
    fig2.add_hline(y=BASE_RATE, line_dash="dot", line_color=GREY,
                   annotation_text=f"average game {BASE_RATE:.0%}", annotation_position="bottom right")
    fig2.add_trace(go.Scatter(x=grid, y=hits, mode="lines+markers", line=dict(color=UP, width=3),
                              hovertemplate=f"{xt}: %{{x}}<br>games scored like this: %{{y:.0%}} reached 50K<extra></extra>"))
    fig2.add_trace(go.Scatter(x=[cur], y=[hit], mode="markers+text", marker=dict(size=13, color=INK),
                              text=["your game"], textposition="bottom right", hoverinfo="skip"))
    fig2.update_layout(
        title=dict(text=f"What if: best {lever.lower()} here is <b>{best_txt}</b> "
                        f"({hits[b]:.0%} vs {hit:.0%} now)", x=0, font_size=16),
        xaxis=dict(title=xt, showgrid=False),
        yaxis=dict(title="Real games scored like this that reached 50K", tickformat=".0%",
                   rangemode="tozero", gridcolor="#eeeeee"),
        height=380, margin=dict(l=10, r=10, t=50, b=50), plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig2, **STRETCH)
    st.caption("Everything else stays as set on the left. The model learns patterns, not causes: "
               "adding languages helps most when the game can find players in those markets.")

# ---------------------------------------------------------------- trust
with st.expander("How far to trust this"):
    m = metrics.loc["XGBoost"]
    st.markdown(
        f"- **Model:** XGBoost on 88 features known before launch, trained on games up to 2021 and "
        f"tested on {len(tp):,} games from 2022 on (only {BASE_RATE:.1%} of them reached 50K).\n"
        f"- **Ranking is the strength:** PR-AUC {m['PR-AUC']:.2f} vs {metrics.loc['Baseline (random)', 'PR-AUC']:.2f} "
        f"for guessing; the top 10% of scores hit {m['Top-10% hit rate']:.0%} of the time.\n"
        "- **Why no raw percentage:** the model weighted the rare hits up while training, so its raw score overstates "
        "the chance. The app reports rank and the real hit rate of similarly scored games instead.\n"
        "- **Free to play** is over-flagged: it was a strong sign before 2022 but far weaker since, as the market filled up.\n"
        "- The full model comparison, lift chart and feature importance are in the Plotly Dash dashboard.")
