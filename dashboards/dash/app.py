import re
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output

# ---------- Colors (used for emphasis, not decoration) ----------
BLUE = "#2a78d6"        # main / best
ORANGE = "#eb6834"      # highlight
GRAY = "#a3a29d"        # everything else
LIGHT_GRAY = "#d9d8d4"  # baseline
px.defaults.template = "plotly_white"   # white background for every chart

MODEL_COLORS = {"XGBoost": BLUE, "Random Forest": GRAY,
                "Logistic Regression": GRAY, "Baseline (random)": LIGHT_GRAY}
METRIC_OPTIONS = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "Top-10% hit rate"]

# ---------- 1. Find the results/ml folder ----------
# app.py is in dashboards/dash/, so go up 2 folders to reach the repo root
ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "results" / "ml"

# ---------- 2. Load the data ----------
metrics = pd.read_csv(ML / "model_metrics.csv")
importance = pd.read_csv(ML / "feature_importance.csv")
lift = pd.read_csv(ML / "lift_by_decile.csv")
genre = pd.read_csv(ML / "genre_actual_vs_predicted.csv")
preds = pd.read_csv(ML / "test_predictions.csv")

# ---------- Charts ----------

# (a) Model comparison and (b) feature importance are built in the callbacks below,
#     because they change when the user picks a different option.

# (c) Lift chart: top decile in orange, others gray, dashed average line
avg_hit_rate = preds["actual_50k"].mean()   # share of all test games that were hits (~0.079)
lift["group"] = lift["decile"].apply(lambda d: "Top 10% of scores" if d == 1 else "Other deciles")

fig_lift = px.bar(
    lift, x="decile", y="hit_rate", text_auto=".0%",
    title="Hit rate by score decile (XGBoost)",
    labels={"decile": "Score decile (1 = highest scores)", "hit_rate": "Hit rate (50K+ owners)", "group": ""},
    color="group",
    color_discrete_map={"Top 10% of scores": ORANGE, "Other deciles": GRAY},
)
fig_lift.add_hline(
    y=avg_hit_rate, line_dash="dash",
    annotation_text=f"average {avg_hit_rate:.1%}", annotation_position="top right",
)
fig_lift.update_yaxes(tickformat=".0%")
fig_lift.update_xaxes(dtick=1)

# (d) Genre: actual hit rate vs. share flagged by the model
genre_sorted = genre.sort_values("actual_rate", ascending=False)
fig_genre = px.bar(
    genre_sorted, x="genre", y=["actual_rate", "flagged_rate"], barmode="group",
    title="Actual hit rate vs. share flagged by the model, by genre",
    labels={"genre": "", "value": "Share of games", "variable": ""},
    color_discrete_map={"actual_rate": GRAY, "flagged_rate": BLUE},
)
fig_genre.for_each_trace(lambda t: t.update(
    name={"actual_rate": "Actual hit rate", "flagged_rate": "Flagged by model"}[t.name]))
fig_genre.update_yaxes(tickformat=".0%")
fig_genre.update_xaxes(tickangle=-35)   # tilt genre names so they don't overlap

# (e) KPI cards for XGBoost
xgb = metrics[metrics["Model"] == "XGBoost"].iloc[0]   # the XGBoost row
top10 = xgb["Top-10% hit rate"]
top10 = top10 / 100 if top10 > 1 else top10            # works whether saved as 0.35 or 35


def kpi_card(label, value):
    return html.Div([
        html.Div(label, style={"fontSize": "14px", "color": "#555"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "color": BLUE}),
    ], style={"border": "1px solid #ddd", "borderRadius": "8px", "padding": "12px 20px",
              "backgroundColor": "white", "flex": "1"})


kpi_row = html.Div([
    kpi_card("PR-AUC (XGBoost)", f"{xgb['PR-AUC']:.3f}"),
    kpi_card("ROC-AUC", f"{xgb['ROC-AUC']:.3f}"),
    kpi_card("F1", f"{xgb['F1']:.3f}"),
    kpi_card("Top-10% hit rate", f"{top10:.0%}"),
    kpi_card("Average hit rate", f"{avg_hit_rate:.1%}"),
], style={"display": "flex", "gap": "16px", "margin": "16px 0"})

# ---------- Tab 2: data for the game explorer ----------
# Split the "genres" text (e.g. "Action, Indie") into a list so we can filter by one genre
preds["genre_list"] = preds["genres"].fillna("").apply(
    lambda s: [g.strip() for g in re.split(r"[,;|]", str(s)) if g.strip()])
ALL_GENRES = sorted({g for lst in preds["genre_list"] for g in lst})
OUTCOMES = sorted(preds["outcome"].dropna().unique())
PRICE_MAX = int(preds["Price"].quantile(0.99)) + 1        # ignore extreme prices for the slider range

# Rows shown in the table (an "id" column lets us find the clicked game)
preds["id"] = preds["AppID"]
TABLE_COLUMNS = [
    {"name": "Game", "id": "Name"},
    {"name": "Genres", "id": "genres"},
    {"name": "Price", "id": "Price", "type": "numeric", "format": dash_table.FormatTemplate.money(2)},
    {"name": "XGBoost score", "id": "prob_xgb", "type": "numeric",
     "format": dash_table.FormatTemplate.percentage(1)},
    {"name": "Predicted 50K+", "id": "pred_50k"},
    {"name": "Actual 50K+", "id": "actual_50k"},
    {"name": "Outcome", "id": "outcome"},
]
OUTCOME_COLORS = {"True positive": BLUE, "False positive": ORANGE,
                  "True negative": GRAY, "Missed hit": "#c23b3b"}


def filter_games(genre, price_range, outcomes):
    """Return the rows of preds that match the Tab 2 filters."""
    df = preds
    if genre and genre != "All genres":
        df = df[df["genre_list"].apply(lambda lst: genre in lst)]
    lo, hi = price_range
    df = df[(df["Price"] >= lo) & ((df["Price"] <= hi) | (hi >= PRICE_MAX))]  # top of slider = "and above"
    if outcomes:
        df = df[df["outcome"].isin(outcomes)]
    return df


# ---------- 3. Create the app and its layout ----------
app = Dash(__name__, title="Steam 50K+ Owners Dashboard")

CARD = {"backgroundColor": "white", "border": "1px solid #ddd", "borderRadius": "8px", "padding": "8px"}

TAB2 = html.Div([
    # Filters row
    html.Div([
        html.Div([html.Label("Genre"),
                  dcc.Dropdown(id="genre-filter", options=["All genres"] + ALL_GENRES,
                               value="All genres", clearable=False)], style={"flex": "1"}),
        html.Div([html.Label("Price range ($)"),
                  dcc.RangeSlider(id="price-filter", min=0, max=PRICE_MAX, step=1,
                                  value=[0, PRICE_MAX],
                                  marks={0: "$0", PRICE_MAX // 2: f"${PRICE_MAX // 2}",
                                         PRICE_MAX: f"${PRICE_MAX}+"},
                                  tooltip={"placement": "bottom"})], style={"flex": "2"}),
        html.Div([html.Label("Outcome"),
                  dcc.Checklist(id="outcome-filter", options=OUTCOMES, value=OUTCOMES,
                                inline=True, labelStyle={"marginRight": "12px"})],
                 style={"flex": "1.5"}),
    ], style={**CARD, "display": "flex", "gap": "24px", "padding": "14px 18px",
              "marginBottom": "16px", "fontSize": "14px"}),

    html.Div(id="filter-summary", style={"margin": "0 0 12px", "color": "#555"}),

    # Score distribution + "why this score" panel
    html.Div([
        html.Div(dcc.Graph(id="score-hist"), style=CARD),
        html.Div(id="why-panel", style={**CARD, "padding": "16px 20px"}),
    ], style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px",
              "marginBottom": "16px"}),

    # Game table
    html.Div([
        html.P("Click any game in the table to see why it got its score. "
               "Use the boxes under the headers to search.",
               style={"fontSize": "13px", "color": "#555", "margin": "4px 0 8px"}),
        dash_table.DataTable(
            id="games-table", columns=TABLE_COLUMNS,
            page_size=12, sort_action="native", filter_action="native",
            sort_by=[{"column_id": "prob_xgb", "direction": "desc"}],
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "Arial, sans-serif", "fontSize": "13px",
                        "padding": "6px", "textAlign": "left", "maxWidth": "260px",
                        "overflow": "hidden", "textOverflow": "ellipsis"},
            style_header={"backgroundColor": "#f0efec", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"filter_query": f'{{outcome}} = "{o}"', "column_id": "outcome"},
                 "color": c, "fontWeight": "bold"} for o, c in OUTCOME_COLORS.items()
            ] + [{"if": {"state": "active"}, "backgroundColor": "#e7f1fb", "border": f"1px solid {BLUE}"}],
        ),
    ], style=CARD),
])

app.layout = html.Div([
    html.H1("Will a Steam game reach 50K+ owners?", style={"marginBottom": "4px"}),
    html.P(f"Test set: {len(preds):,} games released 2022 to Sep 2025", style={"color": "#555"}),
    dcc.Tabs(value="tab-performance", children=[
        dcc.Tab(label="Model performance", value="tab-performance", children=html.Div([
            kpi_row,
            # 2 x 2 grid of charts
            html.Div([
                html.Div([
                    html.Label("Metric: ", style={"fontSize": "14px"}),
                    dcc.Dropdown(id="metric-dropdown", options=METRIC_OPTIONS, value="PR-AUC",
                                 clearable=False, style={"width": "250px"}),
                    dcc.Graph(id="model-graph"),
                ], style=CARD),
                html.Div(dcc.Graph(figure=fig_lift), style=CARD),
                html.Div([
                    dcc.RadioItems(id="importance-type",
                                   options=[{"label": " Permutation importance", "value": "perm_mean"},
                                            {"label": " Mean |SHAP|", "value": "mean_abs_shap"}],
                                   value="perm_mean", inline=True,
                                   labelStyle={"marginRight": "16px", "fontSize": "14px"}),
                    html.Label("Number of features to show:", style={"fontSize": "14px"}),
                    dcc.Slider(id="top-n", min=5, max=30, step=5, value=15),
                    dcc.Graph(id="importance-graph"),
                ], style=CARD),
                html.Div(dcc.Graph(figure=fig_genre), style=CARD),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
        ], style={"paddingTop": "8px"})),
        dcc.Tab(label="Explore games", value="tab-games", children=html.Div(TAB2, style={"paddingTop": "16px"})),
    ]),
], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f6f6f4", "padding": "20px 30px"})

# ---------- Callbacks ----------
# Callback 1: metric dropdown -> model comparison chart
@app.callback(
    Output("model-graph", "figure"),
    Input("metric-dropdown", "value"),
)
def update_model_chart(metric):
    fig = px.bar(
        metrics, x="Model", y=metric, text_auto=".3f",
        title=f"Model comparison: {metric} on 2022+ games",
        color="Model", color_discrete_map=MODEL_COLORS,
    )
    fig.update_layout(showlegend=False, xaxis_title="")
    return fig


# Callback 2: importance type (radio) + number of features (slider) -> importance chart
IMPORTANCE_NAMES = {"perm_mean": "permutation importance", "mean_abs_shap": "mean |SHAP|"}


@app.callback(
    Output("importance-graph", "figure"),
    Input("importance-type", "value"),
    Input("top-n", "value"),
)
def update_importance(col, n):
    top = importance.nlargest(n, col)
    fig = px.bar(
        top, x=col, y="label", orientation="h",
        title=f"Top {n} features ({IMPORTANCE_NAMES[col]}, XGBoost)",
        labels={col: "Importance", "label": ""},
        color_discrete_sequence=[BLUE],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(400, 22 * n + 120))   # taller when showing more features
    return fig


# Callback 3 (Tab 2): filters -> table rows, score histogram, summary text
@app.callback(
    Output("games-table", "data"),
    Output("score-hist", "figure"),
    Output("filter-summary", "children"),
    Input("genre-filter", "value"),
    Input("price-filter", "value"),
    Input("outcome-filter", "value"),
)
def update_explorer(genre, price_range, outcomes):
    df = filter_games(genre, price_range, outcomes)

    fig = px.histogram(
        df, x="prob_xgb", color="outcome", nbins=40,
        color_discrete_map=OUTCOME_COLORS,
        title="Distribution of XGBoost scores for the filtered games",
        labels={"prob_xgb": "XGBoost score (predicted chance of 50K+ owners)", "outcome": "Outcome"},
    )
    fig.update_layout(bargap=0.05, yaxis_title="Number of games")
    fig.update_xaxes(tickformat=".0%")

    hits = int(df["actual_50k"].sum())
    hit_rate = hits / len(df) if len(df) else 0
    summary = (f"{len(df):,} games match the filters · {hits:,} actually reached 50K+ owners "
               f"({hit_rate:.1%}; overall average {avg_hit_rate:.1%})")

    rows = df.sort_values("prob_xgb", ascending=False)[["id"] + [c["id"] for c in TABLE_COLUMNS]]
    return rows.to_dict("records"), fig, summary


# Callback 4 (Tab 2): click a game in the table -> "why this score" panel
@app.callback(
    Output("why-panel", "children"),
    Input("games-table", "active_cell"),
)
def show_why(active_cell):
    if not active_cell:
        return [html.H3("Why this score?", style={"marginTop": 0}),
                html.P("Click a game in the table below to see its score and the top 3 "
                       "reasons (SHAP drivers) behind it.", style={"color": "#555"})]
    game = preds.loc[preds["id"] == active_cell["row_id"]].iloc[0]
    actual = "Yes" if game["actual_50k"] == 1 else "No"
    predicted = "Yes" if game["pred_50k"] == 1 else "No"
    return [
        html.H3(game["Name"], style={"marginTop": 0, "marginBottom": "4px"}),
        html.P(f"{game['genres']} · ${game['Price']:.2f} · released {game['release_date']}",
               style={"color": "#555", "fontSize": "13px", "marginTop": 0}),
        html.Div(f"{game['prob_xgb']:.0%}", style={"fontSize": "40px", "fontWeight": "bold", "color": BLUE}),
        html.Div("XGBoost score (chance of 50K+ owners)", style={"fontSize": "13px", "color": "#555"}),
        html.P([f"Predicted 50K+: {predicted} · Actual 50K+: {actual} · Outcome: ",
                html.B(game["outcome"], style={"color": OUTCOME_COLORS.get(game["outcome"], "#333")})],
               style={"fontSize": "14px"}),
        html.H4("Top 3 reasons for this score", style={"marginBottom": "6px"}),
        html.Ol([html.Li(str(game[d])) for d in ["driver_1", "driver_2", "driver_3"]
                 if pd.notna(game[d])], style={"fontSize": "14px", "paddingLeft": "20px"}),
    ]


# ---------- 4. Run it ----------
if __name__ == "__main__":
    app.run(debug=True)