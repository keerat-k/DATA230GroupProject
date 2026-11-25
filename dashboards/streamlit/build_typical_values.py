"""Rebuild typical_values.json for the Streamlit app.

The app fills in any setting the user does not touch with the value a typical
recent (2022+) Steam game has. This script recomputes those values with the
same feature steps as notebooks/04_modeling.ipynb.

Run from the repo root:
    python dashboards/streamlit/build_typical_values.py data/clean_output/steam_clean.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
meta = json.loads((ROOT / "results/ml/model_features.json").read_text())
FEATURES = meta["features"]

src = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/clean_output/steam_clean.csv"
df = pd.read_csv(src, low_memory=False)
df["release_date"] = pd.to_datetime(df["release_date"])
df = df.sort_values(["release_date", "AppID"]).reset_index(drop=True)

# Same steps as 04_modeling.ipynb
for key, new in [("publisher", "pub_prior_games"), ("developer", "dev_prior_games")]:
    df[new] = (df.groupby(df[key].fillna("Unknown"))["release_date"].rank(method="min") - 1).astype(int)
for out, raw in [("log_price", "price_capped"), ("log_achievements", "achievements_capped"),
                 ("log_about_words", "about_word_count"), ("log_languages", "n_languages"),
                 ("log_audio_langs", "n_audio_languages"), ("log_pub_prior", "pub_prior_games"),
                 ("log_dev_prior", "dev_prior_games")]:
    df[out] = np.log1p(df[raw])
multi = {}
for col, prefix in [("genres", "genre_"), ("categories", "cat_")]:
    items = df[col].fillna("").str.split(",").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    for f in [f for f in FEATURES if f.startswith(prefix)]:
        multi[f] = items.apply(lambda xs, k=f[len(prefix):]: int(k in xs))
df = pd.concat([df, pd.DataFrame(multi)], axis=1)
df["n_categories"] = df["n_categories"] - df["categories"].fillna("").str.contains(
    "Steam Trading Cards", regex=False).astype(int)

recent = df[df["release_date"].dt.year >= 2022][FEATURES].astype(float)
typical = {}
for f in FEATURES:
    s = recent[f]
    if set(s.dropna().unique()) <= {0.0, 1.0}:
        typical[f] = float(s.mean() >= 0.5)           # yes/no: what most recent games do
    else:
        typical[f] = float(s.median())                # numbers: the median recent game
share = {f: round(float(recent[f].mean()), 4) for f in FEATURES if f.startswith(("genre_", "cat_"))}

out = {"note": "Median (numbers) or majority (yes/no) of 2022+ games, in model units (log_* = log1p).",
       "n_games": int(len(recent)), "typical": typical, "share_of_games": share}
(HERE / "typical_values.json").write_text(json.dumps(out, indent=1))
print("wrote", HERE / "typical_values.json", "from", len(recent), "games")
