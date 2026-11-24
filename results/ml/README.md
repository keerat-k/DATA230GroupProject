# ML results: will a Steam game reach 50K+ owners?

Produced by `notebooks/04_modeling.ipynb` (run from the `notebooks/` folder).

## Setup
- Data: `data/clean_output/steam_clean.csv` (about 68 MB, so it is kept in the team Google Drive, not in the repo)
- Extra packages: `pip install xgboost shap scikit-learn pandas matplotlib jinja2` (on a Mac also `brew install libomp`)
- Runtime: about 6 minutes on a laptop CPU

## Model summary
| | |
|---|---|
| Target | `success_50k`: SteamSpy owner lower bound ≥ 50,000 |
| Features | 88 pre-release features: price, platforms, languages, achievements, store page, publisher/developer prior releases, 18 genre and 44 store-category flags |
| Excluded (leakage) | reviews, scores, playtime, CCU, recommendations, DLC, discount, tags, Steam Trading Cards, press quotes, release year |
| Split | train on releases ≤ 2021 (validation 2020–21 for tuning), test on 2022 to Sep 2025 (47,596 games, 7.9% hits) |

| Model (2022+ test) | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Random baseline | 0.079 | 0.500 | – | – | – |
| Logistic Regression | 0.293 | 0.758 | 0.324 | 0.238 | 0.510 |
| Random Forest | 0.346 | 0.774 | 0.365 | 0.292 | 0.487 |
| **XGBoost (best)** | **0.356** | **0.779** | 0.354 | 0.266 | 0.529 |

Top 10% of XGBoost scores: **35% hit rate (4.4× average), containing 44% of all 2022+ hits.**

## Files for the dashboards
| File | Contents | Dashboard idea |
|---|---|---|
| `model_metrics.csv` | metrics per model | model comparison bar chart / KPI cards |
| `test_predictions.csv` | each 2022+ game: `prob_xgb/rf/lr`, `pred_50k`, `actual_50k`, `outcome` (TP/FP/TN/missed), `driver_1..3` (top SHAP reasons) | searchable game table, "why this score" panel, filter by genre/price |
| `feature_importance.csv` | permutation importance + mean \|SHAP\| | feature-importance bar chart |
| `lift_by_decile.csv` | hit rate per score decile | lift chart |
| `genre_actual_vs_predicted.csv` | actual vs flagged rate per genre | genre comparison |
| `xgb_model.json` + `model_features.json` | trained XGBoost + feature order, labels, threshold | **what-if predictor**: sliders for price, languages, etc. |

Loading the model for a what-if predictor:
```python
import json, numpy as np, pandas as pd, xgboost as xgb
meta = json.load(open("results/ml/model_features.json"))
model = xgb.XGBClassifier(); model.load_model("results/ml/xgb_model.json")
row = pd.DataFrame([dict.fromkeys(meta["features"], 0.0)])
row["log_price"] = np.log1p(19.99); row["log_languages"] = np.log1p(10); row["Windows"] = 1; row["n_platforms"] = 1
p = model.predict_proba(row[meta["features"]])[:, 1][0]
print(p, p >= meta["threshold"])   # use as a relative score, not a literal probability
```

## Figures (`figures/ml/`)
1. `01_model_comparison.png`: PR-AUC, ROC-AUC, F1 and recall for the three models against the random baseline
2. `02_pr_roc_curves.png`: precision–recall and ROC curves
3. `03_confusion_matrix.png`: XGBoost at the validation-tuned threshold
4. `04_lift_by_decile.png`: actual hit rate by predicted-score decile
5. `05_permutation_importance.png`: top 15 features
6. `06_shap_summary.png`: SHAP beeswarm (direction and size of each feature's effect)
7. `07_shap_dependence.png`: price, publisher experience, languages, store description
8. `08_genre_actual_vs_predicted.png`: genre check, including the Free-To-Play saturation finding

## Key insights
1. **Localisation is the strongest controllable lever.** The effect peaks at about 10–15 supported languages.
2. **Signals of investment help:** a website, a detailed store page, achievements, cloud saves and multiplayer.
3. **Publisher experience has a sweet spot** (about 15–20 earlier releases). Very prolific "shovelware" publishers and developers are pushed down.
4. **Price is not the lever.** Free or near-free games get a small boost, and above about $5 the effect is flat and slightly negative. The EDA's $20–40 peak reflects who sets higher prices.
5. **Casual and Indie are the most crowded genres and have the lowest odds; MMO, RPG and Strategy have the highest.**
6. **The market is getting harder.** The hit rate fell from 21% (≤ 2021) to 8% (2022+). Free To Play dropped from about 32% (2019–21) to 6% (2024), which is why the model over-flags F2P games.

## Limitations
Owner counts are SteamSpy estimates given as ranges. Recent games have had less time to reach 50K owners. The probabilities are inflated by class weighting, so use them to rank games. Some fields (languages, description) can change after launch. The results show associations, not causes.
