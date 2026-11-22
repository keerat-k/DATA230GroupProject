# Data

The raw data is downloaded by `notebooks/01_cleaning.ipynb` (Kaggle: Steam Games Dataset by Fronkon Games). The cleaned files are too large for GitHub and are in the shared Google Drive folder (link in the main README).

| File | Where | Contents |
|---|---|---|
| `steam_clean.csv` (68 MB) | Drive | One row per cleaned game, readable columns |
| `steam_model.parquet` | Drive | Pre-release features, outcomes and targets for the model |
| `steam_genres_long.csv`, `steam_tags_long.csv` | Drive | One row per game and genre/tag |
| `data_dictionary.csv` | here | Every model column, its type and role (feature, outcome, target) |
| `shared_definitions.json` | here | Price groups, publisher experience groups, core genre list |
| `cleaning_funnel.csv` | here | Rows removed at each cleaning step |
