# What Makes a Steam Game Commercially Successful?

DATA 230 (Data Intelligence and Visualization), Fall 2026 · Group 6

**Question:** Can we predict, from information known *before release*, whether a Steam game will become commercially successful, and which factors matter most?

**Success** = Steam's estimated owners reach at least **50,000** (lower bound of the owner range). 14.6% of the cleaned games (14,018 of 95,787) meet it. We also test 20,000 and 100,000 as cutoffs to check that the conclusions hold.

## Team

| Member | Main contributions |
|---|---|
| Keerat Khandpur | Data cleaning and processing, shared definitions, GPU vs CPU comparison, Streamlit app, repository |
| Rachel Huang | Exploratory data analysis, GPU-accelerated EDA (RAPIDS), Plotly Dash app |
| Chinmayi Bolisetty | Dataset introduction, machine learning model and results |
| Calvin Truong | Tableau dashboard, slide deck, report setup |


## Data

- **Source:** [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) by Fronkon Games on Kaggle, built from the Steam Store API and SteamSpy.
- **Size:** 125,855 games × 39 columns raw (40 after fixing the header); 95,787 games after cleaning.
- **Snapshot:** newest release with sales data is March 1, 2026.
- **Download:** the cleaning notebook downloads it automatically with `kagglehub`.
- **Cleaned files** (too large for GitHub): [Google Drive folder](https://drive.google.com/drive/u/2/folders/1z4WvZC3KNyPUId6esfABMI-pNekhWMl0). Small files such as the data dictionary are in [`data/`](data/).

## How to reproduce

1. Install the packages: `pip install -r requirements.txt` (or run the notebooks in Google Colab).
2. Run the notebooks in order:
   1. [`notebooks/01_cleaning.ipynb`](notebooks/): downloads the raw data, cleans it, and writes the files in `clean_output/`.
   2. [`notebooks/02_eda.ipynb`](notebooks/): exploratory analysis on the cleaned data.
   3. [`notebooks/03_eda_gpu_rapids.ipynb`](notebooks/): the same EDA steps on a GPU with RAPIDS cuDF (needs a Colab GPU runtime).
   4. `notebooks/04_modeling.ipynb`: trains the model and saves the predictions and SHAP values the dashboards use.
3. Run the dashboards (after step 2.4):
   - Streamlit: `streamlit run dashboards/streamlit/app.py`
   - Plotly Dash: `python dashboards/dash/app.py`
   - Tableau: open `dashboards/tableau/Group6 Steam Interactive Dashboard.twbx` in Tableau.

## Repository map

| Folder | Contents |
|---|---|
| `notebooks/` | All code, numbered in the order to run it |
| `figures/` | Charts used in the slides and report: `cleaning/`, `eda/`, `ml/` |
| `dashboards/` | `tableau/`, `streamlit/`, `dash/`, each with its own files |
| `data/` | Data dictionary, shared definitions, cleaning funnel (small files only) |
| `presentations/` | Mid and final presentation slides |
| `report/` | Final technical report (IEEE format) |

## Key decisions

- **Shifted header:** the raw CSV header has 39 names but every row has 40 values (`DiscountDLC count` is two columns). We rebuild the header and check that every column holds the right kind of data.
- **Rows kept:** we removed games with no sales data (owners "0 - 0"), non-game software, and games released in the 6 months before the snapshot (too new to have sold).
- **No leakage:** the model uses only information known before release (price, genres, categories, platforms, languages, publisher's *earlier* releases). Reviews, playtime and player counts are outcomes and are never model inputs.
- **Shared definitions:** price groups, publisher experience and the core genre list are defined once in the cleaning notebook ([`data/shared_definitions.json`](data/)), so EDA, dashboards and the model all agree.

## Key results

| Finding | Number |
|---|---|
| Games reaching 50K owners | 14.6% |
| Hit rate, $20.01–40 vs $0.01–5 | 25.2% vs 12.4% |
| Hit rate, publishers with 5–19 earlier games vs none | 22.3% vs 11.2% |
| Best genre hit rate (Massively Multiplayer) vs worst (Casual) | 32.4% vs 9.5% |
| Model performance | _to add_ |
| Top predictors (SHAP) | _to add_ |

## Dashboards

_Screenshots to add: Tableau, Streamlit, Plotly Dash._

## References

- Fronkon Games. *Steam Games Dataset*. Kaggle. https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
- Libraries: pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn, SHAP, RAPIDS cuDF, Streamlit, Dash, Tableau.
- AI tools used are listed in the AI acknowledgement section of the report.
