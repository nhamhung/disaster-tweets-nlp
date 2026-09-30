# Disaster Tweets NLP

A worked, end-to-end data science project predicting whether a tweet
describes a **real disaster** (`"Disaster"`) or **not** (`"Not
Disaster"`) — the classic "Real or Not? NLP with Disaster Tweets" task.

## Why this dataset, and why not the Kaggle competition directly

The original Kaggle *competition* (`nlp-getting-started`) requires
accepting its rules on the website before the API will serve the data —
verified directly: `kaggle competitions download -c nlp-getting-started`
returns a 403 even when authenticated. This project instead uses a
public *dataset* mirror of the same task, `vstepanenko/disaster-tweets`
— the same 4 input columns (`keyword`, `location`, `text`, `target`), a
larger, re-scraped sample (11,370 rows).

## What's here

| Deliverable | Where |
|---|---|
| A well-documented notebook building the model, applying good practices | `notebooks/01_eda_and_modeling.ipynb` |
| A multi-page Streamlit app to try the model on any tweet you type | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| A research-style writeup | `report/report.qmd` |
| A script that trains and saves the production model | `scripts/train.py` |

All four share one feature-engineering/model source of truth in
`src/disaster_tweets_nlp/`, so the notebook, the app, and the script
can never quietly drift apart — they all load the same trained
`models/model.joblib`.

## The central data-quality decisions this project is built around

- **Exact-duplicate tweets, some with conflicting labels**: 147
  duplicate tweet texts, 2 of which carry *different* labels for the
  same text. Left alone, duplicates could leak across a
  train/validation split; conflicting ones can't be trusted either
  way. `data.split_features_target` drops the conflicting rows and
  deduplicates the rest before any modeling.
- **`keyword` contains raw URL-encoding artifacts** (e.g.
  `"mass%20murder"`) — decoded before use.
- **`location` is too sparse and inconsistent to use as a categorical
  feature** (4,504 distinct free-text values across 7,952 non-null
  rows) — reduced to a single `has_location` flag instead.

See `config.py`'s module docstring for the full detail.

## A model chosen for more than its cross-validation score

The single highest-scoring model measured during development was an
*uncapped*-depth Random Forest (~0.79 macro-F1) — but its trees reached
an average depth of ~330 (mostly memorizing `RandomOverSampler`'s
exact-duplicate minority rows), which made `shap.TreeExplainer` take
over 2 minutes to compute values for just 30 rows. The model this
project actually saves and serves is `LightGBM` + `RandomOverSampler`
instead: slightly lower macro-F1 (~0.78) but SHAP-explainable in
~0.5s for 300 rows. See `model.py`'s and `report/report.qmd`'s "Model
comparison" section for the full reasoning — choosing accuracy *and*
explainability, not accuracy alone, is this project's main
methodological point.

Class-imbalance handling is compared honestly too, the same as this
portfolio's other projects — but with the opposite winner:
`RandomOverSampler` beats `class_weight="balanced"` here, whereas
class-weighting wins on this portfolio's tabular projects. ADASYN isn't
used at all for this project — a k-nearest-neighbors search is a poor
fit for TF-IDF's thousands of sparse, mostly-zero dimensions.

## What this project deliberately doesn't do

This is tweet-topic classification, not disaster-response triage: no
real-time stream processing, no geolocation (only a `has_location`
flag), no truth/severity verification, and no event clustering.
`report/report.qmd`'s Discussion section covers what a real
disaster-response monitoring system would need beyond this.

## Get the data

[Kaggle API](https://www.kaggle.com/docs/api) (`pip install kaggle`,
then put your `kaggle.json`/access token in `~/.kaggle/`):

```bash
kaggle datasets download -d vstepanenko/disaster-tweets -p data/raw
unzip -o data/raw/disaster-tweets.zip -d data/raw
```

This produces `data/raw/tweets.csv`.

## Run the notebook

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Train from the command line

```bash
python scripts/train.py                    # default: LightGBM + RandomOverSampler — the measured winner
python scripts/train.py --model "Random Forest"
python scripts/train.py --no-resample       # class-weighting/plain training instead, for comparison
python scripts/train.py --help
```

## Run the app

A 4-page app: **Predict** (type or paste any tweet, optionally add a
keyword/location, see the predicted label), **Dataset Overview**,
**Feature Engineering**, and **Model Insights** (model comparison +
class-imbalance comparison + decision-threshold tuning + live SHAP).

```bash
streamlit run app/streamlit_app.py
```

Or in Docker:

```bash
docker build -t disaster-tweets-nlp-app -f app/Dockerfile .
docker run -p 8501:8501 disaster-tweets-nlp-app
```

Then open http://localhost:8501.

## Render the research writeup

```bash
quarto render report/report.qmd
```

## Run the tests

```bash
pytest tests/
```

These test the feature engineering, duplicate/conflicting-label
handling, and model logic directly with synthetic data — no download
needed, and they already pass without any real data.

## Deploy

The Streamlit app starts immediately from a bundled 1,000-row stratified sample sourced from Kaggle. Set `USE_FULL_KAGGLE_DATA=true` to fetch and use the complete dataset through the Kaggle API; configure either `KAGGLE_API_TOKEN` or a `[kaggle]` secrets section containing username and key.

- Repository: <https://github.com/nhamhhung/disaster-tweets-nlp>
- Report: <https://nhamhung.github.io/disaster-tweets-nlp/>
- Streamlit: <https://disaster-tweets-nlp.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
