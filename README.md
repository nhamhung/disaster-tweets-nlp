# Disaster Tweets NLP

A worked, end-to-end data science project predicting whether a tweet
describes a **real disaster** (`"Disaster"`) or **not** (`"Not
Disaster"`) — the classic "Real or Not? NLP with Disaster Tweets" task.

**Kaggle leaderboard: public F1 0.84339, rank ~32 of 438** — up from
0.67269 (rank ~416) with this project's first submission. See
[Climbing the leaderboard](#climbing-the-leaderboard) for every step,
including the ones that didn't help.

## Two models, two jobs

This project ships two models, deliberately kept separate:

| | Explainable model | Leaderboard model |
|---|---|---|
| What | TF-IDF + LightGBM + `RandomOverSampler` | 5-fold ensemble of `cardiffnlp/twitter-roberta-base` |
| Trained on | `vstepanenko/disaster-tweets`, a public *dataset* mirror (11,370 rows) | The competition's own `train.csv` (7,613 rows) |
| Used by | Notebook, Streamlit app, SHAP, `scripts/train.py` | `scripts/make_transformer_submission.py` |
| Kaggle public F1 | 0.67269 | **0.84339** |
| Why keep it | Trains in seconds; SHAP explains predictions in readable words; app/Docker stay light | Best score — but needs `torch` (~1GB), ~30 min to train, and its features are 768 anonymous dimensions SHAP can't make readable |

The mirror exists because the original Kaggle *competition*
(`nlp-getting-started`) requires accepting its rules in a browser before
the API serves anything (verified: `kaggle competitions download`
returns a 403 until you have). Using it for the explainable model keeps
that model reproducible without a competition account.

## Prerequisites

Install once, before Setup below:

| Dependency | Why | Install |
|---|---|---|
| **Python 3.12** | This project's `.venv` is built against 3.12 — a different version may resolve incompatible package versions from `requirements.txt`. | [python.org/downloads](https://www.python.org/downloads/) or a version manager (e.g. `pyenv install 3.12`) |
| **Quarto** | Renders `report/report.qmd` — a standalone binary, not a Python package, so `pip install` never gets it. | [quarto.org/docs/get-started](https://quarto.org/docs/get-started/) |
| **Kaggle API token** | Needed only for the complete dataset; the default Streamlit app uses a bundled sample, and `pytest` uses synthetic data. | Kaggle account → **Account → Create New API Token** → save the downloaded file as `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows). See the [Kaggle API docs](https://www.kaggle.com/docs/api). |
| **Docker** (optional) | Only if you want to run the app in its pre-baked container instead of `streamlit run`. | [docker.com/get-started](https://www.docker.com/get-started/) |

## What's here

| Deliverable | Where |
|---|---|
| A well-documented notebook building the model, applying good practices | `notebooks/01_eda_and_modeling.ipynb` |
| A multi-page Streamlit app to try the model on any tweet you type | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| A research-style writeup | `report/report.qmd` |
| A script that trains and saves the explainable model | `scripts/train.py` |
| Kaggle submission scripts — explainable model, and the leaderboard model | `scripts/make_submission.py`, `scripts/make_transformer_submission.py` |

All of these share one source of truth in `src/disaster_tweets_nlp/`,
so the notebook, the app, and the scripts can never quietly drift
apart. The notebook, app, and `make_submission.py` all load the same
trained `models/model.joblib`.

## Project layout

```
data/raw/                   # downloaded Kaggle CSVs (gitignored — see below)
data/processed/              # cached intermediate data, incl. transformer fold predictions (gitignored)
notebooks/                   # the main EDA + modeling notebook (explainable model)
src/disaster_tweets_nlp/     # shared config, data loading, feature engineering, model code, SHAP,
                             #   and transformer.py (the leaderboard model)
models/                      # trained explainable-model artifact (model.joblib)
app/                         # Streamlit app (multi-page, app/pages_src/) + Dockerfile
scripts/                     # train.py, make_submission.py, make_transformer_submission.py
report/                      # Quarto research writeup
tests/                       # pytest tests (synthetic data — no download needed)
requirements-transformer.txt # extra deps for the leaderboard model only (torch, transformers)
```

## Setup

```bash
python3.12 -m venv .venv          # use the 3.12 interpreter specifically
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Only if you want to train the leaderboard model (~1GB more, mostly torch):
pip install -r requirements-transformer.txt
```

## Making changes

`src/disaster_tweets_nlp/` is the single source of truth — `config.py`
(paths, schema, constants), `data.py` (loading/splitting, the
duplicate/conflicting-label handling), `features.py` (TF-IDF + the
`has_location`/keyword-decoding logic), `model.py` (pipelines, training,
evaluation), `interpretability.py` (SHAP), `transformer.py` (the
leaderboard model). The notebook, the app, and the scripts all import
from here; nothing re-derives logic locally, so a change here propagates
everywhere automatically.

The edit loop:

```bash
# 1. Edit src/disaster_tweets_nlp/*.py

# 2. Check it against the test suite (fast, synthetic data, no download needed)
PYTHONPATH=src pytest tests/

# 3. Retrain, so models/model.joblib reflects your change
PYTHONPATH=src python scripts/train.py   # or --model, --no-resample, etc. — see --help
```

`models/model.joblib` is what the notebook, the app, and the report all
load — retraining is the one step that makes a model-code change visible
everywhere else.

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

## Climbing the leaderboard

Every step below was measured before being submitted; the competition
scores F1 on the "Disaster" class (not the macro-F1 used above), so
that's the metric in this table. Ranks are approximate, from a
leaderboard snapshot of 438 teams.

| # | Change | Local F1 | Public LB F1 | Rank |
|---|---|---|---|---|
| 1 | TF-IDF + LightGBM + `RandomOverSampler` (the explainable model) | 0.648 (5-fold CV, mirror) | 0.67269 | ~416 |
| — | Tune the decision threshold for F1 | 0.648 at the default 0.50 — already optimal | not submitted | — |
| 2 | Swap TF-IDF for `all-MiniLM-L6-v2` sentence embeddings, same LightGBM | 0.718 (5-fold CV, mirror) | 0.69659 | ~414 |
| — | Blend TF-IDF + embeddings (30/70) | 0.727 (5-fold CV, mirror) | not submitted | — |
| 3 | Fine-tune `distilbert-base-uncased` | 0.750 (holdout, mirror) | 0.75053 | ~402 |
| 4 | **`twitter-roberta-base` + keyword as a sentence pair + 5-fold ensemble, trained on the competition's own data** | **0.806 (5-fold OOF, official)** | **0.84339** | **~32** |

What actually moved the score:

- **The training data, more than the model.** The mirror is 81/19 Not
  Disaster/Disaster; the competition's `train.csv` is 57/43. Every model
  trained on the mirror learned to say "Disaster" too rarely (15–23% of
  test tweets, vs. 39% after switching), which F1 on the "Disaster"
  class punishes hard. Same RoBERTa recipe, first epoch: 0.718 on the
  mirror, 0.793 on the official data.
- **Local scores on the mirror didn't transfer reliably.** Step 2's CV
  gain (+0.070) became +0.024 on the leaderboard. Step 4's
  out-of-fold F1 on the official data was a much better predictor.
- **A tweet-pretrained transformer beat every classical approach** — but
  only once it was trained on data from the same source as the test set.
- **Tuning the threshold did nothing** (0.50 was already optimal) and a
  blend was too small a gain to spend a submission on — both reported
  rather than dropped.

The top ~7 leaderboard entries score ~1.0. These tweets are real,
searchable historical posts, so those scores almost certainly come from
looking up the true labels, not from modeling; leaving them out, this
project sits around rank 25.

The recipe in step 4 follows a [public
solution](https://github.com/KunyinSun/eda-competition) (private LB
0.83971); `src/disaster_tweets_nlp/transformer.py` reproduces it.

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
PYTHONPATH=src python scripts/train.py                    # default: LightGBM + RandomOverSampler — the measured winner
PYTHONPATH=src python scripts/train.py --model "Random Forest"
PYTHONPATH=src python scripts/train.py --no-resample       # class-weighting/plain training instead, for comparison
PYTHONPATH=src python scripts/train.py --help
```

## Generate a Kaggle submission

Join the competition first — its API 403s on every file until you have:
[kaggle.com/competitions/nlp-getting-started](https://www.kaggle.com/competitions/nlp-getting-started)
→ **Join Competition** / accept its rules, in a browser. Both scripts
then auto-download the competition files they need (see
`data.load_competition_train`/`load_competition_test`) and write
`submission.csv` as `id,target` rows with `1`/`0` labels.

**Leaderboard model** (public F1 0.84339, ~30 min on an Apple M4 GPU —
expect much longer on CPU only):

```bash
pip install -r requirements-transformer.txt   # once
python scripts/make_transformer_submission.py
kaggle competitions submit -c nlp-getting-started -f submission.csv -m "twitter-roberta 5-fold"
```

Each finished fold is cached under `data/processed/transformer_folds/`,
so an interrupted run picks up where it stopped. Delete that folder to
force a clean retrain. `--folds`/`--epochs` change the recipe (see
`--help`).

**Explainable model** (public F1 0.67269, seconds):

```bash
PYTHONPATH=src:. python scripts/make_submission.py
kaggle competitions submit -c nlp-getting-started -f submission.csv -m "tf-idf lightgbm"
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

`report/report.qmd` expects a Jupyter kernel named `disaster-tweets-nlp`
(its `jupyter:` front-matter key) — register it once from the activated
venv before rendering:

```bash
python -m ipykernel install --user --name disaster-tweets-nlp
quarto render report/report.qmd
```

This regenerates both `report/report.html` and `report/report.pdf` (PDF
needs a LaTeX distribution — if you don't have one, run
`quarto install tinytex` once). Render just one format when you don't need
both:

```bash
quarto render report/report.qmd --to html
quarto render report/report.qmd --to pdf
```

Live-preview while editing (auto-rerenders on save):

```bash
quarto preview report/report.qmd
```

A `.qmd` file is Markdown prose plus fenced Python code chunks
(` ```{python} `/` ``` `), executed top to bottom by the kernel above, same
as a notebook cell. Common per-chunk options (a `#|` comment, first line of
the chunk): `#| echo: false` (hide this chunk's source code),
`#| output: false` (suppress its output, e.g. a setup/import cell),
`#| label: fig-foo` + `#| fig-cap: "..."` (name and caption a figure for
cross-referencing). The [Quarto VS Code
extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto)
adds syntax highlighting and a one-click Render button if you're doing more
than a one-line edit.

Troubleshooting:

| Symptom | Likely cause |
|---|---|
| `Jupyter engine failed ... kernel not found` | The `ipykernel install --name disaster-tweets-nlp` step above hasn't been run yet. |
| `ModuleNotFoundError` inside a code chunk | `quarto render` runs with its working directory set to `report/`, not the project root — check the chunk's `sys.path.insert(0, "../src")` points at the right relative path. |
| Output looks stale after editing | Force a clean re-run: `quarto render report/report.qmd --execute-daemon-restart`. |
| PDF render fails, HTML succeeds | Missing LaTeX — run `quarto install tinytex` once, then retry. |

## Run the tests

```bash
PYTHONPATH=src pytest tests/
```

These test the feature engineering, duplicate/conflicting-label
handling, model logic, and the leaderboard model's input handling
directly with synthetic data — no download needed, and they already
pass without any real data. They never import `torch`, so they run
without `requirements-transformer.txt` installed.

## Deploy

The Streamlit app starts immediately from a bundled 1,000-row stratified sample sourced from Kaggle. Set `USE_FULL_KAGGLE_DATA=true` to fetch and use the complete dataset through the Kaggle API; configure either `KAGGLE_API_TOKEN` or a `[kaggle]` secrets section containing username and key.

- Repository: <https://github.com/nhamhhung/disaster-tweets-nlp>
- Report: <https://nhamhung.github.io/disaster-tweets-nlp/>
- Streamlit: <https://disaster-tweets-nlp.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
