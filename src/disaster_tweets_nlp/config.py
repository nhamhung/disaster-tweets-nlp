"""Paths, constants, and column schema for the Disaster Tweets dataset.

Dataset: tweets labeled for whether they describe a real disaster (1) or
not (0) — the classic "Real or Not? NLP with Disaster Tweets" task.
The original Kaggle *competition* (`nlp-getting-started`) requires
accepting its rules on the website before the API will serve it —
verified directly (`kaggle competitions download` returns a 403 until
you have). The explainable TF-IDF model (notebook, app, `train.py`)
uses a public *dataset* mirror of the same task,
`vstepanenko/disaster-tweets` — same 4 input columns (`keyword`,
`location`, `text`, `target`), a larger, re-scraped sample (11,370 rows
vs. the original 7,613) that needs no rule acceptance. The leaderboard
transformer model (`transformer.py`) uses the competition's own data.

Verified directly against the real downloaded CSV (not assumed):
- Class balance is 81.4% "Not Disaster" / 18.6% "Disaster" — moderately
  imbalanced, not as extreme as some of this portfolio's other projects,
  but still enough that accuracy alone would be misleading.
- 147 exact-duplicate tweet texts, 2 of which carry *conflicting*
  labels (the same tweet labeled both 0 and 1) — real annotation noise,
  not a bug in this project's own code.
- `location` is free text with no schema (4,504 distinct values across
  7,952 non-null rows — e.g. "USA", "United States", "US" all appear
  separately) and ~30% missing — too sparse and inconsistent to use as
  a categorical feature directly.
- `keyword` contains raw URL-encoding artifacts (e.g. `"mass%20murder"`
  instead of `"mass murder"`) — a real formatting bug, fixed by
  unescaping before use.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV = DATA_RAW_DIR / "tweets.csv"
SAMPLE_CSV = DATA_SAMPLE_DIR / "tweets_sample.csv.gz"
MODEL_PATH = MODELS_DIR / "model.joblib"

# The real competition's own files — kept separate from RAW_CSV/the
# dataset mirror above, since they're two different sources with no
# guaranteed row overlap. Verified directly: the mirror is 81/19
# Not Disaster/Disaster, the competition's train.csv is 57/43 — a model
# trained on the mirror under-predicts "Disaster" on the competition's
# test set, which is why the leaderboard model trains on
# COMPETITION_TRAIN_CSV instead (see `transformer.py`).
COMPETITION_TRAIN_CSV = DATA_RAW_DIR / "competition_train.csv"
COMPETITION_TEST_CSV = DATA_RAW_DIR / "competition_test.csv"
COMPETITION_SAMPLE_SUBMISSION_CSV = DATA_RAW_DIR / "competition_sample_submission.csv"
TRANSFORMER_FOLDS_DIR = DATA_PROCESSED_DIR / "transformer_folds"

# --- Kaggle source ---------------------------------------------------------
# A dataset mirror, not the competition itself — see the module
# docstring for why the competition API download is unusable here.

KAGGLE_DATASET = "vstepanenko/disaster-tweets"

# --- Columns -------------------------------------------------------------

ID_COL = "id"
TEXT_COL = "text"
KEYWORD_COL = "keyword"
LOCATION_COL = "location"
TARGET_COL = "target"

TARGET_CLASSES = ["Not Disaster", "Disaster"]
RANDOM_SEED = 42

# --- Leaderboard transformer model (see `transformer.py`) ------------------
# Recipe from a public solution (private LB F1 0.83971); reproduced here
# at public LB F1 0.84339. A RoBERTa pretrained on tweets, not generic text.

TRANSFORMER_MODEL_NAME = "cardiffnlp/twitter-roberta-base"
TRANSFORMER_N_FOLDS = 5
TRANSFORMER_EPOCHS = 3
TRANSFORMER_BATCH_SIZE = 32
TRANSFORMER_MAX_LENGTH = 64
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_FRACTION = 0.1
