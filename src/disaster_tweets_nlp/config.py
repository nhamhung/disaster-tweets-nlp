"""Paths, constants, and column schema for the Disaster Tweets dataset.

Dataset: tweets labeled for whether they describe a real disaster (1) or
not (0) — the classic "Real or Not? NLP with Disaster Tweets" task.
The original Kaggle *competition* (`nlp-getting-started`) requires
accepting its rules on the website before the API will serve it —
verified directly (`kaggle competitions download` returns a 403 even
when authenticated). This project instead uses a public *dataset*
mirror of the same task, `vstepanenko/disaster-tweets` — same 4 input
columns (`keyword`, `location`, `text`, `target`), a larger, re-scraped
sample (11,370 rows vs. the original 7,613).

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
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV = DATA_RAW_DIR / "tweets.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

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
