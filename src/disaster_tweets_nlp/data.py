"""Data loading helpers.

The raw CSV is not committed to the repo (best fetched fresh rather than
duplicated here). Download it first — see the project README.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first — see the "
            "README's 'Get the data' section, e.g.:\n"
            f"  kaggle datasets download -d {config.KAGGLE_DATASET} -p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / 'disaster-tweets.zip'} -d {config.DATA_RAW_DIR}"
        )
    return path


def using_demo_data() -> bool:
    """Return whether the compact, repository-packaged sample is active."""
    return not config.RAW_CSV.exists()


def load_tweets() -> pd.DataFrame:
    """Load the full table when present, otherwise the deployment sample."""
    path = config.DEMO_CSV if using_demo_data() else config.RAW_CSV
    return pd.read_csv(_require_file(path))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into (X, y), after resolving two real data-quality issues
    found by checking the raw CSV directly (see `config.py`'s module
    docstring):

    1. **Conflicting-label duplicates**: 2 exact-duplicate tweet texts
       carry different `target` values — genuine annotation
       disagreement, not something a model can learn from either
       version of, so both copies are dropped entirely.
    2. **Consistent-label duplicates**: 145 more duplicate texts (same
       text, same label) are deduplicated to one row each. Left alone,
       the same tweet could land in both a train and a validation fold,
       inflating validation scores with a memorized exact match rather
       than a genuine generalization.

    `y` is mapped from the raw `0`/`1` integers to `config.TARGET_CLASSES`
    string labels, matching the rest of this portfolio's convention of
    human-readable class labels.
    """
    missing = {config.TEXT_COL, config.KEYWORD_COL, config.LOCATION_COL, config.TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Input frame is missing expected columns: {sorted(missing)}")

    label_counts_per_text = df.groupby(config.TEXT_COL)[config.TARGET_COL].nunique()
    conflicting_texts = set(label_counts_per_text[label_counts_per_text > 1].index)
    clean = df[~df[config.TEXT_COL].isin(conflicting_texts)].copy()
    clean = clean.drop_duplicates(subset=[config.TEXT_COL], keep="first")

    X = clean[[config.KEYWORD_COL, config.LOCATION_COL, config.TEXT_COL]].copy()
    y = pd.Series(
        np.where(clean[config.TARGET_COL] == 1, "Disaster", "Not Disaster"),
        index=clean.index,
        name=config.TARGET_COL,
    )
    return X, y
