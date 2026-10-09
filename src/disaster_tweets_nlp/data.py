"""Data loading helpers.

The raw CSV is not committed to the repo (best fetched fresh rather than
duplicated here). Download it first — see the project README — or let
`_require_file` fetch it automatically via the Kaggle API (used when
deploying without a Docker image that already bakes the file in; see
`app/pages_src/shared.py` for how deployed credentials get wired in).
"""

import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import config


def _download_from_kaggle() -> bool:
    """Best-effort automatic fetch via the Kaggle API. Returns whether
    the target file exists afterward. Silently does nothing (returns
    False) if the `kaggle` package isn't installed or no credentials
    are configured — callers fall back to the manual-download error
    message either way, so this never needs to be trusted to succeed.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.DATA_RAW_DIR), unzip=True, quiet=True)
    except Exception:
        return False
    return config.RAW_CSV.exists()


def _require_file(path: Path) -> Path:
    if not path.exists():
        _download_from_kaggle()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, and automatic download via the Kaggle API "
            "didn't produce it either (no credentials configured, or the "
            "`kaggle` package isn't installed). Download the dataset "
            "manually instead — see the README's 'Get the data' section, e.g.:\n"
            f"  kaggle datasets download -d {config.KAGGLE_DATASET} -p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / 'disaster-tweets.zip'} -d {config.DATA_RAW_DIR}"
        )
    return path


def using_sample_data() -> bool:
    """Whether the bundled Kaggle-derived sample is the active data source."""
    return config.SAMPLE_CSV.exists() and os.getenv("USE_FULL_KAGGLE_DATA", "").lower() not in {"1", "true", "yes"}


def load_tweets() -> pd.DataFrame:
    """Load fast sample data by default; opt into the full Kaggle download with USE_FULL_KAGGLE_DATA=true."""
    path = config.SAMPLE_CSV if using_sample_data() else _require_file(config.RAW_CSV)
    return pd.read_csv(path)


def _download_competition_file(remote_name: str, local_path: Path) -> bool:
    """Best-effort automatic fetch of one of the real competition's
    files, renamed to `local_path` so it can't collide with the dataset
    mirror. Requires having accepted the competition's rules on
    kaggle.com first; returns False otherwise (403), same as every other
    failure mode here.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.competition_download_file(
            config.KAGGLE_COMPETITION, remote_name, path=str(config.DATA_RAW_DIR), quiet=True
        )
        downloaded = config.DATA_RAW_DIR / remote_name
        zipped = config.DATA_RAW_DIR / f"{remote_name}.zip"
        if zipped.exists():
            with ZipFile(zipped) as zf:
                zf.extractall(config.DATA_RAW_DIR)
            zipped.unlink()
        downloaded.rename(local_path)
    except Exception:
        return False
    return local_path.exists()


def _require_competition_file(remote_name: str, local_path: Path) -> Path:
    if not local_path.exists():
        _download_competition_file(remote_name, local_path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"{local_path} not found, and automatic download via the Kaggle API "
            "didn't produce it either. Make sure you've joined the competition at "
            f"https://www.kaggle.com/competitions/{config.KAGGLE_COMPETITION} "
            "(accept its rules in a browser — the API 403s until you have), then "
            "download manually:\n"
            f"  kaggle competitions download -c {config.KAGGLE_COMPETITION} -f {remote_name} -p {config.DATA_RAW_DIR}\n"
            f"  mv {config.DATA_RAW_DIR / remote_name} {local_path}"
        )
    return local_path


def load_competition_train() -> pd.DataFrame:
    """Load the real competition's labeled train set (7,613 rows, 57/43
    class balance — unlike the mirror's 81/19). Used by the leaderboard
    transformer model (`scripts/make_transformer_submission.py`).
    """
    return pd.read_csv(_require_competition_file("train.csv", config.COMPETITION_TRAIN_CSV))


def load_competition_test() -> pd.DataFrame:
    """Load the real competition's test set (`id`, `keyword`, `location`,
    `text` — no `target`). Used only to generate Kaggle submissions.
    """
    _require_competition_file("sample_submission.csv", config.COMPETITION_SAMPLE_SUBMISSION_CSV)
    return pd.read_csv(_require_competition_file("test.csv", config.COMPETITION_TEST_CSV))


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
