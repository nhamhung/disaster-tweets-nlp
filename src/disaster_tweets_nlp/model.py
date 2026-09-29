"""Model pipeline construction, training, evaluation, and persistence.

Task: predict whether a tweet describes a real disaster (`"Disaster"`)
or not (`"Not Disaster"`) from its text, keyword, and location fields
only (see `features.py`). Moderately imbalanced (~18.6% Disaster) —
macro-F1 (not accuracy) is used throughout so a model that leans on the
majority class doesn't look artificially good.

Pipelines are built with `imblearn.pipeline.Pipeline` (not sklearn's)
so `build_pipeline(..., resample=True)` can insert oversampling as a
step without a separate code path — a no-op outside of `.fit()`, so
resampled rows never leak into a validation score or a served
prediction.

Resampling choice, made fresh for this project rather than reused: the
other tabular projects in this portfolio compare class-weighting against
**ADASYN**, which synthesizes new rows from a k-nearest-neighbors
search. That search is a poor fit for TF-IDF's thousands of sparse,
mostly-zero dimensions (the resulting "neighbors" are usually just
about noise), so this project compares class-weighting against
**`RandomOverSampler`** instead — duplicates existing minority rows,
which is the standard cheap baseline for text classification imbalance.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from . import config
from .features import build_feature_pipeline


def logistic_estimator() -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)


def balanced_logistic_estimator() -> BaseEstimator:
    """The same Logistic Regression, but with `class_weight="balanced"`
    — reweights the loss inversely to class frequency instead of
    resampling rows, a cheaper alternative worth comparing directly
    rather than assuming either wins.
    """
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED)


def random_forest_estimator() -> BaseEstimator:
    """`max_depth=20` is a deliberate cap, not the sklearn default
    (`None`, unlimited). Measured directly: unlimited depth reaches an
    average tree depth of ~330 on this TF-IDF feature space (mostly
    from memorizing `RandomOverSampler`'s exact-duplicate minority
    rows) and scores highest on macro-F1 (~0.79 vs. ~0.75 capped at 20)
    — but makes `shap.TreeExplainer` intractable (didn't finish
    computing SHAP values for even 30 rows within 2 minutes). Capping
    depth is a deliberate, measured tradeoff: a usable, explainable
    model over the last few points of cross-validated macro-F1.
    """
    return RandomForestClassifier(
        n_estimators=300, max_depth=20, random_state=config.RANDOM_SEED, n_jobs=-1
    )


def lightgbm_estimator() -> BaseEstimator:
    """The measured production default (see `build_pipeline`). Unlike
    `RandomForestClassifier`, LightGBM's leaf-wise growth is bounded by
    `num_leaves` (default 31) rather than growing unconstrained depth —
    so it reaches both the best macro-F1 of every model/resampling
    combination tested *and* a `shap.TreeExplainer` call fast enough to
    use interactively (~0.5s for 300 rows, vs. the capped Random
    Forest's ~6s and the uncapped one's 2+ minutes).
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, random_state=config.RANDOM_SEED, verbosity=-1)


MODEL_FACTORIES: dict[str, "callable[[], BaseEstimator]"] = {
    "Logistic Regression": logistic_estimator,
    "Logistic Regression (balanced)": balanced_logistic_estimator,
    "Random Forest": random_forest_estimator,
    "LightGBM": lightgbm_estimator,
}


def build_pipeline(
    estimator: BaseEstimator | None = None, resample: bool = False, max_tfidf_features: int = 5000
) -> Pipeline:
    """`resample=True` inserts `RandomOverSampler` right after
    preprocessing — see the module docstring for why this project uses
    it instead of ADASYN.
    """
    pipeline = build_feature_pipeline(max_tfidf_features=max_tfidf_features)
    steps = list(pipeline.steps)
    if resample:
        steps.append(("resample", RandomOverSampler(random_state=config.RANDOM_SEED)))
    steps.append(("model", estimator if estimator is not None else lightgbm_estimator()))
    return Pipeline(steps=steps)


def cross_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator | None = None,
    cv: int = 5,
    resample: bool = False,
) -> dict:
    """Stratified, *shuffled* cross_validate; returns mean/std accuracy
    and macro-F1.

    Shuffling isn't the default for an integer `cv` in scikit-learn, and
    it matters more here than in this portfolio's other projects: the
    raw CSV is sorted into contiguous blocks by `keyword` (verified
    directly), and disaster-rate varies a lot keyword-to-keyword — an
    unshuffled split would hand each fold a different, biased keyword
    mix, inflating the std across folds rather than reflecting genuine
    model variance.
    """
    pipeline = build_pipeline(estimator, resample=resample)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
    scores = cross_validate(
        pipeline, X, y, cv=splitter, scoring=["accuracy", "f1_macro"], return_train_score=False
    )
    return {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_macro_mean": scores["test_f1_macro"].mean(),
        "f1_macro_std": scores["test_f1_macro"].std(),
    }


def train_pipeline(
    X: pd.DataFrame, y: pd.Series, estimator: BaseEstimator | None = None, resample: bool = False
) -> Pipeline:
    """Fit a fresh pipeline on the full given data."""
    pipeline = build_pipeline(estimator, resample=resample)
    pipeline.fit(X, y)
    return pipeline


def save_pipeline(pipeline: Pipeline, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train a model first: `python scripts/train.py`."
        )
    return joblib.load(path)


def cross_val_disaster_probabilities(
    X: pd.DataFrame, y: pd.Series, estimator: BaseEstimator | None = None, cv: int = 5
) -> np.ndarray:
    """Out-of-fold predicted probability of the "Disaster" class for
    every row, via `cross_val_predict` — needed for a precision/recall
    threshold sweep.

    Relies on `predict_proba`'s columns being ordered by
    `sorted(y.unique())` (sklearn's convention) — asserted directly
    rather than assumed, since getting this silently backwards would
    flip precision and recall.
    """
    classes = sorted(y.unique())
    assert set(classes) == set(config.TARGET_CLASSES), f"Unexpected classes {classes}, expected {config.TARGET_CLASSES}"
    disaster_index = classes.index("Disaster")

    pipeline = build_pipeline(estimator if estimator is not None else balanced_logistic_estimator())
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
    proba = cross_val_predict(pipeline, X, y, cv=splitter, method="predict_proba")
    return proba[:, disaster_index]


def threshold_sweep(y: pd.Series, proba: np.ndarray, target_recalls: list[float]) -> pd.DataFrame:
    """For each target recall level (recall of the "Disaster" class),
    find the threshold that achieves it and report the precision paid
    for it.
    """
    y_numeric = (y == "Disaster").astype(int)
    precision, recall, thresholds = precision_recall_curve(y_numeric, proba)
    rows = []
    for target in target_recalls:
        idx = int(np.argmin(np.abs(recall[:-1] - target)))
        rows.append(
            {
                "target_recall": target,
                "threshold": thresholds[idx],
                "precision": precision[idx],
                "recall": recall[idx],
            }
        )
    return pd.DataFrame(rows)
