"""SHAP-based model interpretability.

`model.MODEL_FACTORIES` mixes tree-based estimators (`RandomForestClassifier`,
`LGBMClassifier` — the measured production winner) with linear ones
(`LogisticRegression`), so `compute_shap_values` picks the explainer
explicitly rather than relying on `shap.Explainer`'s generic
auto-dispatch: measured directly that the generic path silently falls
back to a slow permutation-based explainer for this project's
`RandomForestClassifier` (~70s for 200 rows) instead of the exact,
near-instant `TreeExplainer` — likely due to the unusual mix of sparse
TF-IDF-derived features it's given. `TreeExplainer` is requested
directly for tree-based estimators; `shap.Explainer`'s auto-dispatch
(which correctly picks `LinearExplainer`) is kept as the fallback for
everything else.
"""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 300, random_state: int = 42
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a fitted pipeline ending in a classifier."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    # `pipeline[:-1]` isn't safe here: when `resample=True` was used to
    # train, the second-to-last step is an imblearn sampler, which only
    # implements `fit_resample` (a no-op at inference time), not
    # `.transform()` — slicing it in would raise. Building the
    # preprocessing-only sub-pipeline explicitly by name sidesteps that.
    preprocessing = Pipeline(
        [(name, step) for name, step in pipeline.steps if name not in ("model", "resample")]
    )
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    transformed = preprocessing.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    X_transformed = pd.DataFrame(transformed, columns=feature_names, index=X.index)

    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_") or type(estimator).__name__ == "LGBMClassifier":
        # Tree-based estimator (RandomForestClassifier/LGBMClassifier from
        # `model.MODEL_FACTORIES`) — explicitly requesting TreeExplainer
        # here, rather than relying on `shap.Explainer`'s auto-dispatch,
        # because that generic path was measured to fall back to a slow
        # permutation-based explainer instead (~70s for 200 rows vs.
        # TreeExplainer's near-instant exact computation).
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.Explainer(estimator, X_transformed)
    explanation = explainer(X_transformed)
    return explanation, X_transformed


def top_shap_features(
    explanation: shap.Explanation, top_n: int = 15, class_index: int | None = None
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Handles both a linear model's single-output explanation (2D:
    `(n_samples, n_features)`) and a tree model's per-class explanation
    (3D: `(n_samples, n_features, n_classes)`, if `model.py`'s
    `MODEL_FACTORIES` is used to swap in `RandomForestClassifier` or
    `LGBMClassifier` instead of the default Logistic Regression).
    """
    values = explanation.values
    if values.ndim == 3:
        importance = abs(values[:, :, class_index]).mean(axis=0) if class_index is not None else abs(values).mean(axis=(0, 2))
    else:
        importance = abs(values).mean(axis=0)

    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
