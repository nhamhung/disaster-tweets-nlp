"""Train the production pipeline on the full dataset and save it.

Equivalent to the notebook's modeling steps, without the notebook.

Default is `LightGBM` + oversampling (`RandomOverSampler`), not the
Logistic Regression that's usually the reflexive first choice for
TF-IDF text — verified directly (see the model sweep in the
notebook/report) that this combination beats every other tested
model/imbalance-handling pairing on macro-F1, including class-weighted
Logistic Regression and Random Forest, and stays fast enough for SHAP
to explain interactively (see `model.lightgbm_estimator`'s docstring).
"""

import argparse

from disaster_tweets_nlp import config, data, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(model.MODEL_FACTORIES.keys()),
        default="LightGBM",
        help="Which model family to train (default: LightGBM).",
    )
    parser.add_argument(
        "--resample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert RandomOverSampler oversampling (default: on — the measured winner).",
    )
    args = parser.parse_args()

    df = data.load_tweets()
    X, y = data.split_features_target(df)
    estimator = model.MODEL_FACTORIES[args.model]()
    pipeline = model.train_pipeline(X, y, estimator=estimator, resample=args.resample)
    model.save_pipeline(pipeline)
    print(f"Trained {args.model} (resample={args.resample}) on {len(df)} rows ({len(X)} after dedup). Saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
