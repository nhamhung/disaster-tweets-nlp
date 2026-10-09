#!/usr/bin/env python
"""Generate a Kaggle-submittable submission.csv from the trained model.

Requires having joined the competition on kaggle.com first (accept its
rules in a browser) — the API 403s on its test set until you have:
https://www.kaggle.com/competitions/nlp-getting-started

Usage:
    python scripts/make_submission.py [--output submission.csv]

Then submit with the Kaggle CLI:
    kaggle competitions submit -c nlp-getting-started -f submission.csv -m "message"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disaster_tweets_nlp import config, data, model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission.csv",
        help="Where to write the submission file (default: ./submission.csv)",
    )
    args = parser.parse_args()

    print(f"Loading the competition's test set from {config.COMPETITION_TEST_CSV} ...")
    test_df = data.load_competition_test()
    feature_cols = [config.KEYWORD_COL, config.LOCATION_COL, config.TEXT_COL]
    missing = set(feature_cols) - set(test_df.columns)
    if missing:
        raise ValueError(f"Competition test data is missing expected columns: {sorted(missing)}")
    X_test = test_df[feature_cols].copy()

    print(f"Loading trained pipeline from {config.MODEL_PATH} ...")
    pipeline = model.load_pipeline()

    print(f"Predicting {len(X_test)} rows ...")
    predictions = pipeline.predict(X_test)
    # The pipeline predicts human-readable labels (config.TARGET_CLASSES);
    # the competition's own format expects the original 0/1 integers.
    target = (predictions == "Disaster").astype(int)

    submission = test_df[[config.ID_COL]].copy()
    submission[config.TARGET_COL] = target

    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
