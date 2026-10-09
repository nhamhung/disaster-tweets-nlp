#!/usr/bin/env python
"""Train the leaderboard model (5-fold twitter-roberta-base ensemble on the
competition's own train set) and write a Kaggle-submittable submission.csv.

Public LB F1 0.84339 (~rank 32 of 438 at the time), vs. 0.67269 for
`make_submission.py`'s TF-IDF model. Needs the extra dependencies:
    pip install -r requirements-transformer.txt

Requires having joined the competition on kaggle.com first (accept its
rules in a browser) — the API 403s on its files until you have:
https://www.kaggle.com/competitions/nlp-getting-started

Usage:
    python scripts/make_transformer_submission.py [--folds 5] [--epochs 3] [--output submission.csv]

A full run is ~30 min on an Apple M4 (MPS); each finished fold is cached
under data/processed/transformer_folds/, so an interrupted run resumes.

Then submit with the Kaggle CLI:
    kaggle competitions submit -c nlp-getting-started -f submission.csv -m "message"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import f1_score  # noqa: E402

from disaster_tweets_nlp import config, data, transformer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=config.TRANSFORMER_N_FOLDS)
    parser.add_argument("--epochs", type=int, default=config.TRANSFORMER_EPOCHS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission.csv",
        help="Where to write the submission file (default: ./submission.csv)",
    )
    args = parser.parse_args()

    train_df = data.load_competition_train()
    X, y = data.split_features_target(train_df)
    print(f"Competition train set: {len(X)} rows after dropping conflicting/duplicate texts (from {len(train_df)})")

    test_df = data.load_competition_test()
    print(f"Competition test set: {len(test_df)} rows")

    print(f"Fine-tuning {config.TRANSFORMER_MODEL_NAME}: {args.folds} folds x {args.epochs} epochs ...")
    oof, test_proba = transformer.train_kfold_ensemble(
        X, y, test_df, n_folds=args.folds, epochs=args.epochs, log=lambda msg: print(msg, flush=True)
    )

    oof_f1 = f1_score((y == "Disaster").astype(int), (oof >= 0.5).astype(int))
    print(f"Out-of-fold F1 (Disaster class, the competition's metric): {oof_f1:.4f}")

    submission = test_df[[config.ID_COL]].copy()
    submission[config.TARGET_COL] = (test_proba >= 0.5).astype(int)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")
    print(f"Predicted 'Disaster' share: {submission[config.TARGET_COL].mean():.1%}")


if __name__ == "__main__":
    main()
