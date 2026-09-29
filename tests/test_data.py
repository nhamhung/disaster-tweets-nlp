"""Tests for the duplicate/conflicting-label handling in
`data.split_features_target` — the central data-quality decision this
project makes (see `config.py`'s module docstring).
"""

import pandas as pd

from disaster_tweets_nlp import config, data


def test_conflicting_label_duplicates_are_dropped():
    df = pd.DataFrame(
        {
            config.ID_COL: [1, 2, 3],
            config.KEYWORD_COL: ["fire", "fire", "flood"],
            config.LOCATION_COL: [None, None, "USA"],
            config.TEXT_COL: ["same tweet text", "same tweet text", "a different tweet"],
            config.TARGET_COL: [0, 1, 1],  # rows 1 and 2 conflict
        }
    )
    X, y = data.split_features_target(df)

    assert len(X) == 1
    assert X[config.TEXT_COL].iloc[0] == "a different tweet"


def test_consistent_label_duplicates_are_deduplicated_to_one_row():
    df = pd.DataFrame(
        {
            config.ID_COL: [1, 2, 3],
            config.KEYWORD_COL: ["fire", "fire", "flood"],
            config.LOCATION_COL: [None, None, "USA"],
            config.TEXT_COL: ["repeated tweet", "repeated tweet", "a different tweet"],
            config.TARGET_COL: [1, 1, 0],
        }
    )
    X, y = data.split_features_target(df)

    assert len(X) == 2
    assert (X[config.TEXT_COL] == "repeated tweet").sum() == 1


def test_target_is_mapped_to_readable_string_labels():
    df = pd.DataFrame(
        {
            config.ID_COL: [1, 2],
            config.KEYWORD_COL: ["fire", "flood"],
            config.LOCATION_COL: [None, "USA"],
            config.TEXT_COL: ["tweet a", "tweet b"],
            config.TARGET_COL: [0, 1],
        }
    )
    _, y = data.split_features_target(df)

    assert set(y) == {"Not Disaster", "Disaster"}
