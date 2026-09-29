"""Tests for feature engineering, using a small synthetic panel instead
of the real disaster-tweets data.
"""

import numpy as np
import pandas as pd

from disaster_tweets_nlp import config
from disaster_tweets_nlp.features import (
    TextFeatureEngineer,
    build_feature_pipeline,
    clean_keyword,
    clean_tweet_text,
)


def make_synthetic_panel(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """One row per tweet, with all raw columns plus the target — mirrors
    the real schema.
    """
    rng = np.random.default_rng(seed)
    disaster_templates = [
        "BREAKING: massive earthquake hits the city, buildings collapsed http://t.co/abc123",
        "Wildfire spreading fast near the highway, evacuation ordered #wildfire",
        "Flood warning issued for the coastal region after heavy rain @weatherbot",
    ]
    normal_templates = [
        "just watched a great movie with my friends tonight, feeling happy",
        "this new song is absolutely on fire, can't stop listening to it",
        "traffic is such a disaster today, stuck for an hour @localnews",
    ]
    texts, targets = [], []
    for i in range(n_rows):
        # Each row gets a unique trailing token so `data.split_features_target`'s
        # duplicate-text detection (a real, deliberate part of this project's
        # data-quality handling) doesn't collapse this synthetic panel down to
        # just the handful of template strings.
        if rng.random() < 0.2:
            texts.append(f"{rng.choice(disaster_templates)} ref{i}")
            targets.append(1)
        else:
            texts.append(f"{rng.choice(normal_templates)} ref{i}")
            targets.append(0)

    df = pd.DataFrame(
        {
            config.ID_COL: range(n_rows),
            config.KEYWORD_COL: rng.choice(["earthquake", "fire", "mass%20murder", None], n_rows),
            config.LOCATION_COL: rng.choice(["USA", "United States", None, None], n_rows),
            config.TEXT_COL: texts,
            config.TARGET_COL: targets,
        }
    )
    return df


def test_clean_tweet_text_strips_urls_mentions_and_hashtag_marker():
    cleaned = clean_tweet_text("BREAKING: fire near @localnews http://t.co/abc123 #wildfire!!")
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "wildfire" in cleaned
    assert cleaned == cleaned.lower()


def test_clean_keyword_unescapes_url_encoding_and_fills_missing():
    assert clean_keyword("mass%20murder") == "mass murder"
    assert clean_keyword(None) == "none"
    assert clean_keyword(float("nan")) == "none"


def test_text_feature_engineer_derives_expected_columns():
    panel = make_synthetic_panel(n_rows=50)
    X = panel[[config.KEYWORD_COL, config.LOCATION_COL, config.TEXT_COL]]
    engineered = TextFeatureEngineer().fit_transform(X)

    for col in ["cleaned_text", "keyword_clean", "has_location", "text_length", "word_count"]:
        assert col in engineered.columns
    assert config.TEXT_COL not in engineered.columns
    assert engineered["has_location"].isin([0, 1]).all()


def test_full_feature_pipeline_runs_end_to_end():
    panel = make_synthetic_panel(n_rows=300)
    X = panel[[config.KEYWORD_COL, config.LOCATION_COL, config.TEXT_COL]]
    y = panel[config.TARGET_COL]

    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(X, y)

    assert transformed.shape[0] == len(X)
