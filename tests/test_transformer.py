"""Tests for the leaderboard model's input handling. Deliberately never
trigger a `torch` import (see `transformer.py`'s docstring) — the
fine-tuning itself is too slow and download-heavy for a unit test.
"""

import sys

import numpy as np
import pandas as pd

from disaster_tweets_nlp import config, transformer


def _frame(keywords, texts):
    return pd.DataFrame(
        {config.KEYWORD_COL: keywords, config.LOCATION_COL: [None] * len(texts), config.TEXT_COL: texts}
    )


def test_keyword_is_url_decoded_and_missing_becomes_none():
    keywords, _ = transformer.build_pair_inputs(_frame(["mass%20murder", np.nan], ["a", "b"]))
    assert keywords == ["mass murder", "none"]


def test_text_is_passed_through_raw():
    raw = "Forest fire near La Ronge @user http://t.co/abc #wildfire"
    _, texts = transformer.build_pair_inputs(_frame(["fire"], [raw]))
    assert texts == [raw]


def test_importing_the_module_does_not_load_torch():
    assert "torch" not in sys.modules
