"""Feature engineering and the shared preprocessing pipeline.

Single source of truth for turning the raw `keyword`/`location`/`text`
columns into model-ready features. The notebook, `scripts/train.py`, and
the Streamlit app all call `build_feature_pipeline()` (wrapped inside
the fitted pipeline saved to `models/model.joblib`) so none of them can
silently diverge.

Design decisions made fresh for text, not reused from this portfolio's
tabular projects:

- **TF-IDF, not raw word counts** — down-weights words that appear in
  almost every tweet (uninformative) relative to words that are
  distinctive of a smaller set of tweets, which is what actually
  separates classes here.
- **URLs and @mentions are stripped, hashtags are kept as words** — a
  URL's specific characters (mostly random shortened-link tokens) carry
  no signal and would only add noise/sparsity to the vocabulary; a
  hashtag's word content (`#earthquake` → `earthquake`) usually does.
- **`location` is reduced to a `has_location` flag, not used as text** —
  measured directly, not assumed: 4,504 distinct free-text values across
  7,952 non-null rows (see `config.py`'s docstring) is too sparse and
  inconsistent (`"USA"` vs. `"United States"` vs. `"US"`) for a
  categorical encoding to generalize.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_tweet_text(text: str) -> str:
    """Lowercase, strip URLs and @mentions, unwrap hashtags to their
    word content, drop remaining punctuation/digits, collapse
    whitespace. Applied identically at train and inference time via
    `TextFeatureEngineer`, so the vectorizer never sees a distribution
    shift between the two.
    """
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r" \1 ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_keyword(keyword) -> str:
    """`keyword` contains raw URL-encoding artifacts (e.g.
    `"mass%20murder"`) — found by checking, not assumed. `unquote`
    reverses that; missing keywords become an explicit `"none"` category
    rather than NaN, so `OneHotEncoder` sees one consistent value.
    """
    from urllib.parse import unquote

    if pd.isna(keyword):
        return "none"
    return unquote(str(keyword)).replace("_", " ")


class TextFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives `cleaned_text`, `keyword_clean`, and a handful of
    hand-crafted numeric signals (length, word/hashtag/mention/URL
    counts, whether a location was given at all) from the 3 raw
    columns.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "TextFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        raw_text = df[config.TEXT_COL].astype(str)

        df["cleaned_text"] = raw_text.apply(clean_tweet_text)
        df["keyword_clean"] = df[config.KEYWORD_COL].apply(clean_keyword)
        df["has_location"] = df[config.LOCATION_COL].notna().astype(int)
        df["text_length"] = raw_text.str.len()
        df["word_count"] = raw_text.str.split().str.len()
        df["url_count"] = raw_text.str.count(r"https?://\S+")
        df["mention_count"] = raw_text.str.count(r"@\w+")
        df["hashtag_count"] = raw_text.str.count(r"#\w+")

        return df.drop(columns=[config.TEXT_COL, config.KEYWORD_COL, config.LOCATION_COL])

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "cleaned_text",
                "keyword_clean",
                "has_location",
                "text_length",
                "word_count",
                "url_count",
                "mention_count",
                "hashtag_count",
            ]
        )


NUMERIC_COLS = ["has_location", "text_length", "word_count", "url_count", "mention_count", "hashtag_count"]


def build_preprocessor(max_tfidf_features: int = 5000) -> ColumnTransformer:
    """TF-IDF (word unigrams+bigrams) on the cleaned tweet text,
    one-hot-encoded cleaned keyword, and standard-scaled hand-crafted
    numeric signals — combined via `ColumnTransformer` into one sparse
    feature matrix.
    """
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=max_tfidf_features,
                    min_df=2,
                    stop_words="english",
                ),
                "cleaned_text",
            ),
            (
                "keyword",
                OneHotEncoder(handle_unknown="ignore"),
                ["keyword_clean"],
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_COLS,
            ),
        ]
    )


def build_feature_pipeline(max_tfidf_features: int = 5000) -> Pipeline:
    """`TextFeatureEngineer` + `build_preprocessor`, without a final
    estimator.
    """
    return Pipeline(
        steps=[
            ("engineer", TextFeatureEngineer()),
            ("preprocess", build_preprocessor(max_tfidf_features=max_tfidf_features)),
        ]
    )
