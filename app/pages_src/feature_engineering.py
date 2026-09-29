"""Feature Engineering page: what `TextFeatureEngineer` derives, and
why — plus the central data-quality decisions this project makes
before any of it runs.
"""

import plotly.express as px
import streamlit as st

from . import shared
from disaster_tweets_nlp import config
from disaster_tweets_nlp.features import TextFeatureEngineer, clean_keyword, clean_tweet_text


def render():
    st.title("🔧 Feature Engineering")
    st.caption("The real engineering decisions behind this project's feature set.")

    df = shared.get_tweets_df()

    st.subheader("1. Resolving duplicates before anything else")
    dup_texts = df[df.duplicated(subset=[config.TEXT_COL], keep=False)]
    conflicting = dup_texts.groupby(config.TEXT_COL)[config.TARGET_COL].nunique()
    conflicting = conflicting[conflicting > 1]
    st.markdown(
        f"**{df.duplicated(subset=[config.TEXT_COL]).sum()} exact-duplicate tweet texts**, "
        f"**{len(conflicting)} of which carry conflicting labels** (the same tweet "
        "labeled both 0 and 1). Left alone, duplicates could leak across a "
        "train/validation split, and conflicting-label rows can't be "
        "trusted either way — `data.split_features_target` drops the "
        "conflicting ones and deduplicates the rest before any modeling."
    )

    st.subheader("2. Cleaning tweet text")
    sample_text = "BREAKING: massive #wildfire spreading fast @localnews http://t.co/abc123"
    st.code(f'Raw:     "{sample_text}"\nCleaned: "{clean_tweet_text(sample_text)}"', language=None)
    st.markdown(
        "Lowercase, strip URLs and `@mentions` (their specific characters "
        "carry no signal), unwrap hashtags to their word content — then "
        "TF-IDF (word uni+bigrams, up to 5,000 features) turns the "
        "cleaned text into a sparse feature matrix."
    )

    st.subheader("3. Fixing a real formatting bug in `keyword`")
    st.code(f'Raw:     "mass%20murder"\nCleaned: "{clean_keyword("mass%20murder")}"', language=None)
    st.markdown(
        "Some `keyword` values contain raw URL-encoding — found by "
        "checking, not assumed. `clean_keyword` reverses it before "
        "one-hot encoding."
    )

    st.subheader("4. `location` reduced to a `has_location` flag")
    st.markdown(
        f"**{df[config.LOCATION_COL].nunique():,} distinct free-text values** across "
        f"{df[config.LOCATION_COL].notna().sum():,} non-null rows — `\"USA\"`, "
        "`\"United States\"`, and `\"US\"` all counted separately. Too sparse "
        "and inconsistent to one-hot-encode usefully, so this project "
        "keeps only whether a location was given at all."
    )
    top_locations = df[config.LOCATION_COL].value_counts().head(10).reset_index()
    top_locations.columns = ["location", "count"]
    st.plotly_chart(px.bar(top_locations, x="location", y="count", title="Top 10 raw location values"), width="stretch")

    st.subheader("5. Hand-crafted numeric signals")
    engineered = TextFeatureEngineer().fit_transform(
        df[[config.KEYWORD_COL, config.LOCATION_COL, config.TEXT_COL]]
    )
    st.dataframe(
        engineered[["text_length", "word_count", "url_count", "mention_count", "hashtag_count", "has_location"]].describe().T
    )
    st.caption(
        "Length and word count turn out to be the single most predictive "
        "features of all (see Model Insights) — worth engineering "
        "explicitly rather than relying on TF-IDF alone to capture them."
    )
