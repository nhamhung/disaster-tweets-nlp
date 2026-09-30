"""Dataset Overview page: what's in the disaster-tweets dataset before
any modeling — class balance, missingness, duplicates, and text length.
"""

import plotly.express as px
import streamlit as st

from . import shared
from disaster_tweets_nlp import config


def render():
    st.title("📊 Dataset Overview")
    st.caption(
        "11,370 real tweets labeled for whether they describe a real "
        "disaster — a public mirror of Kaggle's \"Real or Not? NLP with "
        "Disaster Tweets\" task (the competition itself requires "
        "accepting its rules on the website, which the API can't do)."
    )

    df = shared.get_tweets_df()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tweets", f"{len(df):,}")
    c2.metric("Distinct keywords", f"{df[config.KEYWORD_COL].nunique():,}")
    c3.metric("Duplicate tweet texts", f"{df.duplicated(subset=[config.TEXT_COL]).sum():,}")

    st.subheader("Class balance is moderately imbalanced")
    st.caption(
        "Verified directly — not as extreme as this portfolio's other "
        "imbalanced-classification projects, but still enough that "
        "macro-F1 (not accuracy) is used throughout."
    )
    balance = (
        df[config.TARGET_COL]
        .map({0: "Not Disaster", 1: "Disaster"})
        .value_counts(normalize=True)
        .reset_index()
    )
    balance.columns = ["label", "share"]
    st.plotly_chart(px.bar(balance, x="label", y="share", title="Raw target class balance"), width="stretch")

    st.subheader("Tweet length")
    st.caption(
        "The Model Insights page's SHAP ranking finds tweet length is "
        "the single most predictive signal — real disaster tweets tend "
        "to be longer, news-style reports."
    )
    st.plotly_chart(
        px.histogram(df, x=df[config.TEXT_COL].str.len(), color=df[config.TARGET_COL].map({0: "Not Disaster", 1: "Disaster"}),
                      nbins=40, title="Tweet character length by label", labels={"x": "Character length", "color": "Label"},
                      barmode="overlay", opacity=0.6),
        width="stretch",
    )

    st.subheader("Missing data")
    missing = (df[[config.KEYWORD_COL, config.LOCATION_COL]].isna().mean() * 100).reset_index()
    missing.columns = ["column", "missing_pct"]
    st.plotly_chart(px.bar(missing, x="column", y="missing_pct", title="% missing by column"), width="stretch")
    st.caption(
        "`location` is missing ~30% of the time, and — as the Feature "
        "Engineering page explains — too inconsistent even when present "
        "(4,500+ distinct free-text values) to use directly."
    )
