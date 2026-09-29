"""Cached data/model loaders shared across every page."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from disaster_tweets_nlp import data, interpretability, model  # noqa: E402


@st.cache_resource
def get_pipeline():
    return model.load_pipeline()


@st.cache_data
def get_tweets_df() -> pd.DataFrame:
    return data.load_tweets()


@st.cache_data(show_spinner="Computing SHAP values (first load only)...")
def get_shap_explanation(sample_size: int = 300):
    pipeline = get_pipeline()
    df = get_tweets_df()
    X, _ = data.split_features_target(df)
    explanation, X_transformed = interpretability.compute_shap_values(
        pipeline, X, max_samples=sample_size
    )
    return explanation, X_transformed
