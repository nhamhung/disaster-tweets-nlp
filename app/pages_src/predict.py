"""Predict page: type any tweet (or load a real one), optionally add a
keyword/location, and see the model's predicted label update.
"""

import pandas as pd
import streamlit as st

from . import shared
from disaster_tweets_nlp import config


def _load_random_tweet():
    df = shared.get_tweets_df()
    row = df.sample(1).iloc[0]
    st.session_state["tweet_text"] = row[config.TEXT_COL]
    st.session_state["tweet_keyword"] = row[config.KEYWORD_COL] if pd.notna(row[config.KEYWORD_COL]) else ""
    st.session_state["tweet_location"] = row[config.LOCATION_COL] if pd.notna(row[config.LOCATION_COL]) else ""
    st.session_state["actual_label"] = "Disaster" if row[config.TARGET_COL] == 1 else "Not Disaster"
    st.session_state["loaded_a_record"] = True


def _clear_fields():
    st.session_state["tweet_text"] = ""
    st.session_state["tweet_keyword"] = ""
    st.session_state["tweet_location"] = ""
    st.session_state.pop("actual_label", None)
    st.session_state["loaded_a_record"] = False


def render():
    st.title("🎯 Predict: Is This Tweet About a Real Disaster?")
    st.caption(
        "Type or paste any tweet text and the model predicts whether it "
        "describes a real disaster (**Disaster**) or not (**Not "
        "Disaster**). `keyword` and `location` are optional — the model "
        "was trained to work from text alone plus these two weak "
        "signals."
    )

    try:
        pipeline = shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    for key, default in [("tweet_text", ""), ("tweet_keyword", ""), ("tweet_location", "")]:
        if key not in st.session_state:
            st.session_state[key] = default

    col1, col2 = st.columns(2)
    with col1:
        st.button("🎲 Load a random real tweet", on_click=_load_random_tweet, width="stretch")
    with col2:
        st.button("↺ Clear", on_click=_clear_fields, width="stretch")

    if st.session_state.get("loaded_a_record"):
        st.info("Loaded a real tweet. Its actual label is revealed after you predict.")

    st.text_area("Tweet text", key="tweet_text", height=100, placeholder="e.g. Massive wildfire spreading near the highway, evacuation ordered")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Keyword (optional)", key="tweet_keyword", placeholder="e.g. wildfire")
    with c2:
        st.text_input("Location (optional)", key="tweet_location", placeholder="e.g. California")

    if st.button("Predict", type="primary"):
        if not st.session_state["tweet_text"].strip():
            st.warning("Enter some tweet text first.")
            st.stop()

        row = pd.DataFrame(
            [
                {
                    config.KEYWORD_COL: st.session_state["tweet_keyword"] or None,
                    config.LOCATION_COL: st.session_state["tweet_location"] or None,
                    config.TEXT_COL: st.session_state["tweet_text"],
                }
            ]
        )

        prediction = pipeline.predict(row)[0]
        proba = pipeline.predict_proba(row)[0]
        classes = pipeline.named_steps["model"].classes_

        actual = st.session_state.get("actual_label")
        if actual is not None:
            match = "✅ matches the model" if actual == prediction else "❌ differs from the model"
            st.subheader(f"Prediction: **{prediction}**  |  Actual: **{actual}** ({match})")
        else:
            st.subheader(f"Prediction: **{prediction}**")

        proba_df = pd.DataFrame({"Label": classes, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        st.bar_chart(proba_df.set_index("Label"))
        st.dataframe(proba_df, hide_index=True)
