"""Model Insights page: model family comparison, class-imbalance
handling comparison, decision-threshold tuning, and SHAP — all computed
live (this dataset is only ~11,000 rows, so a 5-fold sweep finishes
quickly).
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared
from disaster_tweets_nlp import model
from disaster_tweets_nlp.interpretability import top_shap_features

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f", "#81b29a"]


@st.cache_data(show_spinner="Sweeping model families (first load only)...")
def _model_sweep():
    from disaster_tweets_nlp import data

    df = shared.get_tweets_df()
    X, y = data.split_features_target(df)
    rows = []
    for name, factory in model.MODEL_FACTORIES.items():
        for resample in [False, True]:
            label = f"{name} + ROS" if resample else name
            result = model.cross_validate_pipeline(X, y, estimator=factory(), cv=5, resample=resample)
            rows.append({"model": label, "f1_macro": result["f1_macro_mean"], "accuracy": result["accuracy_mean"]})
    return pd.DataFrame(rows).sort_values("f1_macro")


@st.cache_data(show_spinner="Comparing class-imbalance handling (first load only)...")
def _imbalance_comparison():
    from disaster_tweets_nlp import data

    df = shared.get_tweets_df()
    X, y = data.split_features_target(df)
    rows = [
        {"approach": "Plain Logistic Regression", **model.cross_validate_pipeline(X, y, estimator=model.logistic_estimator(), cv=5)},
        {"approach": "Logistic Regression + RandomOverSampler", **model.cross_validate_pipeline(X, y, estimator=model.logistic_estimator(), cv=5, resample=True)},
        {"approach": "Logistic Regression (class_weight=balanced)", **model.cross_validate_pipeline(X, y, estimator=model.balanced_logistic_estimator(), cv=5)},
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Sweeping decision thresholds (first load only)...")
def _threshold_sweep():
    from disaster_tweets_nlp import data

    df = shared.get_tweets_df()
    X, y = data.split_features_target(df)
    proba = model.cross_val_disaster_probabilities(X, y, cv=5)
    return model.threshold_sweep(y, proba, target_recalls=[0.3, 0.5, 0.7, 0.9])


def render():
    st.title("🧠 Model Insights")
    st.caption(
        "Why macro-F1 (not accuracy) drives every comparison here, and "
        "why the highest-scoring model wasn't automatically the "
        "production choice."
    )

    try:
        shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Model comparison (5-fold CV, macro-F1)")
    sweep = _model_sweep()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(sweep["model"], sweep["f1_macro"], color=PALETTE[0])
    ax.set_xlabel("Macro-F1")
    for bar, value in zip(bars, sweep["f1_macro"]):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    st.pyplot(fig)
    st.caption(
        "`LightGBM + RandomOverSampler` wins here and is what's saved to "
        "`models/model.joblib` — but an *uncapped* Random Forest scored "
        "even higher (~0.79) in an earlier pass, before `shap.TreeExplainer` "
        "was measured to take over 2 minutes for just 30 rows against it. "
        "Capping its depth (see `model.random_forest_estimator`) trades "
        "that away for a usable, explainable model — the highest raw "
        "score isn't automatically the right production choice."
    )
    st.dataframe(sweep.sort_values("f1_macro", ascending=False), hide_index=True)

    st.subheader("Does class-imbalance handling help?")
    imbalance = _imbalance_comparison()
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    bars2 = ax2.barh(imbalance["approach"], imbalance["f1_macro_mean"], color=[PALETTE[3], PALETTE[2], PALETTE[0]])
    ax2.set_xlabel("Macro-F1")
    for bar, value in zip(bars2, imbalance["f1_macro_mean"]):
        ax2.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    st.pyplot(fig2)
    st.caption(
        "`RandomOverSampler` beats class-weighting here — the opposite "
        "finding from this portfolio's tabular projects. Both are "
        "measured directly rather than one assumed to generalize."
    )

    st.subheader("Precision/recall tradeoff for catching disaster tweets")
    sweep_pr = _threshold_sweep()
    fig_pr, ax_pr = plt.subplots(figsize=(6, 3.5))
    ax_pr.plot(sweep_pr["recall"], sweep_pr["precision"], marker="o", color=PALETTE[0])
    for _, row in sweep_pr.iterrows():
        ax_pr.annotate(f"thr={row['threshold']:.2f}", (row["recall"], row["precision"]), textcoords="offset points", xytext=(6, 4))
    ax_pr.set_xlabel("Recall (share of real disaster tweets caught)")
    ax_pr.set_ylabel("Precision (share of \"Disaster\" alerts that are real)")
    st.pyplot(fig_pr)

    st.subheader("What does the saved model actually rely on? (SHAP)")
    st.caption("Computed live from the saved model — first load takes a few seconds.")
    explanation, _ = shared.get_shap_explanation(sample_size=300)
    top = top_shap_features(explanation, top_n=15).sort_values("mean_abs_shap")

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    ax3.barh(top["feature"], top["mean_abs_shap"], color=PALETTE[0])
    ax3.set_xlabel("Mean |SHAP value|")
    ax3.set_title("Top 15 features")
    st.pyplot(fig3)
    st.caption(
        "`text_length`/`word_count` dominate — real disaster tweets tend "
        "to be longer, news-style reports."
    )
