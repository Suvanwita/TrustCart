from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aspects import extract_aspects
from src.data_loader import filter_by_length, load_pasted_reviews, load_sample_data, load_uploaded_csv
from src.emotion import analyze_emotions
from src.fake_review import analyze_fake_review_risk
from src.sentiment import analyze_sentiment, sentiment_distribution
from src.summarizer import summarize_pros_cons
from src.trust_score import buy_avoid_suggestion, compute_trust_score
from src.utils import SAMPLE_DATA_PATH, save_analysis_artifact, validate_reviews


st.set_page_config(page_title="TrustCart AI", page_icon="TC", layout="wide")


@st.cache_data(show_spinner=False)
def run_analysis(records: tuple[tuple[str, float | None], ...], sensitivity: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_df = pd.DataFrame(records, columns=["review", "rating"])
    reviews = base_df["review"].tolist()
    sentiment_df = analyze_sentiment(reviews).drop(columns=["review"])
    emotion_df = analyze_emotions(reviews).drop(columns=["review"])
    fake_df = analyze_fake_review_risk(reviews, sensitivity=sensitivity).drop(columns=["review"])
    analysis_df = pd.concat([base_df.reset_index(drop=True), sentiment_df, emotion_df, fake_df], axis=1)
    distribution_df = sentiment_distribution(analysis_df)
    return analysis_df, distribution_df


def dataframe_records(df: pd.DataFrame) -> tuple[tuple[str, float | None], ...]:
    records = []
    for row in df.itertuples(index=False):
        rating = None if pd.isna(row.rating) else float(row.rating)
        records.append((str(row.review), rating))
    return tuple(records)


st.title("TrustCart AI")
st.caption("Text-only product review trust analyzer")

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Choose input mode", ["Paste text", "Upload CSV", "Use sample data"])
    sensitivity = st.slider("Fake review sensitivity", 0.5, 2.0, 1.0, 0.1)
    min_length = st.slider("Minimum review length", 0, 250, 20, 5)

raw_df = pd.DataFrame(columns=["review", "rating"])

try:
    if input_mode == "Paste text":
        pasted = st.text_area(
            "Paste reviews",
            height=220,
            placeholder="Paste one review per line...",
        )
        if pasted.strip():
            raw_df = load_pasted_reviews(pasted)
    elif input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV with a review column and optional rating column", type=["csv"])
        if uploaded is not None:
            raw_df = load_uploaded_csv(uploaded)
    else:
        raw_df = load_sample_data(SAMPLE_DATA_PATH)

    filtered_df = filter_by_length(raw_df, min_length)
    st.subheader("Parsed Reviews")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    validate_reviews(filtered_df)

    analysis_df, distribution_df = run_analysis(dataframe_records(filtered_df), sensitivity)
    trust = compute_trust_score(analysis_df)
    aspects_df = extract_aspects(analysis_df["review"].tolist(), top_n=10)
    summaries = summarize_pros_cons(analysis_df)
    suggestion = buy_avoid_suggestion(trust["score"], trust["average_fake_risk"], trust["negative_share"])
    artifact_path = save_analysis_artifact(
        {
            "analysis": analysis_df,
            "sentiment_distribution": distribution_df,
            "aspects": aspects_df,
            "trust": trust,
            "summaries": summaries,
            "suggestion": suggestion,
        }
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Trust Score", f"{trust['score']}/100", trust["label"])
    metric_cols[1].metric("Reviews", len(analysis_df))
    metric_cols[2].metric("Fake Risk", f"{trust['average_fake_risk'] * 100:.1f}%")
    metric_cols[3].metric("Avg Rating", trust["average_rating"] if trust["average_rating"] is not None else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment Distribution")
        fig = px.pie(distribution_df, names="sentiment", values="share", hole=0.45)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Fake Review Risk")
        risk_counts = analysis_df["fake_risk_level"].value_counts().rename_axis("risk").reset_index(name="count")
        fig = px.bar(risk_counts, x="risk", y="count", color="risk", text="count")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Common Product Aspects")
    if aspects_df.empty:
        st.info("No aspects detected.")
    else:
        st.dataframe(aspects_df, use_container_width=True, hide_index=True)

    st.subheader("Pros and Cons Summary")
    pro_col, con_col = st.columns(2)
    pro_col.success(summaries["pros"])
    con_col.warning(summaries["cons"])
    st.info(summaries["summary"])

    st.subheader("Final Suggestion")
    st.write(suggestion)
    st.caption(f"Latest analysis saved to {artifact_path.relative_to(ROOT)}")

    st.subheader("Detailed Review Analysis")
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)

except ValueError as exc:
    st.info(str(exc))
except Exception as exc:
    st.error(f"Unable to analyze reviews: {exc}")
