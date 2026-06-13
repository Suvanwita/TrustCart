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
from src.emotion import analyze_emotion_with_pipeline, load_emotion_pipeline
from src.fake_review import detect_fake_reviews
from src.sentiment import analyze_sentiment_with_pipeline, load_sentiment_pipeline, sentiment_distribution
from src.summarizer import summarize_pros_cons
from src.trust_score import calculate_trust_score
from src.utils import SAMPLE_DATA_PATH, save_analysis_artifact, validate_reviews


st.set_page_config(page_title="TrustCart AI", page_icon="TC", layout="wide")


@st.cache_data(show_spinner=False)
def run_analysis(records: tuple[tuple[str, float | None], ...], sensitivity: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_df = pd.DataFrame(records, columns=["review", "rating"])
    reviews = base_df["review"].tolist()
    sentiment_df = analyze_sentiment_with_pipeline(reviews, get_sentiment_model()).drop(columns=["review"])
    emotion_df = analyze_emotion_with_pipeline(reviews, get_emotion_model()).drop(columns=["review"])
    fake_df = detect_fake_reviews(reviews, ratings=base_df["rating"].tolist(), sensitivity=sensitivity).drop(columns=["review"])
    analysis_df = pd.concat([base_df.reset_index(drop=True), sentiment_df, emotion_df, fake_df], axis=1)
    distribution_df = sentiment_distribution(analysis_df)
    return analysis_df, distribution_df


@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return load_sentiment_pipeline()


@st.cache_resource(show_spinner=False)
def get_emotion_model():
    return load_emotion_pipeline()


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
    sensitivity = st.slider("Fake review sensitivity", 0.0, 1.0, 0.5, 0.05)
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
    aspects_result = extract_aspects(analysis_df)
    aspects_df = aspects_result["aspects_table"]
    trust = calculate_trust_score(analysis_df, aspects_result, ratings=analysis_df["rating"].tolist())
    summaries = summarize_pros_cons(analysis_df)
    artifact_path = save_analysis_artifact(
        {
            "analysis": analysis_df,
            "sentiment_distribution": distribution_df,
            "aspects": aspects_result,
            "trust": trust,
            "summaries": summaries,
            "suggestion": trust["trust_label"],
        }
    )

    st.subheader("Trust Score")
    score_col, decision_col = st.columns([2, 1])
    with score_col:
        st.markdown(
            f"""
            <div style="border:1px solid #ddd;border-radius:8px;padding:22px 24px;">
                <div style="font-size:15px;color:#666;">Trust Score</div>
                <div style="font-size:48px;font-weight:700;line-height:1;">{trust['trust_score']:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with decision_col:
        st.markdown("**Final Decision**")
        if trust["trust_label"] == "Strong Buy":
            st.success(trust["trust_label"])
        elif trust["trust_label"] == "Buy with Caution":
            st.warning(trust["trust_label"])
        elif trust["trust_label"] == "Mixed / Compare Alternatives":
            st.info(trust["trust_label"])
        else:
            st.error(trust["trust_label"])

    component_df = pd.DataFrame(
        [{"component": name.replace("_", " ").title(), "score": score} for name, score in trust["components"].items()]
    )
    fig = px.bar(component_df, x="component", y="score", text="score")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if trust["warnings"]:
        st.warning("Warnings: " + "; ".join(trust["warnings"]))
    else:
        st.success("No major trust warnings detected.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Reviews", len(analysis_df))
    metric_cols[1].metric("Average Fake Risk", f"{analysis_df['fake_risk_score'].mean() * 100:.1f}%")
    avg_rating = pd.to_numeric(analysis_df["rating"], errors="coerce").dropna()
    metric_cols[2].metric("Avg Rating", f"{avg_rating.mean():.2f}" if not avg_rating.empty else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment Distribution")
        fig = px.bar(distribution_df, x="sentiment", y="share", color="sentiment", text="percent")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Emotion Distribution")
        emotion_counts = (
            analysis_df["emotion_label"].value_counts(normalize=True).rename_axis("emotion").reset_index(name="share")
        )
        emotion_counts["percent"] = (emotion_counts["share"] * 100).round(1)
        fig = px.bar(emotion_counts, x="emotion", y="share", color="emotion", text="percent")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment and Emotion")
    st.dataframe(
        analysis_df[["review", "sentiment_label", "sentiment_score", "emotion_label", "emotion_score"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Fake Review Risk")
    high_risk_count = int((analysis_df["fake_risk_label"] == "High").sum())
    medium_risk_count = int((analysis_df["fake_risk_label"] == "Medium").sum())
    average_fake_risk = analysis_df["fake_risk_score"].mean()
    fake_cols = st.columns(3)
    fake_cols[0].metric("High Risk Reviews", high_risk_count)
    fake_cols[1].metric("Medium Risk Reviews", medium_risk_count)
    fake_cols[2].metric("Average Fake Risk", f"{average_fake_risk * 100:.1f}%")

    risk_counts = analysis_df["fake_risk_label"].value_counts().rename_axis("risk").reset_index(name="count")
    fig = px.bar(risk_counts, x="risk", y="count", color="risk", text="count")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Suspicious Reviews")
    suspicious_df = analysis_df.sort_values("fake_risk_score", ascending=False).head(5)
    st.dataframe(
        suspicious_df[
            [
                "review",
                "rating",
                "fake_risk_score",
                "fake_risk_label",
                "duplicate_score",
                "anomaly_score",
                "reasons",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Analysis Health"):
        st.write(
            "Sentiment model:",
            "Hugging Face pipeline" if get_sentiment_model() is not None else "rule-based fallback",
        )
        st.write(
            "Emotion model:",
            "Hugging Face pipeline" if get_emotion_model() is not None else "rule-based fallback",
        )

    st.subheader("Common Product Aspects")
    if aspects_df.empty:
        st.info("No aspects detected.")
    else:
        st.dataframe(aspects_df, use_container_width=True, hide_index=True)
        aspect_cols = st.columns(2)
        with aspect_cols[0]:
            st.markdown("**Top Positive Aspects**")
            if aspects_result["positive_aspects"]:
                for aspect in aspects_result["positive_aspects"]:
                    st.success(aspect.title())
            else:
                st.info("No clearly positive aspects detected.")
        with aspect_cols[1]:
            st.markdown("**Top Negative Aspects**")
            if aspects_result["negative_aspects"]:
                for aspect in aspects_result["negative_aspects"]:
                    st.warning(aspect.title())
            else:
                st.info("No clearly negative aspects detected.")

        st.subheader("Aspect Review Samples")
        for row in aspects_df.head(6).itertuples(index=False):
            with st.expander(f"{row.aspect.title()} ({row.mention_count} mentions)"):
                if row.positive_review_examples:
                    st.markdown("**Positive examples**")
                    for example in str(row.positive_review_examples).split(" | "):
                        st.write(example)
                if row.negative_review_examples:
                    st.markdown("**Negative examples**")
                    for example in str(row.negative_review_examples).split(" | "):
                        st.write(example)
                if not row.positive_review_examples and not row.negative_review_examples:
                    st.write("No positive or negative samples available for this aspect.")

    st.subheader("Pros and Cons Summary")
    pro_col, con_col = st.columns(2)
    pro_col.success(summaries["pros"])
    con_col.warning(summaries["cons"])
    st.info(summaries["summary"])

    st.subheader("Final Suggestion")
    st.write(trust["trust_label"])
    st.caption(f"Latest analysis saved to {artifact_path.relative_to(ROOT)}")

    st.subheader("Detailed Review Analysis")
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)

except ValueError as exc:
    st.info(str(exc))
except Exception as exc:
    st.error(f"Unable to analyze reviews: {exc}")
