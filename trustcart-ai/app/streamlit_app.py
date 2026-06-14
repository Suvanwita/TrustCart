from __future__ import annotations

import json
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
from src.summarizer import generate_summary
from src.trust_score import calculate_trust_score
from src.utils import SAMPLE_DATA_PATH, save_analysis_artifact, validate_reviews


st.set_page_config(page_title="TrustCart AI", page_icon="TC", layout="wide")

CHART_COLORS = ["#ff4d8d", "#20e3b2", "#ffd166", "#9b5cff", "#00bbf9", "#ff8a3d"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --tc-ink: #f8fbff;
                --tc-muted: #a9b8d4;
                --tc-line: rgba(157, 188, 255, .22);
                --tc-surface: #121a2d;
                --tc-soft: #17223a;
                --tc-coral: #ff4d8d;
                --tc-teal: #20e3b2;
                --tc-mango: #ffd166;
                --tc-violet: #9b5cff;
                --tc-mint: #2ee59d;
                --tc-sky: #00bbf9;
                --tc-orange: #ff8a3d;
            }

            .stApp {
                background:
                    radial-gradient(circle at 18% 8%, rgba(255, 77, 141, .25), transparent 28%),
                    radial-gradient(circle at 83% 0%, rgba(32, 227, 178, .18), transparent 24%),
                    radial-gradient(circle at 74% 54%, rgba(155, 92, 255, .16), transparent 28%),
                    linear-gradient(180deg, #080c18 0%, #0d1324 42%, #111827 100%);
                color: var(--tc-ink);
            }

            .stApp, .stMarkdown, p, li, label, span, div {
                color: var(--tc-ink);
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, #12081f 0%, #172554 48%, #052e2b 100%);
            }

            [data-testid="stSidebar"] * {
                color: #f9fafb;
            }

            [data-testid="stSidebar"] [data-baseweb="radio"] label,
            [data-testid="stSidebar"] .stSlider label,
            [data-testid="stSidebar"] .stFileUploader label {
                color: #f9fafb !important;
            }

            .block-container {
                padding-top: 2.4rem;
                max-width: 1240px;
            }

            h2, h3 {
                letter-spacing: 0;
                color: var(--tc-ink);
            }

            .tc-hero-card {
                border: 1px solid rgba(32, 227, 178, .30);
                background:
                    linear-gradient(135deg, rgba(18, 26, 45, .98) 0%, rgba(49, 28, 79, .96) 48%, rgba(5, 52, 59, .96) 100%);
                border-radius: 18px;
                padding: 1.5rem;
                margin: 1.75rem auto 1.6rem;
                max-width: 1120px;
                box-shadow: 0 18px 50px rgba(0, 0, 0, .32);
            }

            .tc-hero-kicker {
                color: var(--tc-mango);
                font-size: 12px;
                font-weight: 800;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .tc-hero-title {
                color: transparent;
                background: linear-gradient(90deg, var(--tc-coral), var(--tc-mango), var(--tc-teal), var(--tc-sky));
                -webkit-background-clip: text;
                background-clip: text;
                font-size: clamp(2.1rem, 4vw, 3.6rem);
                line-height: 1.02;
                font-weight: 850;
                margin: 0;
            }

            .tc-hero-subtitle {
                color: #e9fbf8;
                font-size: 1rem;
                line-height: 1.45;
                margin: .75rem 0 0;
                max-width: 820px;
            }

            .tc-hero-note {
                color: #a9f4e3;
                font-size: .92rem;
                margin: .6rem 0 0;
            }

            .tc-score-card {
                border: 1px solid rgba(255, 77, 141, .35);
                background: linear-gradient(135deg, rgba(18, 26, 45, .98) 0%, rgba(49, 28, 79, .96) 52%, rgba(5, 52, 59, .96) 100%);
                border-radius: 8px;
                padding: 24px;
                box-shadow: 0 16px 40px rgba(255, 77, 141, 0.16);
                min-height: 150px;
            }

            .tc-score-label {
                color: var(--tc-muted);
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: .08em;
            }

            .tc-score-value {
                color: transparent;
                background: linear-gradient(90deg, var(--tc-coral), var(--tc-orange), var(--tc-teal));
                -webkit-background-clip: text;
                background-clip: text;
                font-size: 56px;
                font-weight: 800;
                line-height: 1;
                margin-top: 8px;
            }

            .tc-card {
                border: 1px solid var(--tc-line);
                background: linear-gradient(135deg, rgba(18, 26, 45, .96) 0%, rgba(20, 34, 55, .96) 100%);
                border-radius: 8px;
                padding: 18px 20px;
                margin: 8px 0 14px;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
            }

            .tc-badge {
                display: inline-block;
                border-radius: 999px;
                padding: 8px 13px;
                font-size: 13px;
                font-weight: 750;
                border: 1px solid transparent;
            }

            .tc-badge-strong { background: rgba(46, 229, 157, .15); color: #7dffc7; border-color: rgba(46, 229, 157, .46); }
            .tc-badge-caution { background: rgba(255, 209, 102, .15); color: #ffe39a; border-color: rgba(255, 209, 102, .5); }
            .tc-badge-mixed { background: rgba(155, 92, 255, .17); color: #d2b9ff; border-color: rgba(155, 92, 255, .5); }
            .tc-badge-avoid { background: rgba(255, 77, 141, .16); color: #ffaccb; border-color: rgba(255, 77, 141, .55); }

            .tc-list-card {
                border: 1px solid rgba(0, 166, 166, .18);
                border-radius: 8px;
                background: linear-gradient(180deg, rgba(18, 26, 45, .96) 0%, rgba(10, 35, 45, .94) 100%);
                padding: 16px 18px;
                min-height: 190px;
            }

            .tc-list-card h4 {
                margin: 0 0 10px;
                font-size: 15px;
            }

            .tc-list-card ul {
                margin: 0;
                padding-left: 18px;
            }

            .tc-list-card li {
                margin-bottom: 8px;
                color: var(--tc-ink);
            }

            [data-testid="stMetric"] {
                background: linear-gradient(135deg, rgba(18, 26, 45, .96) 0%, rgba(38, 30, 55, .94) 100%);
                border: 1px solid rgba(255, 209, 102, .30);
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: 0 6px 18px rgba(16, 24, 40, 0.05);
            }

            .stDataFrame {
                border: 1px solid var(--tc-line);
                border-radius: 8px;
                overflow: hidden;
            }

            div[data-testid="stExpander"] {
                border: 1px solid var(--tc-line);
                border-radius: 8px;
                background: var(--tc-surface);
            }

            .stTextArea textarea,
            .stTextInput input,
            [data-baseweb="select"] > div {
                background: #0b1220 !important;
                color: var(--tc-ink) !important;
                border-color: rgba(32, 227, 178, .35) !important;
            }

            .stFileUploader {
                border: 1px dashed rgba(32, 227, 178, .35);
                border-radius: 8px;
                padding: 8px;
                background: rgba(18, 26, 45, .45);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def decision_badge(label: str) -> str:
    badge_class = {
        "Strong Buy": "tc-badge-strong",
        "Buy with Caution": "tc-badge-caution",
        "Mixed / Compare Alternatives": "tc-badge-mixed",
        "Avoid": "tc-badge-avoid",
    }.get(label, "tc-badge-mixed")
    return f'<span class="tc-badge {badge_class}">{label}</span>'


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


def report_to_csv(report: dict) -> str:
    rows = []
    for key, value in report.items():
        if isinstance(value, list):
            value = " | ".join(str(item) for item in value)
        elif isinstance(value, dict):
            value = json.dumps(value)
        rows.append({"section": key, "value": value})
    return pd.DataFrame(rows).to_csv(index=False)


inject_styles()

st.markdown(
    """
    <section class="tc-hero-card">
        <div class="tc-hero-kicker">REVIEW TRUST INTELLIGENCE</div>
        <h1 class="tc-hero-title">TrustCart AI</h1>
        <p class="tc-hero-subtitle">
            Local sentiment, fake-risk, aspect, and buy-or-avoid analysis.
        </p>
        <p class="tc-hero-note">
            Analyze product reviews and generate a trust score in seconds.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Review input source", ["Paste text", "Upload CSV", "Use sample data"])
    sensitivity = st.slider("Fake review sensitivity", 0.0, 1.0, 0.5, 0.05)
    min_length = st.slider("Minimum review length (characters)", 0, 250, 0, 5)

raw_df = pd.DataFrame(columns=["review", "rating"])

try:
    if input_mode == "Paste text":
        pasted = st.text_area(
            "Product reviews to analyze",
            height=220,
            placeholder="Paste one review per line...",
        )
        if pasted.strip():
            raw_df = load_pasted_reviews(pasted)
    elif input_mode == "Upload CSV":
        uploaded = st.file_uploader("Reviews CSV file (required: review, optional: rating)", type=["csv"])
        if uploaded is not None:
            raw_df = load_uploaded_csv(uploaded)
    else:
        raw_df = load_sample_data(SAMPLE_DATA_PATH)

    filtered_df = filter_by_length(raw_df, min_length)
    st.subheader("Parsed Reviews")
    st.dataframe(filtered_df, width="stretch", hide_index=True)

    validate_reviews(filtered_df)

    analysis_df, distribution_df = run_analysis(dataframe_records(filtered_df), sensitivity)
    aspects_result = extract_aspects(analysis_df)
    aspects_df = aspects_result["aspects_table"]
    trust = calculate_trust_score(analysis_df, aspects_result, ratings=analysis_df["rating"].tolist())
    final_summary = generate_summary(trust, aspects_result, analysis_df)
    suspicious_patterns = [
        reason
        for reason in analysis_df.sort_values("fake_risk_score", ascending=False)["reasons"].dropna().astype(str).tolist()
        if reason != "no strong fake/spam signals"
    ][:5]
    if not suspicious_patterns:
        suspicious_patterns = ["No major suspicious review pattern detected."]
    report = {
        "trust_score": trust["trust_score"],
        "trust_label": trust["trust_label"],
        "verdict": final_summary["verdict"],
        "pros": final_summary["pros"],
        "cons": final_summary["cons"],
        "suspicious_review_patterns": suspicious_patterns,
        "buy_for": final_summary["buy_for"],
        "avoid_if": final_summary["avoid_if"],
        "warnings": trust["warnings"],
        "confidence_notes": trust["confidence_notes"],
        "components": trust["components"],
    }
    artifact_path = save_analysis_artifact(
        {
            "analysis": analysis_df,
            "sentiment_distribution": distribution_df,
            "aspects": aspects_result,
            "trust": trust,
            "summaries": final_summary,
            "report": report,
            "suggestion": trust["trust_label"],
        }
    )

    st.subheader("Trust Score")
    score_col, decision_col = st.columns([2, 1])
    with score_col:
        st.markdown(
            f"""
            <div class="tc-score-card">
                <div class="tc-score-label">Trust Score</div>
                <div class="tc-score-value">{trust['trust_score']:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with decision_col:
        st.markdown("**Final Decision**")
        st.markdown(decision_badge(trust["trust_label"]), unsafe_allow_html=True)

    component_df = pd.DataFrame(
        [{"component": name.replace("_", " ").title(), "score": score} for name, score in trust["components"].items()]
    )
    fig = px.bar(component_df, x="component", y="score", text="score", color="component", color_discrete_sequence=CHART_COLORS)
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    warning_text = "; ".join(trust["warnings"]) if trust["warnings"] else "No major trust warnings detected."
    st.markdown(f'<div class="tc-card"><strong>Warnings</strong><br>{warning_text}</div>', unsafe_allow_html=True)
    confidence_items = "".join(f"<li>{note}</li>" for note in trust["confidence_notes"])
    st.markdown(
        f'<div class="tc-card"><strong>Why this score?</strong><ul>{confidence_items}</ul></div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Reviews", len(analysis_df))
    metric_cols[1].metric("Average Fake Risk", f"{analysis_df['fake_risk_score'].mean() * 100:.1f}%")
    avg_rating = pd.to_numeric(analysis_df["rating"], errors="coerce").dropna()
    metric_cols[2].metric("Avg Rating", f"{avg_rating.mean():.2f}" if not avg_rating.empty else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment Distribution")
        fig = px.bar(
            distribution_df,
            x="sentiment",
            y="share",
            color="sentiment",
            text="percent",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Emotion Distribution")
        emotion_counts = (
            analysis_df["emotion_label"].value_counts(normalize=True).rename_axis("emotion").reset_index(name="share")
        )
        emotion_counts["percent"] = (emotion_counts["share"] * 100).round(1)
        fig = px.bar(
            emotion_counts,
            x="emotion",
            y="share",
            color="emotion",
            text="percent",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Sentiment and Emotion")
    st.dataframe(
        analysis_df[["review", "sentiment_label", "sentiment_score", "emotion_label", "emotion_score"]],
        width="stretch",
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
    fig = px.bar(risk_counts, x="risk", y="count", color="risk", text="count", color_discrete_sequence=CHART_COLORS)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

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
        width="stretch",
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
        st.dataframe(aspects_df, width="stretch", hide_index=True)
        aspect_cols = st.columns(2)
        with aspect_cols[0]:
            positive_items = "".join(f"<li>{aspect.title()}</li>" for aspect in aspects_result["positive_aspects"])
            if not positive_items:
                positive_items = "<li>No clearly positive aspects detected.</li>"
            st.markdown(
                f'<div class="tc-list-card"><h4>Top Positive Aspects</h4><ul>{positive_items}</ul></div>',
                unsafe_allow_html=True,
            )
        with aspect_cols[1]:
            negative_items = "".join(f"<li>{aspect.title()}</li>" for aspect in aspects_result["negative_aspects"])
            if not negative_items:
                negative_items = "<li>No clearly negative aspects detected.</li>"
            st.markdown(
                f'<div class="tc-list-card"><h4>Top Negative Aspects</h4><ul>{negative_items}</ul></div>',
                unsafe_allow_html=True,
            )

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

    st.subheader("Final Report")
    st.markdown(
        f"""
        <div class="tc-card">
            <div class="tc-score-label">Report Verdict</div>
            <div style="margin:10px 0 12px;">{decision_badge(trust["trust_label"])}</div>
            <div style="font-size:18px;font-weight:700;margin-bottom:6px;">{trust['trust_score']:.1f}/100</div>
            <div style="color:var(--tc-muted);">{final_summary["verdict"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    report_cols = st.columns(2)
    with report_cols[0]:
        pros = "".join(f"<li>{item}</li>" for item in final_summary["pros"])
        buy_for = "".join(f"<li>{item}</li>" for item in final_summary["buy_for"])
        st.markdown(
            f'<div class="tc-list-card"><h4>Pros</h4><ul>{pros}</ul><h4>Who Should Buy</h4><ul>{buy_for}</ul></div>',
            unsafe_allow_html=True,
        )
    with report_cols[1]:
        cons = "".join(f"<li>{item}</li>" for item in final_summary["cons"])
        avoid_if = "".join(f"<li>{item}</li>" for item in final_summary["avoid_if"])
        st.markdown(
            f'<div class="tc-list-card"><h4>Cons</h4><ul>{cons}</ul><h4>Who Should Avoid</h4><ul>{avoid_if}</ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Suspicious Review Patterns**")
    patterns = "".join(f"<li>{pattern}</li>" for pattern in suspicious_patterns)
    st.markdown(f'<div class="tc-card"><ul>{patterns}</ul></div>', unsafe_allow_html=True)

    st.markdown("**Confidence Notes**")
    final_confidence_items = "".join(f"<li>{note}</li>" for note in trust["confidence_notes"])
    st.markdown(f'<div class="tc-card"><ul>{final_confidence_items}</ul></div>', unsafe_allow_html=True)

    st.markdown("**Buy/Avoid Recommendation**")
    st.markdown(decision_badge(trust["trust_label"]), unsafe_allow_html=True)

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download Report JSON",
        data=json.dumps(report, indent=2),
        file_name="trustcart_report.json",
        mime="application/json",
    )
    download_cols[1].download_button(
        "Download Report CSV",
        data=report_to_csv(report),
        file_name="trustcart_report.csv",
        mime="text/csv",
    )

    st.caption(f"Latest analysis saved to {artifact_path.relative_to(ROOT)}")

    st.subheader("Detailed Review Analysis")
    st.dataframe(analysis_df, width="stretch", hide_index=True)

except ValueError as exc:
    st.info(str(exc))
except Exception as exc:
    st.error(f"Unable to analyze reviews: {exc}")
