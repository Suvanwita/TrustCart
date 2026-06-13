from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _summarization_pipeline():
    try:
        from transformers import pipeline

        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None


def _aspect_names(aspects_result: dict, key: str, limit: int = 5) -> list[str]:
    return [str(aspect).title() for aspect in aspects_result.get(key, [])[:limit]]


def _negative_aspect_names(aspects_result: dict, limit: int = 5) -> list[str]:
    return _aspect_names(aspects_result, "negative_aspects", limit=limit)


def _positive_aspect_names(aspects_result: dict, limit: int = 5) -> list[str]:
    return _aspect_names(aspects_result, "positive_aspects", limit=limit)


def _sentiment_counts(reviews_df: pd.DataFrame) -> dict:
    if reviews_df.empty or "sentiment_label" not in reviews_df:
        return {}
    return reviews_df["sentiment_label"].str.lower().value_counts().to_dict()


def _suspicious_patterns(reviews_df: pd.DataFrame) -> list[str]:
    if reviews_df.empty:
        return ["No reviews were available for fake-risk analysis."]

    patterns = []
    high_risk = int((reviews_df["fake_risk_label"] == "High").sum()) if "fake_risk_label" in reviews_df else 0
    medium_risk = int((reviews_df["fake_risk_label"] == "Medium").sum()) if "fake_risk_label" in reviews_df else 0
    fake_source = reviews_df["fake_risk_score"] if "fake_risk_score" in reviews_df else pd.Series([0] * len(reviews_df))
    duplicate_source = reviews_df["duplicate_score"] if "duplicate_score" in reviews_df else pd.Series([0] * len(reviews_df))
    avg_risk = float(pd.to_numeric(fake_source, errors="coerce").fillna(0).mean())
    avg_duplicate = float(pd.to_numeric(duplicate_source, errors="coerce").fillna(0).mean())

    if high_risk:
        patterns.append(f"{high_risk} high-risk review(s) detected.")
    if medium_risk:
        patterns.append(f"{medium_risk} medium-risk review(s) detected.")
    if avg_risk >= 0.35:
        patterns.append(f"Average fake-risk score is elevated at {avg_risk * 100:.1f}%.")
    if avg_duplicate >= 0.55:
        patterns.append(f"Average duplicate similarity is high at {avg_duplicate * 100:.1f}%.")
    return patterns or ["No major suspicious review pattern detected."]


def _hf_verdict(reviews_df: pd.DataFrame) -> str:
    model = _summarization_pipeline()
    if model is None or reviews_df.empty:
        return ""

    review_text = " ".join(reviews_df["review"].dropna().astype(str).head(20).tolist())
    if len(review_text.split()) < 45:
        return ""

    try:
        result = model(review_text[:3500], max_length=70, min_length=20, do_sample=False)
        return str(result[0]["summary_text"]).strip()
    except Exception:
        return ""


def _template_summary(trust_result: dict, aspects_result: dict, reviews_df: pd.DataFrame) -> dict:
    trust_score = float(trust_result.get("trust_score", 0))
    trust_label = str(trust_result.get("trust_label", "Avoid"))
    warnings = list(trust_result.get("warnings", []))
    sentiments = _sentiment_counts(reviews_df)
    positives = _positive_aspect_names(aspects_result)
    negatives = _negative_aspect_names(aspects_result)
    suspicious = _suspicious_patterns(reviews_df)

    positive_count = sentiments.get("positive", 0)
    negative_count = sentiments.get("negative", 0)
    neutral_count = sentiments.get("neutral", 0)

    verdict = (
        f"{trust_label}: TrustCart scores this product {trust_score:.1f}/100 from "
        f"{len(reviews_df)} review(s). Sentiment mix is {positive_count} positive, "
        f"{negative_count} negative, and {neutral_count} neutral."
    )

    pros = []
    if positives:
        pros.append("Positive mentions cluster around " + ", ".join(positives) + ".")
    if positive_count > negative_count:
        pros.append("Positive sentiment outweighs negative sentiment.")
    if "high fake review risk" not in warnings:
        pros.append("Fake-review risk is not the dominant signal.")
    if not pros:
        pros.append("No strong product strengths were detected.")

    cons = []
    if negatives:
        cons.append("Negative mentions cluster around " + ", ".join(negatives) + ".")
    if negative_count >= positive_count and len(reviews_df) > 0:
        cons.append("Negative sentiment is comparable to or stronger than positive sentiment.")
    if warnings:
        cons.append("Warnings: " + "; ".join(warnings) + ".")
    if suspicious and suspicious != ["No major suspicious review pattern detected."]:
        cons.extend(suspicious[:2])
    if not cons:
        cons.append("No major product drawbacks were detected from the available reviews.")

    buy_for = []
    if positives:
        buy_for.append("Buyers who care about " + ", ".join(positives[:3]) + ".")
    if trust_score >= 65:
        buy_for.append("Shoppers comfortable with the current trust signal.")
    if not buy_for:
        buy_for.append("Buyers who can verify the product with additional sources.")

    avoid_if = []
    if negatives:
        avoid_if.append("Avoid if you are sensitive to issues around " + ", ".join(negatives[:3]) + ".")
    if warnings:
        avoid_if.append("Avoid if you need very clean, low-risk review signals.")
    if trust_score < 50:
        avoid_if.append("Avoid if this is a high-value or hard-to-return purchase.")
    if not avoid_if:
        avoid_if.append("Avoid if the listed pros do not match your main use case.")

    return {
        "verdict": verdict,
        "pros": pros,
        "cons": cons,
        "buy_for": buy_for,
        "avoid_if": avoid_if,
    }


def generate_summary(trust_result: dict, aspects_result: dict, reviews_df: pd.DataFrame) -> dict:
    summary = _template_summary(trust_result, aspects_result, reviews_df)
    hf_verdict = _hf_verdict(reviews_df)
    if hf_verdict:
        summary["verdict"] = f"{summary['verdict']} Local model summary: {hf_verdict}"
    return summary


def summarize_pros_cons(analysis_df: pd.DataFrame) -> dict:
    """Backward-compatible adapter for older app code."""
    fallback = generate_summary(
        {"trust_score": 0, "trust_label": "Review Summary", "warnings": []},
        {"aspects_table": pd.DataFrame(), "positive_aspects": [], "negative_aspects": []},
        analysis_df,
    )
    return {
        "pros": " ".join(fallback["pros"]),
        "cons": " ".join(fallback["cons"]),
        "summary": fallback["verdict"],
    }
