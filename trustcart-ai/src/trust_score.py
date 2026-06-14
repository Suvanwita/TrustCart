from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _label(score: float) -> str:
    if score >= 80:
        return "Strong Buy"
    if score >= 65:
        return "Buy with Caution"
    if score >= 50:
        return "Mixed / Compare Alternatives"
    return "Avoid"


def _signed_sentiment(row: pd.Series) -> float:
    label = str(row.get("sentiment_label", "neutral")).lower()
    score = float(row.get("sentiment_score", 0) or 0)
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def _positive_sentiment_score(reviews_df: pd.DataFrame) -> float:
    if reviews_df.empty:
        return 0.0
    signed = reviews_df.apply(_signed_sentiment, axis=1)
    return float(np.clip(((signed.mean() + 1) / 2) * 100, 0, 100))


def _aspect_quality_score(aspects_result: dict) -> tuple[float, bool]:
    aspects_table = aspects_result.get("aspects_table", pd.DataFrame())
    if aspects_table.empty or "average_sentiment_score" not in aspects_table:
        return 50.0, False

    weighted = aspects_table["average_sentiment_score"] * aspects_table["mention_count"].clip(lower=1)
    average = weighted.sum() / aspects_table["mention_count"].clip(lower=1).sum()
    score = float(np.clip(((average + 1) / 2) * 100, 0, 100))

    positive_count = len(aspects_result.get("positive_aspects", []))
    negative_count = len(aspects_result.get("negative_aspects", []))
    negative_dominate = negative_count > positive_count and negative_count >= 2
    return score, negative_dominate


def _rating_consistency_score(reviews_df: pd.DataFrame, ratings: list[int] | None) -> float:
    reviews_df = reviews_df.reset_index(drop=True)
    rating_values = ratings
    if rating_values is None and "rating" in reviews_df.columns:
        rating_values = reviews_df["rating"].tolist()
    if not rating_values:
        return 60.0

    ratings_series = pd.to_numeric(pd.Series(rating_values), errors="coerce").dropna()
    if ratings_series.empty:
        return 60.0

    normalized_ratings = ((ratings_series - 1) / 4).clip(0, 1)
    sentiment = reviews_df.loc[ratings_series.index].apply(_signed_sentiment, axis=1)
    normalized_sentiment = ((sentiment + 1) / 2).clip(0, 1)
    mismatch = (normalized_ratings - normalized_sentiment).abs().mean()
    variance_penalty = min(float(ratings_series.std(ddof=0) or 0) / 2.0, 1.0)
    score = 100 - (mismatch * 70 + variance_penalty * 30)
    return float(np.clip(score, 0, 100))


def _review_volume_score(review_count: int) -> float:
    if review_count <= 0:
        return 0.0
    return float(np.clip(math.log1p(review_count) / math.log1p(50) * 100, 0, 100))


def _authenticity_score(reviews_df: pd.DataFrame) -> tuple[float, float, float]:
    if reviews_df.empty:
        return 0.0, 0.0, 0.0

    fake_source = reviews_df["fake_risk_score"] if "fake_risk_score" in reviews_df else pd.Series([0] * len(reviews_df))
    duplicate_source = reviews_df["duplicate_score"] if "duplicate_score" in reviews_df else pd.Series([0] * len(reviews_df))
    fake_risk = pd.to_numeric(fake_source, errors="coerce").fillna(0)
    duplicate_scores = pd.to_numeric(duplicate_source, errors="coerce").fillna(0)
    average_fake_risk = float(fake_risk.mean() * 100)
    average_duplicate_score = float(duplicate_scores.mean() * 100)
    score = 100 - average_fake_risk
    return float(np.clip(score, 0, 100)), average_fake_risk, average_duplicate_score


def _negative_aspect_phrase(aspects_result: dict) -> str:
    negative_aspects = [str(aspect).lower() for aspect in aspects_result.get("negative_aspects", [])[:3]]
    if not negative_aspects:
        return "Negative aspects dominate"
    if len(negative_aspects) == 1:
        return f"Negative {negative_aspects[0]} aspect dominates"
    return f"Negative {'/'.join(negative_aspects)} aspects dominate"


def _confidence_notes(
    positive_sentiment: float,
    aspect_quality: float,
    rating_consistency: float,
    review_volume: float,
    authenticity: float,
    average_fake_risk: float,
    average_duplicate_score: float,
    negative_aspects_dominate: bool,
    aspects_result: dict,
) -> list[str]:
    notes: list[str] = []

    if positive_sentiment >= 70 and review_volume < 55:
        notes.append("Good sentiment but weak review volume")
    elif positive_sentiment >= 70:
        notes.append("Good sentiment supports the score")
    elif positive_sentiment < 45:
        notes.append("Weak sentiment pulls the score down")

    if average_fake_risk >= 35:
        notes.append("High fake-risk phrases detected")
    elif authenticity >= 75:
        notes.append("Low fake-risk signals support authenticity")

    if negative_aspects_dominate or aspect_quality < 45:
        notes.append(_negative_aspect_phrase(aspects_result))
    elif aspect_quality >= 70:
        notes.append("Positive product aspects support the score")

    if rating_consistency < 55:
        notes.append("Ratings and review sentiment are not fully consistent")
    elif rating_consistency >= 75:
        notes.append("Ratings are broadly consistent with review sentiment")

    if average_duplicate_score >= 55:
        notes.append("Duplicate-like review patterns reduce confidence")

    if review_volume < 45:
        notes.append("Too few reviews to be highly confident")

    return notes[:6] or ["Balanced signals; no single factor dominates the score"]


def calculate_trust_score(
    reviews_df: pd.DataFrame,
    aspects_result: dict,
    ratings: list[int] | None = None,
) -> dict:
    if reviews_df.empty:
        return {
            "trust_score": 0.0,
            "trust_label": "Avoid",
            "components": {
                "positive_sentiment_score": 0.0,
                "aspect_quality_score": 0.0,
                "rating_consistency_score": 0.0,
                "review_volume_score": 0.0,
                "authenticity_score": 0.0,
            },
            "warnings": ["too few reviews"],
            "confidence_notes": ["Too few reviews to be highly confident"],
        }

    positive_sentiment = _positive_sentiment_score(reviews_df)
    aspect_quality, negative_aspects_dominate = _aspect_quality_score(aspects_result)
    rating_consistency = _rating_consistency_score(reviews_df, ratings)
    review_volume = _review_volume_score(len(reviews_df))
    authenticity, average_fake_risk, average_duplicate_score = _authenticity_score(reviews_df)

    trust_score = (
        0.30 * positive_sentiment
        + 0.20 * aspect_quality
        + 0.15 * rating_consistency
        + 0.15 * review_volume
        + 0.20 * authenticity
    )
    trust_score = round(float(np.clip(trust_score, 0, 100)), 2)

    warnings: list[str] = []
    if average_fake_risk >= 45:
        warnings.append("high fake review risk")
    duplicate_source = reviews_df["duplicate_score"] if "duplicate_score" in reviews_df else pd.Series([0] * len(reviews_df))
    duplicate_scores = pd.to_numeric(duplicate_source, errors="coerce").fillna(0)
    if average_duplicate_score >= 65 or (duplicate_scores >= 0.88).mean() >= 0.2:
        warnings.append("too many duplicate reviews")
    if negative_aspects_dominate:
        warnings.append("negative aspects dominate")
    if len(reviews_df) < 5:
        warnings.append("too few reviews")

    confidence_notes = _confidence_notes(
        positive_sentiment=positive_sentiment,
        aspect_quality=aspect_quality,
        rating_consistency=rating_consistency,
        review_volume=review_volume,
        authenticity=authenticity,
        average_fake_risk=average_fake_risk,
        average_duplicate_score=average_duplicate_score,
        negative_aspects_dominate=negative_aspects_dominate,
        aspects_result=aspects_result,
    )

    return {
        "trust_score": trust_score,
        "trust_label": _label(trust_score),
        "components": {
            "positive_sentiment_score": round(positive_sentiment, 2),
            "aspect_quality_score": round(aspect_quality, 2),
            "rating_consistency_score": round(rating_consistency, 2),
            "review_volume_score": round(review_volume, 2),
            "authenticity_score": round(authenticity, 2),
        },
        "warnings": warnings,
        "confidence_notes": confidence_notes,
    }


def compute_trust_score(analysis_df: pd.DataFrame) -> dict:
    """Backward-compatible wrapper for older code paths."""
    trust = calculate_trust_score(analysis_df, {"aspects_table": pd.DataFrame(), "positive_aspects": [], "negative_aspects": []})
    fake_source = analysis_df["fake_risk_score"] if "fake_risk_score" in analysis_df else pd.Series([0] * len(analysis_df))
    rating_source = analysis_df["rating"] if "rating" in analysis_df else pd.Series(dtype=float)
    fake_risk = pd.to_numeric(fake_source, errors="coerce").fillna(0)
    rating = pd.to_numeric(rating_source, errors="coerce").dropna()
    sentiment_col = "sentiment_label" if "sentiment_label" in analysis_df.columns else "sentiment"
    return {
        "score": trust["trust_score"],
        "label": trust["trust_label"],
        "positive_share": round(float((analysis_df[sentiment_col] == "positive").mean()), 3) if not analysis_df.empty else 0,
        "negative_share": round(float((analysis_df[sentiment_col] == "negative").mean()), 3) if not analysis_df.empty else 0,
        "average_fake_risk": round(float(fake_risk.mean()), 3) if not analysis_df.empty else 0,
        "average_rating": None if rating.empty else round(float(rating.mean()), 2),
    }


def buy_avoid_suggestion(score: float, fake_risk: float, negative_share: float) -> str:
    label = _label(score)
    if label == "Strong Buy":
        return "Strong Buy: review signals are trustworthy and broadly positive."
    if label == "Buy with Caution":
        return "Buy with Caution: review signals are mostly favorable, but check the warnings."
    if label == "Mixed / Compare Alternatives":
        return "Mixed / Compare Alternatives: compare against similar products before deciding."
    return "Avoid: trust and review-quality signals are weak."
