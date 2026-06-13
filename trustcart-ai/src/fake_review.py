from __future__ import annotations

from functools import lru_cache
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity


GENERIC_PHRASES = {
    "best product",
    "best ever",
    "highly recommend",
    "highly recommended",
    "must buy",
    "perfect product",
    "trust me",
    "worst ever",
    "worst product",
}


@lru_cache(maxsize=1)
def _sentence_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _rating_extremity(rating) -> float:
    if rating is None or pd.isna(rating):
        return 0.0
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return 0.0
    if value <= 1 or value >= 5:
        return 1.0
    if value <= 2 or value >= 4:
        return 0.5
    return 0.0


def _generic_phrase_count(text: str) -> int:
    lowered = text.lower()
    return sum(1 for phrase in GENERIC_PHRASES if phrase in lowered)


def _engineer_features(reviews: list[str], ratings: list[int] | None) -> pd.DataFrame:
    rows = []
    safe_ratings = list(ratings or [])
    if len(safe_ratings) < len(reviews):
        safe_ratings.extend([None] * (len(reviews) - len(safe_ratings)))
    safe_ratings = safe_ratings[: len(reviews)]

    for review, rating in zip(reviews, safe_ratings):
        clean = review.strip()
        words = _tokens(clean)
        word_count = len(words)
        unique_count = len(set(words))
        repeated_words_ratio = 1 - (unique_count / max(word_count, 1))
        letter_count = sum(char.isalpha() for char in clean)
        uppercase_ratio = sum(char.isupper() for char in clean) / max(letter_count, 1)
        exclamation_count = clean.count("!")
        phrase_count = _generic_phrase_count(clean)

        rows.append(
            {
                "review_length": word_count,
                "exclamation_count": exclamation_count,
                "uppercase_ratio": uppercase_ratio,
                "repeated_words_ratio": repeated_words_ratio,
                "rating_extremity": _rating_extremity(rating),
                "generic_phrase_count": phrase_count,
            }
        )

    return pd.DataFrame(rows)


def _duplicate_scores(reviews: list[str]) -> np.ndarray:
    if len(reviews) < 2:
        return np.zeros(len(reviews))

    model = _sentence_model()
    if model is None:
        return np.zeros(len(reviews))

    try:
        embeddings = model.encode(reviews, normalize_embeddings=True, show_progress_bar=False)
        similarity = cosine_similarity(embeddings)
        np.fill_diagonal(similarity, 0)
        return similarity.max(axis=1)
    except Exception:
        return np.zeros(len(reviews))


def _anomaly_scores(features: pd.DataFrame) -> np.ndarray:
    if len(features) < 4:
        return np.zeros(len(features))

    try:
        model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
        model.fit(features)
        raw = -model.decision_function(features)
        minimum = float(raw.min())
        maximum = float(raw.max())
        if maximum == minimum:
            return np.zeros(len(features))
        return (raw - minimum) / (maximum - minimum)
    except Exception:
        return np.zeros(len(features))


def _linguistic_risk(row: pd.Series) -> float:
    very_short_review = 1.0 if row["review_length"] < 6 else 0.0
    brief_review = 1.0 if 6 <= row["review_length"] < 9 else 0.0
    very_long_review = 1.0 if row["review_length"] > 220 else 0.0
    exclamation_signal = min(row["exclamation_count"] / 5, 1)
    phrase_signal = min(row["generic_phrase_count"] / 2, 1)

    risk = (
        very_short_review * 0.18
        + brief_review * 0.06
        + very_long_review * 0.06
        + exclamation_signal * 0.14
        + row["uppercase_ratio"] * 0.16
        + row["repeated_words_ratio"] * 0.2
        + row["rating_extremity"] * 0.14
        + phrase_signal * 0.2
    )
    return float(np.clip(risk, 0, 1))


def _risk_label(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.36:
        return "Medium"
    return "Low"


def _reasons(feature_row: pd.Series, duplicate_score: float, anomaly_score: float) -> str:
    reasons: list[str] = []
    if feature_row["review_length"] < 6:
        reasons.append("very short review")
    elif feature_row["review_length"] < 9:
        reasons.append("brief review")
    if feature_row["exclamation_count"] >= 3:
        reasons.append("many exclamation marks")
    if feature_row["uppercase_ratio"] >= 0.22:
        reasons.append("high uppercase ratio")
    if feature_row["repeated_words_ratio"] >= 0.45:
        reasons.append("repeated wording")
    if feature_row["rating_extremity"] >= 1:
        reasons.append("extreme rating")
    if feature_row["generic_phrase_count"] > 0:
        reasons.append("generic promotional phrase")
    if duplicate_score >= 0.88:
        reasons.append("near-duplicate review")
    if anomaly_score >= 0.65:
        reasons.append("feature anomaly")
    return ", ".join(reasons) if reasons else "no strong fake/spam signals"


def detect_fake_reviews(
    reviews: list[str],
    ratings: list[int] | None = None,
    sensitivity: float = 0.5,
) -> pd.DataFrame:
    if not reviews:
        return pd.DataFrame(
            columns=[
                "review",
                "fake_risk_score",
                "fake_risk_label",
                "duplicate_score",
                "anomaly_score",
                "reasons",
            ]
        )

    features = _engineer_features(reviews, ratings)
    duplicate_scores = _duplicate_scores(reviews)
    anomaly_scores = _anomaly_scores(features)
    sensitivity_multiplier = 0.75 + float(np.clip(sensitivity, 0, 1))

    rows = []
    for index, review in enumerate(reviews):
        feature_row = features.iloc[index]
        linguistic_score = _linguistic_risk(feature_row)
        duplicate_score = float(duplicate_scores[index])
        anomaly_score = float(anomaly_scores[index])

        risk = (
            linguistic_score * 0.55
            + min(duplicate_score, 1) * 0.3
            + anomaly_score * 0.15
        )
        risk = float(np.clip(risk * sensitivity_multiplier, 0, 1))

        rows.append(
            {
                "review": review,
                "fake_risk_score": round(risk, 3),
                "fake_risk_label": _risk_label(risk),
                "duplicate_score": round(duplicate_score, 3),
                "anomaly_score": round(anomaly_score, 3),
                "reasons": _reasons(feature_row, duplicate_score, anomaly_score),
            }
        )

    return pd.DataFrame(rows)


def analyze_fake_review_risk(reviews: list[str], ratings: list[int] | None = None, sensitivity: float = 0.5) -> pd.DataFrame:
    """Backward-compatible wrapper for older app code."""
    result = detect_fake_reviews(reviews, ratings=ratings, sensitivity=sensitivity)
    return result.rename(
        columns={
            "fake_risk_score": "fake_risk",
            "fake_risk_label": "fake_risk_level",
            "reasons": "fake_risk_reasons",
        }
    )
