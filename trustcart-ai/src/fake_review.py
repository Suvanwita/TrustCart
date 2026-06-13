from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MARKETING_TERMS = {
    "must buy",
    "life changing",
    "100%",
    "five star",
    "best ever",
    "highly recommend",
    "no doubt",
    "trust me",
}


def _review_risk(text: str, duplicate_score: float, sensitivity: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    clean = text.strip()
    words = re.findall(r"[A-Za-z']+", clean.lower())
    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)
    exclamation_count = clean.count("!")
    uppercase_ratio = sum(1 for char in clean if char.isupper()) / max(sum(1 for char in clean if char.isalpha()), 1)

    risk = 0.0
    if word_count < 12:
        risk += 0.2
        reasons.append("very short review")
    if unique_ratio < 0.48 and word_count > 8:
        risk += 0.18
        reasons.append("repetitive wording")
    if exclamation_count >= 3:
        risk += 0.14
        reasons.append("excessive punctuation")
    if uppercase_ratio > 0.25:
        risk += 0.12
        reasons.append("unusual capitalization")
    if any(term in clean.lower() for term in MARKETING_TERMS):
        risk += 0.18
        reasons.append("promotional language")
    if duplicate_score > 0.88:
        risk += 0.22
        reasons.append("similar to another review")

    adjusted = np.clip(risk * sensitivity, 0, 1)
    return round(float(adjusted), 3), reasons or ["no strong fake-review signals"]


def analyze_fake_review_risk(reviews: list[str], sensitivity: float = 1.0) -> pd.DataFrame:
    if not reviews:
        return pd.DataFrame(columns=["review", "fake_risk", "fake_risk_level", "fake_risk_reasons"])

    duplicate_scores = np.zeros(len(reviews))
    if len(reviews) > 1:
        try:
            matrix = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(reviews)
            similarity = cosine_similarity(matrix)
            np.fill_diagonal(similarity, 0)
            duplicate_scores = similarity.max(axis=1)
        except ValueError:
            duplicate_scores = np.zeros(len(reviews))

    rows = []
    for review, duplicate_score in zip(reviews, duplicate_scores):
        risk, reasons = _review_risk(review, float(duplicate_score), sensitivity)
        if risk >= 0.55:
            level = "high"
        elif risk >= 0.3:
            level = "medium"
        else:
            level = "low"
        rows.append(
            {
                "review": review,
                "fake_risk": risk,
                "fake_risk_level": level,
                "fake_risk_reasons": ", ".join(reasons),
            }
        )
    return pd.DataFrame(rows)
