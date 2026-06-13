from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trust_score(analysis_df: pd.DataFrame) -> dict:
    if analysis_df.empty:
        return {
            "score": 0,
            "label": "No data",
            "positive_share": 0,
            "negative_share": 0,
            "average_fake_risk": 0,
            "average_rating": None,
        }

    sentiment_col = "sentiment_label" if "sentiment_label" in analysis_df.columns else "sentiment"
    positive_share = (analysis_df[sentiment_col] == "positive").mean()
    negative_share = (analysis_df[sentiment_col] == "negative").mean()
    average_fake_risk = analysis_df["fake_risk"].mean()
    average_rating = analysis_df["rating"].dropna().mean() if "rating" in analysis_df else np.nan
    rating_component = 0.5
    if not np.isnan(average_rating):
        rating_component = np.clip((average_rating - 1) / 4, 0, 1)

    score = (
        positive_share * 42
        + (1 - negative_share) * 18
        + (1 - average_fake_risk) * 28
        + rating_component * 12
    )
    score = int(round(np.clip(score, 0, 100)))

    if score >= 75:
        label = "High trust"
    elif score >= 55:
        label = "Moderate trust"
    elif score >= 35:
        label = "Low trust"
    else:
        label = "Very low trust"

    return {
        "score": score,
        "label": label,
        "positive_share": round(float(positive_share), 3),
        "negative_share": round(float(negative_share), 3),
        "average_fake_risk": round(float(average_fake_risk), 3),
        "average_rating": None if np.isnan(average_rating) else round(float(average_rating), 2),
    }


def buy_avoid_suggestion(score: int, fake_risk: float, negative_share: float) -> str:
    if score >= 75 and fake_risk < 0.3:
        return "Buy: review signals are broadly trustworthy and positive."
    if score >= 55 and negative_share < 0.35:
        return "Consider buying: check the common cons before deciding."
    if fake_risk >= 0.5:
        return "Avoid or verify elsewhere: fake-review risk is elevated."
    return "Avoid: trust and sentiment signals are weak."
