from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "comfortable",
    "durable",
    "easy",
    "excellent",
    "fast",
    "good",
    "great",
    "happy",
    "love",
    "perfect",
    "premium",
    "reliable",
    "smooth",
    "solid",
    "useful",
    "worth",
}

NEGATIVE_WORDS = {
    "bad",
    "broken",
    "cheap",
    "defect",
    "disappointed",
    "fake",
    "hard",
    "hate",
    "issue",
    "late",
    "noisy",
    "poor",
    "refund",
    "returned",
    "slow",
    "terrible",
    "waste",
    "weak",
    "worst",
}


@lru_cache(maxsize=1)
def _sentiment_pipeline():
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        return None


def _lexicon_sentiment(text: str) -> tuple[str, float]:
    tokens = {token.strip(".,!?;:()[]{}\"'").lower() for token in text.split()}
    positive_hits = len(tokens & POSITIVE_WORDS)
    negative_hits = len(tokens & NEGATIVE_WORDS)
    score = positive_hits - negative_hits

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    confidence = min(0.95, 0.55 + abs(score) * 0.12)
    return label, confidence


def analyze_sentiment(reviews: list[str]) -> pd.DataFrame:
    """Classify each review as positive, neutral, or negative."""
    model = _sentiment_pipeline()
    rows = []

    if model:
        try:
            predictions = model(reviews, truncation=True)
            for review, pred in zip(reviews, predictions):
                raw_label = pred["label"].lower()
                label = "positive" if "pos" in raw_label else "negative"
                score = float(pred["score"])
                if score < 0.62:
                    label = "neutral"
                rows.append({"review": review, "sentiment": label, "sentiment_confidence": score})
            return pd.DataFrame(rows)
        except Exception:
            pass

    for review in reviews:
        label, score = _lexicon_sentiment(review)
        rows.append({"review": review, "sentiment": label, "sentiment_confidence": score})
    return pd.DataFrame(rows)


def sentiment_distribution(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    counts = sentiment_df["sentiment"].value_counts(normalize=True).rename_axis("sentiment").reset_index(name="share")
    counts["percent"] = np.round(counts["share"] * 100, 1)
    return counts
