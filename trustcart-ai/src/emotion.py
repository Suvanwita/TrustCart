from __future__ import annotations

import pandas as pd


EMOTION_KEYWORDS = {
    "joy": {"love", "happy", "great", "excellent", "perfect", "amazing"},
    "trust": {"reliable", "solid", "durable", "authentic", "original", "premium"},
    "anger": {"hate", "refund", "broken", "terrible", "worst", "waste"},
    "sadness": {"disappointed", "poor", "bad", "weak", "issue"},
    "surprise": {"unexpected", "surprised", "shock", "wow"},
}


def load_emotion_pipeline():
    """Load a lightweight Hugging Face emotion pipeline when available."""
    try:
        from transformers import pipeline

        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    except Exception:
        return None


def _keyword_emotion(text: str) -> tuple[str, float]:
    tokens = {token.strip(".,!?;:()[]{}\"'").lower() for token in text.split()}
    scores = {emotion: len(tokens & words) for emotion, words in EMOTION_KEYWORDS.items()}
    emotion, hits = max(scores.items(), key=lambda item: item[1])
    if hits == 0:
        return "neutral", 0.5
    return emotion, min(0.95, 0.55 + hits * 0.15)


def analyze_emotion_with_pipeline(reviews: list[str], model=None) -> pd.DataFrame:
    """Classify review emotions using a provided pipeline or safe fallback."""
    rows = []

    if model:
        try:
            predictions = model(reviews, truncation=True)
            for review, pred_group in zip(reviews, predictions):
                pred = pred_group[0] if isinstance(pred_group, list) else pred_group
                rows.append(
                    {
                        "review": review,
                        "emotion_label": pred["label"].lower(),
                        "emotion_score": float(pred["score"]),
                    }
                )
            return pd.DataFrame(rows)
        except Exception:
            pass

    for review in reviews:
        emotion, score = _keyword_emotion(review)
        rows.append({"review": review, "emotion_label": emotion, "emotion_score": score})
    return pd.DataFrame(rows)


def analyze_emotion(reviews: list[str]) -> pd.DataFrame:
    """Return emotion labels and scores for each review."""
    return analyze_emotion_with_pipeline(reviews, load_emotion_pipeline())


def analyze_emotions(reviews: list[str]) -> pd.DataFrame:
    """Backward-compatible alias for older app code."""
    return analyze_emotion(reviews)
