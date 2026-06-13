from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _summarization_pipeline():
    try:
        from transformers import pipeline

        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None


def _top_terms(reviews: list[str], limit: int = 5) -> list[str]:
    words = []
    stop = {
        "about",
        "after",
        "again",
        "also",
        "because",
        "been",
        "from",
        "have",
        "just",
        "product",
        "really",
        "that",
        "this",
        "very",
        "with",
    }
    for review in reviews:
        words.extend(
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in review.split()
            if len(token.strip(".,!?;:()[]{}\"'")) > 3
        )
    terms = [word for word in words if word and word not in stop]
    return [term for term, _ in Counter(terms).most_common(limit)]


def summarize_pros_cons(analysis_df: pd.DataFrame) -> dict:
    if analysis_df.empty:
        return {"pros": "No reviews available.", "cons": "No reviews available.", "summary": "No review text to analyze."}

    positive_reviews = analysis_df.loc[analysis_df["sentiment"] == "positive", "review"].tolist()
    negative_reviews = analysis_df.loc[analysis_df["sentiment"] == "negative", "review"].tolist()
    all_reviews = analysis_df["review"].tolist()

    model = _summarization_pipeline()
    joined = " ".join(all_reviews)
    model_summary = ""
    if model and len(joined.split()) > 45:
        try:
            model_summary = model(joined[:3500], max_length=80, min_length=25, do_sample=False)[0]["summary_text"]
        except Exception:
            model_summary = ""

    pro_terms = _top_terms(positive_reviews)
    con_terms = _top_terms(negative_reviews)
    pros = "Positive reviews often mention " + ", ".join(pro_terms) + "." if pro_terms else "Few clear pros were detected."
    cons = "Negative reviews often mention " + ", ".join(con_terms) + "." if con_terms else "Few clear cons were detected."

    summary = model_summary or f"{len(all_reviews)} reviews analyzed, with strongest positive terms around {', '.join(pro_terms[:3]) or 'none'} and concern terms around {', '.join(con_terms[:3]) or 'none'}."
    return {"pros": pros, "cons": cons, "summary": summary}
