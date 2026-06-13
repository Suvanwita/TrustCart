from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


ASPECT_GROUPS = {
    "battery": {"battery", "batteries", "charging", "charge", "charger", "power", "backup"},
    "camera": {"camera", "photo", "photos", "picture", "pictures", "video", "lens", "selfie"},
    "display": {"display", "screen", "brightness", "resolution", "touch", "panel"},
    "performance": {"performance", "lag", "lags", "processor", "ram", "smooth", "responsive"},
    "heating": {"heat", "heating", "hot", "warm", "overheat", "overheating"},
    "delivery": {"delivery", "delivered", "shipping", "shipment", "courier", "late", "arrived"},
    "packaging": {"packaging", "package", "packed", "box", "seal", "sealed", "opened"},
    "price": {"price", "cost", "cheap", "expensive", "value", "money", "worth"},
    "quality": {"quality", "premium", "cheap", "defect", "defective", "finish", "material", "materials"},
    "durability": {"durable", "durability", "sturdy", "solid", "broke", "broken", "scratch", "scratches"},
    "size": {"size", "small", "large", "compact", "big", "heavy", "light", "weight"},
    "comfort": {"comfort", "comfortable", "fit", "fits", "soft", "tight", "wear", "ergonomic"},
    "support": {"support", "service", "warranty", "refund", "replace", "replacement", "seller", "customer"},
}

ASPECT_TABLE_COLUMNS = [
    "aspect",
    "mention_count",
    "average_sentiment_score",
    "positive_review_examples",
    "negative_review_examples",
]

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "been",
    "bought",
    "from",
    "have",
    "item",
    "just",
    "ordered",
    "product",
    "purchase",
    "really",
    "review",
    "that",
    "thing",
    "this",
    "using",
    "very",
    "with",
}


@lru_cache(maxsize=1)
def _keybert_model():
    try:
        from keybert import KeyBERT

        return KeyBERT()
    except Exception:
        return None


def _extract_keywords_keybert(text: str, top_n: int) -> list[str]:
    model = _keybert_model()
    if model is None:
        return []
    try:
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.55,
        )
        return [keyword for keyword, _ in keywords]
    except Exception:
        return []


def _extract_keywords_yake(text: str, top_n: int) -> list[str]:
    try:
        import yake

        extractor = yake.KeywordExtractor(lan="en", n=2, top=top_n)
        return [keyword for keyword, _ in extractor.extract_keywords(text)]
    except Exception:
        return []


def _extract_keywords_frequency(reviews: list[str], top_n: int) -> list[str]:
    try:
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = vectorizer.fit_transform(reviews)
        terms = vectorizer.get_feature_names_out()
        counts = matrix.sum(axis=0).A1
        counter = Counter(dict(zip(terms, counts)))
        keywords = []
        for term, _ in counter.most_common(top_n * 3):
            parts = term.split()
            if term in STOPWORDS or any(part in STOPWORDS for part in parts):
                continue
            keywords.append(term)
            if len(keywords) == top_n:
                break
        return keywords
    except ValueError:
        return []


def _extract_keywords(reviews: list[str], top_n: int = 40) -> list[str]:
    text = " ".join(reviews)
    if not text.strip():
        return []
    return (
        _extract_keywords_keybert(text, top_n)
        or _extract_keywords_yake(text, top_n)
        or _extract_keywords_frequency(reviews, top_n)
    )


def _clean_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def _mentioned_aspects(text: str, keyword_aspects: set[str]) -> set[str]:
    tokens = _clean_tokens(text)
    mentioned = set()

    for aspect, terms in ASPECT_GROUPS.items():
        if aspect in keyword_aspects and tokens & terms:
            mentioned.add(aspect)
            continue
        if tokens & terms:
            mentioned.add(aspect)

    return mentioned


def _keyword_aspects(keywords: list[str]) -> set[str]:
    mapped = set()
    for keyword in keywords:
        lowered = keyword.lower()
        keyword_tokens = _clean_tokens(lowered)
        for aspect, terms in ASPECT_GROUPS.items():
            if aspect in lowered or keyword_tokens & terms:
                mapped.add(aspect)
    return mapped


def _signed_sentiment(row: pd.Series) -> float:
    label = str(row.get("sentiment_label", "neutral")).lower()
    score = float(row.get("sentiment_score", 0) or 0)
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def _examples(aspect_df: pd.DataFrame, sentiment_label: str, limit: int = 2) -> list[str]:
    if aspect_df.empty:
        return []
    filtered = aspect_df[aspect_df["sentiment_label"].str.lower() == sentiment_label]
    if filtered.empty:
        return []
    sorted_df = filtered.sort_values(["fake_risk_score", "sentiment_score"], ascending=[True, False])
    return sorted_df["review"].head(limit).tolist()


def extract_aspects(reviews_df: pd.DataFrame) -> dict:
    required = {"review", "sentiment_label", "sentiment_score", "fake_risk_score"}
    missing = required - set(reviews_df.columns)
    if missing:
        raise ValueError(f"reviews_df is missing required columns: {', '.join(sorted(missing))}")

    if reviews_df.empty:
        return {
            "aspects_table": pd.DataFrame(columns=ASPECT_TABLE_COLUMNS),
            "positive_aspects": [],
            "negative_aspects": [],
        }

    working_df = reviews_df.copy()
    working_df["review"] = working_df["review"].fillna("").astype(str)
    working_df["sentiment_label"] = working_df["sentiment_label"].fillna("neutral").astype(str)
    working_df["sentiment_score"] = pd.to_numeric(working_df["sentiment_score"], errors="coerce").fillna(0)
    working_df["fake_risk_score"] = pd.to_numeric(working_df["fake_risk_score"], errors="coerce").fillna(0)
    working_df["signed_sentiment"] = working_df.apply(_signed_sentiment, axis=1)

    keywords = _extract_keywords(working_df["review"].tolist())
    keyword_aspects = _keyword_aspects(keywords)
    aspect_mentions = {aspect: [] for aspect in ASPECT_GROUPS}

    for index, row in working_df.iterrows():
        for aspect in _mentioned_aspects(row["review"], keyword_aspects):
            aspect_mentions[aspect].append(index)

    rows = []
    for aspect, indexes in aspect_mentions.items():
        if not indexes:
            continue
        aspect_df = working_df.loc[indexes]
        positive_examples = _examples(aspect_df, "positive")
        negative_examples = _examples(aspect_df, "negative")
        rows.append(
            {
                "aspect": aspect,
                "mention_count": int(len(indexes)),
                "average_sentiment_score": round(float(aspect_df["signed_sentiment"].mean()), 3),
                "positive_review_examples": " | ".join(positive_examples),
                "negative_review_examples": " | ".join(negative_examples),
            }
        )

    aspects_table = pd.DataFrame(rows, columns=ASPECT_TABLE_COLUMNS)
    if not aspects_table.empty:
        aspects_table = aspects_table.sort_values(
            ["mention_count", "average_sentiment_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

    positive_aspects = (
        aspects_table[aspects_table["average_sentiment_score"] > 0.1]
        .sort_values(["average_sentiment_score", "mention_count"], ascending=[False, False])["aspect"]
        .head(5)
        .tolist()
        if not aspects_table.empty
        else []
    )
    negative_aspects = (
        aspects_table[aspects_table["average_sentiment_score"] < -0.1]
        .sort_values(["average_sentiment_score", "mention_count"], ascending=[True, False])["aspect"]
        .head(5)
        .tolist()
        if not aspects_table.empty
        else []
    )

    return {
        "aspects_table": aspects_table,
        "positive_aspects": positive_aspects,
        "negative_aspects": negative_aspects,
    }
