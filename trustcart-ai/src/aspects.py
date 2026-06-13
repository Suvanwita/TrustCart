from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


PRODUCT_STOPWORDS = {
    "product",
    "item",
    "thing",
    "really",
    "very",
    "just",
    "use",
    "using",
    "bought",
    "buy",
    "purchase",
    "ordered",
    "review",
}


@lru_cache(maxsize=1)
def _keybert_model():
    try:
        from keybert import KeyBERT

        return KeyBERT()
    except Exception:
        return None


def extract_aspects(reviews: list[str], top_n: int = 10) -> pd.DataFrame:
    text = " ".join(reviews)
    if not text.strip():
        return pd.DataFrame(columns=["aspect", "score"])

    kw_model = _keybert_model()
    if kw_model:
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_n,
                use_mmr=True,
                diversity=0.55,
            )
            return pd.DataFrame(keywords, columns=["aspect", "score"])
        except Exception:
            pass

    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(reviews)
    terms = vectorizer.get_feature_names_out()
    counts = matrix.sum(axis=0).A1
    counter = Counter(dict(zip(terms, counts)))
    rows = []
    for term, count in counter.most_common(top_n * 2):
        if term in PRODUCT_STOPWORDS or any(part in PRODUCT_STOPWORDS for part in term.split()):
            continue
        rows.append({"aspect": term, "score": round(float(count), 3)})
        if len(rows) == top_n:
            break
    return pd.DataFrame(rows)
