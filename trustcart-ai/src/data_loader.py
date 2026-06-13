from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ["review", "rating"]


def load_sample_data(sample_path: str | Path) -> pd.DataFrame:
    """Load bundled sample reviews."""
    return normalize_reviews(pd.read_csv(sample_path))


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Load reviews from a Streamlit uploaded CSV file."""
    return normalize_reviews(pd.read_csv(uploaded_file))


def load_pasted_reviews(text: str) -> pd.DataFrame:
    """Create a reviews dataframe from pasted newline-separated text."""
    reviews = [line.strip() for line in text.splitlines() if line.strip()]
    return normalize_reviews(pd.DataFrame({"review": reviews, "rating": [None] * len(reviews)}))


def normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean dataframe with review text and optional numeric ratings."""
    if "review" not in df.columns:
        raise ValueError("CSV must include a 'review' column.")

    normalized = df.copy()
    normalized["review"] = normalized["review"].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["review"] != ""]

    if "rating" not in normalized.columns:
        normalized["rating"] = None
    normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")

    return normalized[REQUIRED_COLUMNS].reset_index(drop=True)


def filter_by_length(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Filter reviews by character length."""
    if df.empty:
        return df
    mask = df["review"].str.len() >= int(min_length)
    return df[mask].reset_index(drop=True)


def dataframe_from_records(records: Iterable[dict]) -> pd.DataFrame:
    """Small helper used by tests and notebooks."""
    return normalize_reviews(pd.DataFrame(records))
