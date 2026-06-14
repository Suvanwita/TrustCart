from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ["review", "rating"]
VALID_RATING_MIN = 1
VALID_RATING_MAX = 5


class ReviewDataError(ValueError):
    """Friendly validation error for user-provided review data."""


def _format_columns(columns: Iterable[str]) -> str:
    column_names = [str(column) for column in columns]
    return ", ".join(column_names) if column_names else "no columns found"


def load_sample_data(sample_path: str | Path) -> pd.DataFrame:
    """Load bundled sample reviews."""
    return validate_and_normalize_reviews(pd.read_csv(sample_path))["data"]


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Load reviews from a Streamlit uploaded CSV file."""
    return validate_and_normalize_reviews(pd.read_csv(uploaded_file))["data"]


def load_uploaded_csv_with_validation(uploaded_file) -> dict:
    """Load uploaded CSV and return data plus friendly validation warnings."""
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ReviewDataError("The uploaded CSV is empty. Add at least one row with a 'review' column.") from exc
    except UnicodeDecodeError as exc:
        raise ReviewDataError("The uploaded file could not be read as a CSV. Please upload a UTF-8 CSV file.") from exc
    except pd.errors.ParserError as exc:
        raise ReviewDataError("The uploaded CSV could not be parsed. Check that commas and quotes are formatted correctly.") from exc
    return validate_and_normalize_reviews(df)


def load_pasted_reviews(text: str) -> pd.DataFrame:
    """Create a reviews dataframe from pasted newline-separated text."""
    reviews = [line.strip() for line in text.splitlines() if line.strip()]
    return validate_and_normalize_reviews(pd.DataFrame({"review": reviews, "rating": [None] * len(reviews)}))["data"]


def validate_and_normalize_reviews(df: pd.DataFrame) -> dict:
    """Return clean review data plus non-blocking validation warnings."""
    warnings: list[str] = []

    if df.empty and not len(df.columns):
        raise ReviewDataError("The file is empty. Add a 'review' column and at least one review row.")

    if "review" not in df.columns:
        raise ReviewDataError(
            "Missing required 'review' column. Rename the review text column to 'review'. "
            f"Detected columns: {_format_columns(df.columns)}."
        )

    normalized = df.copy()
    normalized["review"] = normalized["review"].fillna("").astype(str).str.strip()
    blank_count = int((normalized["review"] == "").sum())
    if blank_count:
        warnings.append(f"Removed {blank_count} blank review row(s).")
    normalized = normalized[normalized["review"] != ""].copy()

    if normalized.empty:
        raise ReviewDataError("No usable reviews found. The 'review' column is present, but all review cells are blank.")

    duplicate_count = int(normalized.duplicated(subset=["review"], keep="first").sum())
    if duplicate_count:
        warnings.append(f"Removed {duplicate_count} duplicate review row(s).")
        normalized = normalized.drop_duplicates(subset=["review"], keep="first").copy()

    if "rating" not in normalized.columns:
        normalized["rating"] = None
    else:
        raw_ratings = normalized["rating"]
        parsed_ratings = pd.to_numeric(raw_ratings, errors="coerce")
        present_mask = raw_ratings.notna() & (raw_ratings.astype(str).str.strip() != "")
        invalid_mask = present_mask & (parsed_ratings.isna() | ~parsed_ratings.between(VALID_RATING_MIN, VALID_RATING_MAX))
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            warnings.append(
                f"Ignored {invalid_count} invalid rating value(s). Ratings should be numbers from "
                f"{VALID_RATING_MIN} to {VALID_RATING_MAX}."
            )
            parsed_ratings.loc[invalid_mask] = pd.NA
        normalized["rating"] = parsed_ratings

    return {"data": normalized[REQUIRED_COLUMNS].reset_index(drop=True), "warnings": warnings}


def normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean dataframe with review text and optional numeric ratings."""
    return validate_and_normalize_reviews(df)["data"]


def filter_by_length(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Filter reviews by character length."""
    if df.empty:
        return df
    mask = df["review"].str.len() >= int(min_length)
    return df[mask].reset_index(drop=True)


def dataframe_from_records(records: Iterable[dict]) -> pd.DataFrame:
    """Small helper used by tests and notebooks."""
    return normalize_reviews(pd.DataFrame(records))
