from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DATA_PATH = DATA_DIR / "sample_reviews.csv"


def validate_reviews(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("No reviews found after applying filters.")
    if "review" not in df.columns:
        raise ValueError("Expected a dataframe with a 'review' column.")


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def risk_color(level: str) -> str:
    return {"low": "green", "medium": "orange", "high": "red"}.get(level, "gray")


def save_analysis_artifact(payload: dict, filename: str = "latest_analysis.joblib") -> Path:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / filename
    joblib.dump(payload, path)
    return path
