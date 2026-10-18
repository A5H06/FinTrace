"""Input cleaning shared by training and real-time inference."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def normalize_column_name(name: Any) -> str:
    """Convert a workbook column heading into a stable snake_case identifier."""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers, whitespace, dates, labels, and obvious numeric types.

    This function deliberately does not remove duplicate rows: duplicate rows can
    themselves be evidence in an invoice-fraud investigation.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("clean_dataframe expects a pandas DataFrame")
    frame = data.copy(deep=True)
    frame.columns = [normalize_column_name(col) for col in frame.columns]
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Column names collide after normalization: {duplicated}")

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].astype("string").str.strip()
        frame[column] = values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    if "invoice_date" in frame.columns:
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce", utc=False)

    numeric_candidates = {
        "invoice_amount", "submission_hour", "supplier_age_days", "supplier_risk_score",
        "blacklisted_flag", "avg_invoice_amount", "supplier_invoice_count_30d",
        "supplier_avg_amount_90d", "invoice_amount_zscore", "duplicate_invoice_flag",
        "split_invoice_flag", "late_night_submission_flag", "is_fraud", "invoice_month",
        "invoice_day_of_week", "invoice_is_weekend", "vendor_name_similarity",
        "bank_mismatch_flag", "sanctions_match_flag", "bank_change_flag",
    }
    for column in numeric_candidates.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ("invoice_id", "supplier_id", "department_id"):
        if column in frame.columns:
            # String IDs preserve leading zeros; categorical encoding is applied by the loader.
            frame[column] = frame[column].astype("string").str.strip()

    if "is_fraud" in frame.columns:
        if frame["is_fraud"].dtype == object or str(frame["is_fraud"].dtype).startswith("string"):
            label_map = {
                "1": 1, "true": 1, "yes": 1, "fraud": 1, "y": 1,
                "0": 0, "false": 0, "no": 0, "legitimate": 0, "none": 0, "n": 0,
            }
            labels = frame["is_fraud"].astype("string").str.strip().str.lower()
            mapped = labels.map(label_map)
            fallback = pd.to_numeric(labels, errors="coerce")
            frame["is_fraud"] = mapped.where(mapped.notna(), fallback)
        frame["is_fraud"] = pd.to_numeric(frame["is_fraud"], errors="coerce")
        invalid = ~frame["is_fraud"].isin([0, 1])
        if invalid.any():
            bad_count = int(invalid.sum())
            raise ValueError(f"is_fraud must contain only 0/1 labels; found {bad_count} invalid or missing rows")
        frame["is_fraud"] = frame["is_fraud"].astype("int8")
    return frame


def to_number(value: Any, default: float = 0.0) -> float:
    """Convert one value to a finite float, returning default on invalid input."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def truthy_flag(value: Any) -> bool:
    """Safely interpret spreadsheet/API boolean and numeric flags."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "blocked"}
    try:
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0)
    except (TypeError, ValueError):
        return bool(value)
