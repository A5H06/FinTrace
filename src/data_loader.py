"""Memory-conscious loader and schema validation for the procurement workbook."""
from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd

from src.config import REQUIRED_DATASET_COLUMNS
from src.preprocessor import clean_dataframe

LOGGER = logging.getLogger(__name__)

STRING_ID_COLUMNS = ("invoice_id", "supplier_id", "department_id")


def validate_dataset(frame: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_DATASET_COLUMNS) -> pd.DataFrame:
    """Validate the expected columns and target distribution without altering labels."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("The dataset contains no rows")
    if frame["is_fraud"].nunique(dropna=False) < 2:
        raise ValueError("Training requires both legitimate (0) and fraud (1) labels")
    if frame["invoice_amount"].notna().sum() == 0:
        raise ValueError("invoice_amount contains no numeric values")
    missing_optional = [
        column for column in (
            "invoice_id", "department_id", "invoice_date", "currency", "payment_terms",
            "invoice_type", "submission_hour", "supplier_id", "supplier_country",
            "supplier_age_days", "supplier_risk_score", "blacklisted_flag",
            "avg_invoice_amount", "supplier_invoice_count_30d", "supplier_avg_amount_90d",
            "invoice_amount_zscore", "duplicate_invoice_flag", "split_invoice_flag",
            "late_night_submission_flag", "fraud_type", "explanations",
        ) if column not in frame.columns
    ]
    if missing_optional:
        LOGGER.warning("Optional dataset columns not supplied; related features will use defaults: %s", missing_optional)
    fraud_rate = float(frame["is_fraud"].mean())
    LOGGER.info(
        "Validated %s rows, %s columns; fraud rate %.2f%%; estimated memory %.1f MiB",
        len(frame), len(frame.columns), fraud_rate * 100,
        frame.memory_usage(deep=True).sum() / (1024 * 1024),
    )
    return frame


def load_dataset(path: str) -> pd.DataFrame:
    """Read an Excel workbook while keeping high-cardinality identifiers compact.

    The Excel file is first loaded with identifier columns as strings so leading
    zeros are preserved. High-cardinality IDs are then converted to categorical
    dtype to reduce downstream memory pressure.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place Procurement_Invoice_Fraud_Master.xlsx "
            "inside the project's data/ directory, or run `python main.py --demo` "
            "to smoke-test the pipeline with synthetic data."
        )
    LOGGER.info("Loading workbook: %s", path)
    id_dtypes = {name: "string" for name in STRING_ID_COLUMNS}
    raw = pd.read_excel(path, engine="openpyxl", dtype=id_dtypes)
    frame = clean_dataframe(raw)
    for column in ("supplier_id", "department_id"):
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    for column in ("supplier_country", "currency", "payment_terms", "invoice_type"):
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    return validate_dataset(frame)
