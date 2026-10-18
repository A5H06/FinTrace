"""Leakage-aware feature creation for invoice, vendor, and policy signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS, RAW_NUMERIC_COLUMNS, CATEGORICAL_FEATURES,
    SPLIT_BAND_HIGH, SPLIT_BAND_LOW, SPLIT_HEURISTIC_CURRENCY,
)
from src.preprocessor import clean_dataframe
from src.vendor_similarity import bank_identifiers_match, pairwise_text_similarity


def _series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _number(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    return pd.to_numeric(_series(frame, column, default), errors="coerce")


def build_feature_frame(data: pd.DataFrame | dict) -> pd.DataFrame:
    """Build the same ordered model feature frame for training and inference.

    Target labels, fraud_type, explanations, invoice IDs, and raw supplier IDs are
    never included as model features. Explicit rule flags are retained, so ensure
    they are available before payment decisions and are not derived from labels.
    """
    if isinstance(data, dict):
        frame = clean_dataframe(pd.DataFrame([data]))
    else:
        frame = clean_dataframe(data)
    if frame.empty:
        raise ValueError("Cannot engineer features for an empty dataset")

    features = pd.DataFrame(index=frame.index)
    amount = _number(frame, "invoice_amount")
    age_days = _number(frame, "supplier_age_days")
    count_30d = _number(frame, "supplier_invoice_count_30d")
    vendor_avg_90d = _number(frame, "supplier_avg_amount_90d")
    avg_amount = _number(frame, "avg_invoice_amount")
    zscore = _number(frame, "invoice_amount_zscore")
    submission_hour = _number(frame, "submission_hour")

    if "invoice_date" in frame.columns:
        invoice_date = pd.to_datetime(frame["invoice_date"], errors="coerce")
        features["invoice_month"] = invoice_date.dt.month.astype(float)
        features["invoice_day_of_week"] = invoice_date.dt.dayofweek.astype(float)
        features["invoice_is_weekend"] = invoice_date.dt.dayofweek.isin([5, 6]).astype(float)
    else:
        features["invoice_month"] = _number(frame, "invoice_month")
        features["invoice_day_of_week"] = _number(frame, "invoice_day_of_week")
        features["invoice_is_weekend"] = _number(frame, "invoice_is_weekend", 0.0).fillna(0.0)

    features["invoice_amount"] = amount
    features["submission_hour"] = submission_hour
    features["supplier_age_days"] = age_days
    features["supplier_risk_score"] = _number(frame, "supplier_risk_score")
    features["blacklisted_flag"] = _number(frame, "blacklisted_flag", 0.0).fillna(0.0)
    features["avg_invoice_amount"] = avg_amount
    features["supplier_invoice_count_30d"] = count_30d
    features["supplier_avg_amount_90d"] = vendor_avg_90d
    features["invoice_amount_zscore"] = zscore

    invoice_id = _series(frame, "invoice_id", pd.NA).astype("string")
    computed_duplicates = invoice_id.notna() & invoice_id.duplicated(keep="first")
    explicit_duplicates = _number(frame, "duplicate_invoice_flag", 0.0).fillna(0.0).astype(float).gt(0)
    features["duplicate_invoice_flag"] = (computed_duplicates | explicit_duplicates).astype(float)

    currency = _series(frame, "currency", "").astype("string").str.upper().fillna("")
    near_band = amount.between(SPLIT_BAND_LOW, SPLIT_BAND_HIGH, inclusive="both")
    currency_ok = currency.eq(SPLIT_HEURISTIC_CURRENCY) | currency.eq("")
    near_band = (near_band & currency_ok).fillna(False)
    explicit_split = _number(frame, "split_invoice_flag", 0.0).fillna(0.0).gt(0)
    features["split_invoice_flag"] = (explicit_split | near_band).astype(float)
    features["late_night_submission_flag"] = (
        _number(frame, "late_night_submission_flag", 0.0).fillna(0.0).gt(0)
        | submission_hour.isin([0, 1, 2, 3, 4, 5, 22, 23])
    ).astype(float)
    features["invoice_amount_log1p"] = np.log1p(amount.clip(lower=0))
    features["invoice_vs_supplier_avg_ratio"] = amount / vendor_avg_90d.replace(0, np.nan)
    features["invoice_vs_avg_amount_ratio"] = amount / avg_amount.replace(0, np.nan)
    features["supplier_activity_per_30d"] = count_30d / (age_days.clip(lower=1) / 30.0)
    features["abs_invoice_zscore"] = zscore.abs()
    features["near_approval_threshold_flag"] = near_band.astype(float)

    explicit_similarity = _number(frame, "vendor_name_similarity")
    incoming_name_column = next((name for name in ("vendor_name", "supplier_name", "company_name") if name in frame.columns), None)
    registered_name_column = next((name for name in ("registered_vendor_name", "registered_supplier_name", "master_vendor_name") if name in frame.columns), None)
    if incoming_name_column and registered_name_column:
        derived_similarity = pairwise_text_similarity(
            frame[incoming_name_column].fillna("").astype(str).tolist(),
            frame[registered_name_column].fillna("").astype(str).tolist(),
        )
        explicit_similarity = explicit_similarity.where(explicit_similarity.notna(), pd.Series(derived_similarity, index=frame.index))
    features["vendor_name_similarity"] = explicit_similarity

    explicit_bank_flag = _number(frame, "bank_mismatch_flag", 0.0).fillna(0.0).gt(0)
    payment_bank_col = next((name for name in ("payment_bank", "payment_bank_account", "submitted_bank_account") if name in frame.columns), None)
    registered_bank_col = next((name for name in ("registered_bank", "registered_bank_account", "master_bank_account") if name in frame.columns), None)
    mismatch_values = np.zeros(len(frame), dtype=bool)
    if payment_bank_col and registered_bank_col:
        matches = [bank_identifiers_match(a, b) for a, b in zip(frame[registered_bank_col], frame[payment_bank_col])]
        mismatch_values = np.asarray([match is False for match in matches], dtype=bool)
    features["bank_mismatch_flag"] = (explicit_bank_flag.to_numpy() | mismatch_values).astype(float)

    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            values = frame[column].astype("string").fillna("__MISSING__").str.strip()
            features[column] = values.replace("", "__MISSING__")
        else:
            features[column] = "__MISSING__"

    # Preserve fixed column order even when optional source fields are absent.
    for column in RAW_NUMERIC_COLUMNS:
        if column not in features.columns:
            features[column] = np.nan
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column not in features.columns:
            features[column] = "__MISSING__"
    return features.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
