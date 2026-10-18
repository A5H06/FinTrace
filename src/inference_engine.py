"""Real-time hybrid risk scoring, policy firewall, and evidence generation."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    APPROVE_THRESHOLD, BLOCK_THRESHOLD, VENDOR_NAME_REVIEW_THRESHOLD,
    XGBOOST_WEIGHT, ISOLATION_FOREST_WEIGHT,
)
from src.feature_engineering import build_feature_frame
from src.preprocessor import truthy_flag, to_number
from src.train_model import predict_components
from src.vendor_similarity import bank_identifiers_match


def _first_value(payload: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if pd.isna(value):
            continue
        return value
    return None


def evaluate_hard_rules(payload: dict, duplicate_detected: bool = False) -> list[str]:
    """Evaluate mandatory payment-policy checks before the statistical models."""
    triggered = []
    if duplicate_detected or truthy_flag(payload.get("duplicate_invoice_flag", 0)):
        triggered.append("DUPLICATE_INVOICE")
    if truthy_flag(payload.get("blacklisted_flag", 0)) or truthy_flag(payload.get("sanctions_match_flag", 0)):
        triggered.append("BLACKLISTED_OR_SANCTIONED_VENDOR")
    if truthy_flag(payload.get("bank_mismatch_flag", 0)) or truthy_flag(payload.get("bank_change_flag", 0)):
        triggered.append("BANK_DETAIL_MISMATCH_OR_CHANGE")
    registered_bank = _first_value(payload, ("registered_bank", "registered_bank_account", "master_bank_account"))
    submitted_bank = _first_value(payload, ("payment_bank", "payment_bank_account", "submitted_bank_account"))
    match = bank_identifiers_match(registered_bank, submitted_bank)
    if match is False and "BANK_DETAIL_MISMATCH_OR_CHANGE" not in triggered:
        triggered.append("BANK_DETAIL_MISMATCH_OR_CHANGE")
    return triggered


def _decision(risk: float) -> str:
    if risk < APPROVE_THRESHOLD:
        return "APPROVE"
    if risk < BLOCK_THRESHOLD:
        return "HOLD"
    return "BLOCK"


def _friendly_feature(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ")
    return name.strip().capitalize()


def _local_xgb_contributions(bundle: dict, feature_frame: pd.DataFrame, limit: int = 5) -> tuple[list[str], list[str]]:
    """Use XGBoost's native per-row contribution values when the backend supports it."""
    if bundle.get("classifier_backend") != "xgboost":
        return [], []
    try:
        import xgboost as xgb
        encoded = bundle["classifier_preprocessor"].transform(feature_frame.loc[:, bundle["feature_columns"]])
        matrix = xgb.DMatrix(encoded)
        contributions = bundle["classifier"].get_booster().predict(matrix, pred_contribs=True)[0, :-1]
        names = list(bundle["classifier_preprocessor"].get_feature_names_out())
        pairs = list(zip(names, contributions))
        positive = sorted((item for item in pairs if item[1] > 0), key=lambda item: item[1], reverse=True)[:limit]
        negative = sorted((item for item in pairs if item[1] < 0), key=lambda item: item[1])[:limit]
        for_names = [f"{_friendly_feature(name)} increased model risk (contribution {value:.3f})" for name, value in positive]
        against_names = [f"{_friendly_feature(name)} reduced model risk (contribution {value:.3f})" for name, value in negative]
        return for_names, against_names
    except (ValueError, TypeError, AttributeError, ImportError, IndexError):
        return [], []


def score_transaction(transaction: dict, bundle: dict, duplicate_detected: bool = False) -> dict:
    """Score one invoice and return the risk decision plus evidence and latency.

    A hard-rule hit is checked first and immediately returns BLOCK without invoking
    either model. Non-blocked transactions use the specified 70/30 weighted score.
    A verified vendor-name near-match is only a review signal, never proof of fraud.
    """
    started = time.perf_counter()
    payload = dict(transaction or {})
    hard_rules = evaluate_hard_rules(payload, duplicate_detected=duplicate_detected)
    if hard_rules:
        result = {
            "risk_score": 1.0,
            "p_xgboost": None,
            "isolation_anomaly_score": None,
            "base_composite_score": None,
            "decision": "BLOCK",
            "hard_rule_triggered": True,
            "hard_rules": hard_rules,
            "evidence_for": [f"Mandatory policy rule triggered: {rule}" for rule in hard_rules],
            "evidence_against": ["No statistical-model score was needed because a mandatory policy rule blocked this payment."],
            "vendor_name_similarity": None,
            "model_backend": bundle.get("classifier_backend"),
        }
        result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    feature_frame = build_feature_frame(payload)
    p_xgb_array, anomaly_array, base_risk_array = predict_components(bundle, feature_frame)
    p_xgb = float(p_xgb_array[0])
    anomaly = float(anomaly_array[0])
    base_risk = float(base_risk_array[0])
    risk = base_risk
    evidence_for = []
    evidence_against = []

    similarity = to_number(payload.get("vendor_name_similarity"), default=float("nan"))
    if not np.isfinite(similarity):
        raw_similarity = feature_frame.iloc[0]["vendor_name_similarity"]
        similarity = float(raw_similarity) if pd.notna(raw_similarity) else float("nan")
    if np.isfinite(similarity) and similarity < VENDOR_NAME_REVIEW_THRESHOLD:
        risk = max(risk, APPROVE_THRESHOLD)
        evidence_for.append(f"Vendor-name similarity is low ({similarity:.2f}); manual identity verification is recommended.")
    elif np.isfinite(similarity) and similarity >= 0.85:
        evidence_against.append(f"Vendor name closely matches the registered profile (similarity {similarity:.2f}).")

    if p_xgb >= 0.75:
        evidence_for.append(f"Supervised model assigned a high fraud probability ({p_xgb:.2f}).")
    elif p_xgb < 0.20:
        evidence_against.append(f"Supervised model assigned a low fraud probability ({p_xgb:.2f}).")
    if anomaly >= 0.80:
        evidence_for.append(f"Transaction is a strong behavioral outlier (anomaly score {anomaly:.2f}).")
    elif anomaly < 0.25:
        evidence_against.append(f"Transaction is relatively consistent with learned behavior (anomaly score {anomaly:.2f}).")

    row = feature_frame.iloc[0]
    if to_number(row.get("duplicate_invoice_flag"), 0.0) > 0:
        evidence_for.append("Duplicate invoice signal is present.")
    if to_number(row.get("split_invoice_flag"), 0.0) > 0:
        evidence_for.append("Invoice splitting or near-approval-band signal is present; inspect related invoices and purchase orders.")
    if to_number(row.get("late_night_submission_flag"), 0.0) > 0:
        evidence_for.append("Invoice was submitted during the configured late-night window.")
    if abs(to_number(row.get("invoice_amount_zscore"), 0.0)) >= 3.0:
        evidence_for.append("Invoice amount deviates by at least three supplied standard deviations from its baseline.")
    age_days = to_number(row.get("supplier_age_days"), 0.0)
    amount = to_number(row.get("invoice_amount"), 0.0)
    if age_days >= 0 and age_days <= 30 and amount >= 100_000:
        evidence_for.append("New supplier submitted a high-value invoice; verify onboarding and purchase-order evidence.")
    if to_number(row.get("blacklisted_flag"), 0.0) == 0:
        evidence_against.append("No blacklist flag was present in the supplied vendor record.")
    if to_number(row.get("abs_invoice_zscore"), float("inf")) <= 1.0:
        evidence_against.append("Invoice amount is within one supplied standard deviation of the vendor baseline.")
    if not evidence_for:
        evidence_for.append("No individual deterministic risk indicator crossed its explanatory threshold; review the model scores and supplied context.")
    shap_for, shap_against = _local_xgb_contributions(bundle, feature_frame)
    evidence_for.extend(shap_for[:3])
    evidence_against.extend(shap_against[:3])

    decision = _decision(risk)
    result = {
        "risk_score": round(float(np.clip(risk, 0.0, 1.0)), 6),
        "p_xgboost": round(p_xgb, 6),
        "isolation_anomaly_score": round(anomaly, 6),
        "base_composite_score": round(base_risk, 6),
        "risk_formula": f"({XGBOOST_WEIGHT:.2f} * P_XGBoost) + ({ISOLATION_FOREST_WEIGHT:.2f} * S_IsolationForest)",
        "decision": decision,
        "hard_rule_triggered": False,
        "hard_rules": [],
        "evidence_for": evidence_for[:8],
        "evidence_against": evidence_against[:8],
        "vendor_name_similarity": round(similarity, 6) if np.isfinite(similarity) else None,
        "model_backend": bundle.get("classifier_backend"),
    }
    result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return result
