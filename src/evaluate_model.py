"""Evaluation reports for classification quality and firewall routing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from src.config import BLOCK_THRESHOLD, FRAUD_CLASS_THRESHOLD, XGBOOST_WEIGHT, ISOLATION_FOREST_WEIGHT
from src.preprocessor import truthy_flag
from src.train_model import predict_components


def evaluate_model(bundle: dict, x_test_features: pd.DataFrame, y_test, raw_test: pd.DataFrame | None = None) -> dict:
    """Calculate test-set metrics, including firewall-level rule overlays."""
    y_true = np.asarray(y_test, dtype=int)
    p_xgb, anomaly, base_risk = predict_components(bundle, x_test_features)
    operational_risk = base_risk.copy()
    hard_rule_count = 0
    if raw_test is not None:
        for position, (_, row) in enumerate(raw_test.iterrows()):
            duplicate = truthy_flag(row.get("duplicate_invoice_flag", 0))
            blacklisted = truthy_flag(row.get("blacklisted_flag", 0)) or truthy_flag(row.get("sanctions_match_flag", 0))
            bank_mismatch = truthy_flag(row.get("bank_mismatch_flag", 0)) or truthy_flag(row.get("bank_change_flag", 0))
            if duplicate or blacklisted or bank_mismatch:
                operational_risk[position] = max(operational_risk[position], BLOCK_THRESHOLD)
                hard_rule_count += 1
    predicted = (operational_risk >= FRAUD_CLASS_THRESHOLD).astype(int)
    firewall_flagged = (operational_risk >= 0.35).astype(int)
    try:
        auc = float(roc_auc_score(y_true, operational_risk))
    except ValueError:
        auc = None
    try:
        average_precision = float(average_precision_score(y_true, operational_risk))
    except ValueError:
        average_precision = None
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    flagged_matrix = confusion_matrix(y_true, firewall_flagged, labels=[0, 1])
    report = {
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("model_version"),
        "classifier_backend": bundle.get("classifier_backend"),
        "test_rows": int(len(y_true)),
        "fraud_prevalence": float(y_true.mean()),
        "scale_pos_weight": float(bundle.get("scale_pos_weight", 1.0)),
        "weights": {"xgboost": XGBOOST_WEIGHT, "isolation_forest": ISOLATION_FOREST_WEIGHT},
        "metrics_at_0_50": {
            "threshold": FRAUD_CLASS_THRESHOLD,
            "accuracy": float(accuracy_score(y_true, predicted)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
            "roc_auc": auc,
            "average_precision": average_precision,
            "confusion_matrix_labels_0_1": matrix.tolist(),
            "classification_report": classification_report(
                y_true, predicted, labels=[0, 1], target_names=["legitimate", "fraud"],
                output_dict=True, zero_division=0,
            ),
        },
        "firewall_operations": {
            "approve_count": int((operational_risk < 0.35).sum()),
            "hold_count": int(((operational_risk >= 0.35) & (operational_risk < BLOCK_THRESHOLD)).sum()),
            "block_count": int((operational_risk >= BLOCK_THRESHOLD).sum()),
            "flagged_as_hold_or_block_count": int(firewall_flagged.sum()),
            "confusion_matrix_labels_0_1_at_0_35": flagged_matrix.tolist(),
            "hard_rule_trigger_count": int(hard_rule_count),
        },
        "component_summary": {
            "xgboost_probability_mean": float(np.mean(p_xgb)),
            "isolation_anomaly_score_mean": float(np.mean(anomaly)),
            "composite_risk_mean": float(np.mean(operational_risk)),
            "composite_risk_p95": float(np.quantile(operational_risk, 0.95)),
        },
    }
    return report
