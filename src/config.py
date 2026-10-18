"""Central paths, feature schema, and decision policy for FinTrace."""
from __future__ import annotations

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATASET_PATH = os.path.join(DATA_DIR, "Procurement_Invoice_Fraud_Master.xlsx")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_fintrace.pkl")
DEMO_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_fintrace_demo.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
DEMO_METRICS_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics_demo.json")
AUDIT_LOG_PATH = os.path.join(OUTPUT_DIR, "audit_chain.jsonl")
DUPLICATE_LEDGER_PATH = os.path.join(OUTPUT_DIR, "invoice_ledger.sqlite3")

RANDOM_STATE = 42
TEST_SIZE = 0.20
FRAUD_CLASS_THRESHOLD = 0.50
APPROVE_THRESHOLD = 0.35
BLOCK_THRESHOLD = 0.75
XGBOOST_WEIGHT = 0.70
ISOLATION_FOREST_WEIGHT = 0.30

# The near-threshold heuristic is meaningful only when invoice_amounts are in
# the same normalized currency units. It is an evidence feature, not a hard block.
APPROVAL_LIMIT = 100_000.0
SPLIT_BAND_LOW = 94_000.0
SPLIT_BAND_HIGH = 98_500.0
SPLIT_HEURISTIC_CURRENCY = "INR"
VENDOR_NAME_REVIEW_THRESHOLD = 0.60

ID_COLUMNS = ("invoice_id", "supplier_id")
TARGET_COLUMNS = ("is_fraud", "fraud_type", "explanations")
RAW_NUMERIC_COLUMNS = (
    "invoice_amount",
    "submission_hour",
    "supplier_age_days",
    "supplier_risk_score",
    "blacklisted_flag",
    "avg_invoice_amount",
    "supplier_invoice_count_30d",
    "supplier_avg_amount_90d",
    "invoice_amount_zscore",
    "duplicate_invoice_flag",
    "split_invoice_flag",
    "late_night_submission_flag",
    "invoice_month",
    "invoice_day_of_week",
    "invoice_is_weekend",
    "invoice_amount_log1p",
    "invoice_vs_supplier_avg_ratio",
    "invoice_vs_avg_amount_ratio",
    "supplier_activity_per_30d",
    "abs_invoice_zscore",
    "near_approval_threshold_flag",
    "vendor_name_similarity",
    "bank_mismatch_flag",
)
CATEGORICAL_FEATURES = (
    "department_id",
    "currency",
    "payment_terms",
    "invoice_type",
    "supplier_country",
)
FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + CATEGORICAL_FEATURES
REQUIRED_DATASET_COLUMNS = ("invoice_amount", "is_fraud")
