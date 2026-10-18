"""Synthetic data generator used only by `python main.py --demo` smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_dataset(rows: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """Create synthetic AP-like data with intentionally planted fraud patterns.

    These data and their metrics are not suitable for reporting real-world model
    accuracy; use the organization's labelled procurement workbook for evaluation.
    """
    if rows < 500:
        raise ValueError("Demo dataset needs at least 500 rows")
    rng = np.random.default_rng(random_state)
    n = int(rows)
    fraud = (rng.random(n) < 0.22).astype(int)
    invoice_amount = rng.lognormal(mean=10.7, sigma=0.75, size=n)
    supplier_age = rng.integers(20, 3200, size=n).astype(float)
    submission_hour = rng.integers(7, 19, size=n)
    zscore = rng.normal(0, 1.0, size=n)
    duplicate = np.zeros(n, dtype=int)
    split = np.zeros(n, dtype=int)
    late = np.zeros(n, dtype=int)
    fraud_types = np.full(n, "NONE", dtype=object)
    fraud_indices = np.flatnonzero(fraud)
    patterns = rng.choice(["DUPLICATE", "SPLIT", "GHOST_SUPPLIER", "INFLATED", "DOC_TAMPER"], size=len(fraud_indices))
    for idx, kind in zip(fraud_indices, patterns):
        fraud_types[idx] = kind
        if kind == "DUPLICATE":
            duplicate[idx] = 1
        elif kind == "SPLIT":
            invoice_amount[idx] = rng.uniform(94_000, 98_500)
            split[idx] = 1
        elif kind == "GHOST_SUPPLIER":
            supplier_age[idx] = rng.integers(0, 30)
            invoice_amount[idx] = rng.uniform(100_000, 450_000)
        elif kind == "INFLATED":
            zscore[idx] = rng.uniform(3.5, 8.0)
            invoice_amount[idx] *= rng.uniform(2.0, 4.0)
        elif kind == "DOC_TAMPER":
            late[idx] = int(rng.random() < 0.65)
            submission_hour[idx] = int(rng.choice([0, 1, 2, 3, 23])) if late[idx] else submission_hour[idx]
            zscore[idx] = rng.uniform(2.0, 5.0)
    legitimate_indices = np.flatnonzero(fraud == 0)
    submission_hour[legitimate_indices] = rng.integers(7, 19, size=len(legitimate_indices))
    supplier_risk = np.clip(rng.beta(2, 8, size=n) + 0.35 * fraud, 0, 1)
    currency = rng.choice(["INR", "USD", "EUR"], size=n, p=[0.82, 0.12, 0.06])
    amount_baseline = np.maximum(1.0, invoice_amount / np.maximum(0.2, rng.normal(1.0 + 0.3 * fraud, 0.35, size=n)))
    return pd.DataFrame({
        "invoice_id": [f"INV-{i:07d}" for i in range(n)],
        "department_id": rng.choice(["FIN", "OPS", "IT", "HR", "PROC", "LEGAL"], size=n),
        "invoice_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 650, size=n), unit="D"),
        "invoice_amount": invoice_amount.round(2),
        "currency": currency,
        "payment_terms": rng.choice(["NET30", "NET45", "NET60", "DUE_ON_RECEIPT"], size=n),
        "invoice_type": rng.choice(["GOODS", "SERVICES", "CAPEX", "OPEX"], size=n),
        "submission_hour": submission_hour,
        "supplier_id": [f"SUP-{i:05d}" for i in rng.integers(1, max(2, n // 3), size=n)],
        "supplier_country": rng.choice(["IN", "US", "GB", "DE", "SG"], size=n, p=[0.72, 0.1, 0.07, 0.06, 0.05]),
        "supplier_age_days": supplier_age,
        "supplier_risk_score": supplier_risk.round(4),
        "blacklisted_flag": np.zeros(n, dtype=int),
        "avg_invoice_amount": amount_baseline.round(2),
        "supplier_invoice_count_30d": rng.poisson(3 + 1.5 * fraud, size=n),
        "supplier_avg_amount_90d": amount_baseline.round(2),
        "invoice_amount_zscore": zscore.round(3),
        "duplicate_invoice_flag": duplicate,
        "split_invoice_flag": split,
        "late_night_submission_flag": late,
        "is_fraud": fraud,
        "fraud_type": fraud_types,
        "explanations": ["synthetic_demo_label" if label else "synthetic_demo_legitimate" for label in fraud],
    })
