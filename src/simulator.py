"""In-memory scenario simulation; never writes changed fields to a transaction store."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.inference_engine import score_transaction


def simulate_transaction(transaction: dict, changes: dict, bundle: dict) -> dict:
    """Compare original and proposed values using transient dictionaries only."""
    original = deepcopy(dict(transaction or {}))
    updated = deepcopy(original)
    applied_changes = deepcopy(dict(changes or {}))
    updated.update(applied_changes)
    baseline_result = score_transaction(original, bundle, duplicate_detected=False)
    simulated_result = score_transaction(updated, bundle, duplicate_detected=False)
    return {
        "simulation_only": True,
        "production_record_modified": False,
        "changed_fields": sorted(applied_changes.keys()),
        "baseline": baseline_result,
        "simulated": simulated_result,
        "risk_delta": round(float(simulated_result["risk_score"] - baseline_result["risk_score"]), 6),
        "decision_changed": baseline_result["decision"] != simulated_result["decision"],
    }
