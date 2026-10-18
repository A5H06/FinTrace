"""Run the complete FinTrace training, evaluation, export, and inference pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import joblib

from src.config import (
    DATASET_PATH, DEMO_METRICS_PATH, DEMO_MODEL_PATH, METRICS_PATH,
    MODEL_DIR, MODEL_PATH, OUTPUT_DIR, RANDOM_STATE,
)
from src.data_loader import load_dataset, validate_dataset
from src.demo_data import generate_demo_dataset
from src.evaluate_model import evaluate_model
from src.feature_engineering import build_feature_frame
from src.inference_engine import score_transaction
from src.preprocessor import clean_dataframe
from src.train_model import train_hybrid_models


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _example_transaction(raw_record: dict) -> dict:
    excluded = {"is_fraud", "fraud_type", "explanations"}
    return {key: value for key, value in raw_record.items() if key not in excluded and value is not None}


def run_pipeline(demo: bool = False, demo_rows: int = 6000) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if demo:
        logging.getLogger(__name__).warning(
            "DEMO MODE: training and metrics are based on synthetic data and are not evidence of real-world accuracy."
        )
        dataset = validate_dataset(clean_dataframe(generate_demo_dataset(rows=demo_rows, random_state=RANDOM_STATE)))
        dataset_source = "synthetic demo data"
    else:
        dataset = load_dataset(DATASET_PATH)
        dataset_source = DATASET_PATH

    logging.getLogger(__name__).info("Starting sequential pipeline on %s", dataset_source)
    training = train_hybrid_models(dataset, random_state=RANDOM_STATE)
    bundle = training["bundle"]
    metrics = evaluate_model(bundle, training["x_test_features"], training["y_test"], training["raw_test"])
    metrics["dataset_source"] = dataset_source
    metrics["evaluation_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["data_rows"] = int(len(dataset))
    metrics["data_columns"] = int(len(dataset.columns))
    metrics["demo_mode"] = bool(demo)
    bundle["training_dataset_source"] = dataset_source
    bundle["demo_mode"] = bool(demo)
    artifact_path = DEMO_MODEL_PATH if demo else MODEL_PATH
    metrics_path = DEMO_METRICS_PATH if demo else METRICS_PATH

    joblib.dump(bundle, artifact_path, compress=3)
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False, default=str)

    example = _example_transaction(dataset.drop(columns=["is_fraud", "fraud_type", "explanations"], errors="ignore").iloc[0].to_dict())
    example_result = score_transaction(example, bundle)
    print("\n========== FINTRACE TRAINING SUMMARY ==========")
    print(f"Dataset source       : {dataset_source}")
    print(f"Rows / columns       : {len(dataset):,} / {len(dataset.columns)}")
    print(f"Fraud prevalence     : {dataset['is_fraud'].mean():.2%}")
    print(f"Classifier backend   : {bundle['classifier_backend']}")
    print(f"scale_pos_weight     : {bundle['scale_pos_weight']:.4f}")
    print(f"Accuracy             : {metrics['metrics_at_0_50']['accuracy']:.4f}")
    print(f"Precision            : {metrics['metrics_at_0_50']['precision']:.4f}")
    print(f"Recall               : {metrics['metrics_at_0_50']['recall']:.4f}")
    print(f"F1                   : {metrics['metrics_at_0_50']['f1_score']:.4f}")
    print(f"ROC-AUC              : {metrics['metrics_at_0_50']['roc_auc']}")
    print(f"APPROVE / HOLD / BLOCK: {metrics['firewall_operations']['approve_count']:,} / "
          f"{metrics['firewall_operations']['hold_count']:,} / {metrics['firewall_operations']['block_count']:,}")
    print(f"Model artifact       : {artifact_path}")
    print(f"Metrics JSON         : {metrics_path}")
    print("\n========== LIVE INFERENCE SMOKE TEST ==========")
    print(json.dumps({"invoice_id": example.get("invoice_id"), **example_result}, indent=2, default=str))
    if demo:
        print("\nNOTE: The model artifact and metrics are synthetic-demo outputs. They are saved under *_demo names and must not be used as production performance evidence.")
    return {"bundle": bundle, "metrics": metrics, "example_result": example_result}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the FinTrace hybrid AP fraud firewall.")
    parser.add_argument("--demo", action="store_true", help="Run a smoke test with explicitly synthetic data if the real workbook is unavailable.")
    parser.add_argument("--demo-rows", type=int, default=6000, help="Number of synthetic rows for --demo (minimum 500).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    try:
        run_pipeline(demo=args.demo, demo_rows=args.demo_rows)
    except Exception as exc:
        logging.getLogger(__name__).exception("FinTrace pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
