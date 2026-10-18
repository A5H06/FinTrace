"""Train the supervised XGBoost and unsupervised Isolation Forest engines."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, ISOLATION_FOREST_WEIGHT,
    RANDOM_STATE, TEST_SIZE, XGBOOST_WEIGHT,
)
from src.feature_engineering import build_feature_frame
from src.preprocessor import clean_dataframe

LOGGER = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


def _build_classifier_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def _build_isolation_preprocessor(numeric_features: list[str]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", RobustScaler()),
    ])


def _build_classifier(scale_pos_weight: float, random_state: int):
    if XGBOOST_AVAILABLE:
        jobs = int(os.environ.get("FINTRACE_N_JOBS", str(max(1, (os.cpu_count() or 2) - 1))))
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            scale_pos_weight=float(scale_pos_weight),
            tree_method="hist",
            n_jobs=jobs,
            random_state=random_state,
        )
        return classifier, "xgboost"
    LOGGER.warning("XGBoost could not be imported; using a RandomForest fallback. Install the IBM Z-compatible XGBoost build for the primary backend.")
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=int(os.environ.get("FINTRACE_N_JOBS", str(max(1, (os.cpu_count() or 2) - 1)))),
        random_state=random_state,
    )
    return classifier, "random_forest_fallback"


def train_hybrid_models(data: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> dict:
    """Fit preprocessing and both models; return artifact bundle plus test split.

    All learned transforms are fitted only on the training fold. The Isolation
    Forest's anomaly score is calibrated to the empirical training-fold score
    distribution so it can be combined with supervised probability on a 0..1 scale.
    """
    raw = clean_dataframe(data)
    if "is_fraud" not in raw.columns:
        raise ValueError("The training data must include is_fraud")
    y = raw["is_fraud"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least one row in each fraud class")
    features = build_feature_frame(raw)
    numeric_features = [column for column in features.columns if column not in CATEGORICAL_FEATURES]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in features.columns]
    indices = np.arange(len(raw))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y,
    )
    x_train = features.iloc[train_indices].reset_index(drop=True)
    x_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = y[train_indices]
    y_test = y[test_indices]
    raw_test = raw.iloc[test_indices].reset_index(drop=True)

    classifier_preprocessor = _build_classifier_preprocessor(numeric_features, categorical_features)
    x_train_encoded = classifier_preprocessor.fit_transform(x_train)
    x_test_encoded = classifier_preprocessor.transform(x_test)

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = float(negatives / max(positives, 1))
    classifier, classifier_backend = _build_classifier(scale_pos_weight, random_state)
    LOGGER.info(
        "Training supervised engine on %s rows (legitimate=%s, fraud=%s; scale_pos_weight=%.4f)",
        len(y_train), negatives, positives, scale_pos_weight,
    )
    classifier.fit(x_train_encoded, y_train)

    numeric_train = x_train[numeric_features]
    numeric_test = x_test[numeric_features]
    isolation_preprocessor = _build_isolation_preprocessor(numeric_features)
    x_train_anomaly = isolation_preprocessor.fit_transform(numeric_train)
    x_test_anomaly = isolation_preprocessor.transform(numeric_test)
    isolation_forest = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=int(os.environ.get("FINTRACE_N_JOBS", str(max(1, (os.cpu_count() or 2) - 1)))),
    )
    LOGGER.info("Training Isolation Forest behavioral engine on %s rows", len(y_train))
    isolation_forest.fit(x_train_anomaly)
    reference_decisions = np.sort(isolation_forest.decision_function(x_train_anomaly).astype(float))

    bundle = {
        "model_name": "FinTrace Hybrid AP Fraud Model",
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "classifier_backend": classifier_backend,
        "classifier": classifier,
        "classifier_preprocessor": classifier_preprocessor,
        "isolation_forest": isolation_forest,
        "isolation_preprocessor": isolation_preprocessor,
        "isolation_reference_decisions": reference_decisions,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": list(FEATURE_COLUMNS),
        "scale_pos_weight": scale_pos_weight,
        "training_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "xgboost_weight": XGBOOST_WEIGHT,
        "isolation_forest_weight": ISOLATION_FOREST_WEIGHT,
    }
    return {
        "bundle": bundle,
        "x_test_features": x_test,
        "y_test": y_test,
        "raw_test": raw_test,
    }


def predict_components(bundle: dict, feature_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return XGBoost probability, calibrated anomaly score, and base composite risk."""
    features = feature_frame.loc[:, bundle["feature_columns"]]
    encoded = bundle["classifier_preprocessor"].transform(features)
    probabilities = bundle["classifier"].predict_proba(encoded)[:, 1].astype(float)
    numeric = features.loc[:, bundle["numeric_features"]]
    anomaly_matrix = bundle["isolation_preprocessor"].transform(numeric)
    raw_decision = bundle["isolation_forest"].decision_function(anomaly_matrix).astype(float)
    reference = np.asarray(bundle["isolation_reference_decisions"], dtype=float)
    percentile = np.searchsorted(reference, raw_decision, side="right") / max(len(reference), 1)
    anomaly_scores = np.clip(1.0 - percentile, 0.0, 1.0)
    risk = (
        float(bundle["xgboost_weight"]) * probabilities
        + float(bundle["isolation_forest_weight"]) * anomaly_scores
    )
    return probabilities, anomaly_scores, np.clip(risk, 0.0, 1.0)
