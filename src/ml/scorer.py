"""Fraud scoring — XGBoost probability + IsolationForest anomaly + decision."""
import json
import logging
from typing import Any, Dict

import joblib
import pandas as pd

from src.ml.config import DatasetConfig
from src.ml.features import engineer_single

logger = logging.getLogger(__name__)

_xgb = None
_iso = None
_meta = None
_cfg = None


def _load():
    global _xgb, _iso, _meta, _cfg
    if _xgb is None:
        _xgb = joblib.load("models/xgb_fraud.pkl")
        _iso = joblib.load("models/isolation_forest.pkl")
        with open("models/metadata.json") as f:
            _meta = json.load(f)
        _cfg = DatasetConfig.from_metadata(_meta)
        logger.info("Fraud models loaded (%s).", _meta["model_version"])


def is_ready() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def metadata() -> Dict[str, Any]:
    _load()
    return _meta


def _decide(prob: float, is_anomaly: bool) -> str:
    if prob >= _meta["flag_threshold"]:
        return "FLAG"
    if prob >= _meta["review_threshold"] or is_anomaly:
        return "REVIEW"
    return "ALLOW"


def score(raw_txn: Dict[str, Any]) -> Dict[str, Any]:
    """Score one transaction. Returns probability, anomaly, and a decision."""
    _load()
    X = pd.DataFrame([engineer_single(raw_txn, _cfg)])[_cfg.model_feature_cols]
    prob = float(_xgb.predict_proba(X)[0, 1])
    anomaly_raw = float(-_iso.score_samples(X)[0])      # higher = more anomalous
    is_anomaly = bool(_iso.predict(X)[0] == -1)
    p99 = _meta.get("anomaly_percentiles", {}).get("99", 1.0) or 1.0
    return {
        "fraud_probability": round(prob, 6),
        "anomaly_score": round(max(0.0, min(anomaly_raw / p99, 1.5)), 4),
        "is_anomaly": is_anomaly,
        "decision": _decide(prob, is_anomaly),
        "model_version": _meta["model_version"],
    }


def score_batch(X) -> list:
    """Vectorized scoring of a feature matrix (DataFrame with FEATURE_COLUMNS).
    Two matrix calls instead of N per-row calls — used for fast seeding."""
    _load()
    X = X[_cfg.model_feature_cols]
    probs = _xgb.predict_proba(X)[:, 1]
    anomaly_raw = -_iso.score_samples(X)
    flags = _iso.predict(X)
    p99 = _meta.get("anomaly_percentiles", {}).get("99", 1.0) or 1.0
    out = []
    for prob, araw, fl in zip(probs, anomaly_raw, flags):
        prob = float(prob)
        is_anom = bool(fl == -1)
        out.append({
            "fraud_probability": round(prob, 6),
            "anomaly_score": round(max(0.0, min(float(araw) / p99, 1.5)), 4),
            "is_anomaly": is_anom,
            "decision": _decide(prob, is_anom),
            "model_version": _meta["model_version"],
        })
    return out
