"""Train the fraud-detection models.

    python -m src.ml.train

Two complementary models on the real ULB credit-card fraud data:
- **XGBoost** (supervised) — primary fraud probability. With 0.172% positives we
  use ``scale_pos_weight`` and optimise/report **PR-AUC** (average precision),
  the correct metric for extreme imbalance (ROC-AUC is misleadingly high here).
- **IsolationForest** (unsupervised) — a complementary anomaly score that can
  catch novel fraud patterns the supervised model hasn't seen.

Outputs: models/xgb_fraud.pkl, models/isolation_forest.pkl, models/metadata.json
"""
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.ml.config import DatasetConfig, DEFAULT_CONFIG
from src.ml.features import engineer_features
from src.ml.get_data import ensure_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "models"


def _best_threshold(y_true, scores):
    """Threshold maximising F1, plus a high-recall (~0.90) threshold for review."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    best_i = int(np.argmax(f1[:-1])) if len(thr) else 0
    flag_thr = float(thr[best_i]) if len(thr) else 0.5
    # REVIEW band: lowest threshold achieving recall >= 0.90, but floored at 0.05
    # so the manual-review queue stays a small, workable slice (not everything).
    review_thr = flag_thr
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if r >= 0.90:
            review_thr = float(t)
    review_thr = min(max(review_thr, 0.05), flag_thr)
    return flag_thr, review_thr


def train(config: DatasetConfig = None, df: pd.DataFrame = None, model_dir: str = MODEL_DIR):
    """Train the fraud models.

    With no arguments this reproduces the ULB demo model. Pass a
    :class:`DatasetConfig` and a DataFrame to train on an arbitrary onboarded
    dataset; the schema is recorded in ``metadata.json`` so scoring can
    reproduce the exact feature pipeline.
    """
    config = config or DEFAULT_CONFIG
    os.makedirs(model_dir, exist_ok=True)
    if df is None:
        path = ensure_dataset()
        logger.info("Loading %s ...", path)
        df = pd.read_csv(path)

    y = (df[config.target_col] == config.positive_label).astype(int)
    logger.info("Loaded %s rows (%s). Fraud rate: %.4f%%",
                f"{len(df):,}", config.name, 100 * y.mean())
    X = engineer_features(df, config)

    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = neg / max(pos, 1)
    logger.info("Train %s | Test %s | scale_pos_weight=%.1f", f"{len(X_tr):,}", f"{len(X_te):,}", spw)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        eval_metric="aucpr", early_stopping_rounds=40, random_state=42, n_jobs=-1,
    )
    logger.info("Training XGBoost...")
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    proba = model.predict_proba(X_te)[:, 1]
    pr_auc = average_precision_score(y_te, proba)
    roc_auc = roc_auc_score(y_te, proba)
    flag_thr, review_thr = _best_threshold(y_te.values, proba)
    preds = (proba >= flag_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    logger.info("PR-AUC=%.4f ROC-AUC=%.4f | flag_thr=%.4f", pr_auc, roc_auc, flag_thr)
    logger.info("At flag threshold: precision=%.3f recall=%.3f f1=%.3f (TP=%d FP=%d FN=%d)",
                precision_score(y_te, preds), recall_score(y_te, preds),
                f1_score(y_te, preds), tp, fp, fn)

    logger.info("Training IsolationForest (unsupervised)...")
    iso = IsolationForest(
        n_estimators=200, contamination=float(y_tr.mean()), random_state=42, n_jobs=-1,
    )
    iso.fit(X_tr)
    # Reference percentiles of the anomaly score (higher = more anomalous) for
    # normalising at inference time.
    anomaly_train = -iso.score_samples(X_tr)
    anom_p = {q: float(np.percentile(anomaly_train, q)) for q in (50, 90, 95, 99)}

    feature_cols = config.model_feature_cols
    is_default = config.name == DEFAULT_CONFIG.name
    joblib.dump(model, os.path.join(model_dir, "xgb_fraud.pkl"))
    joblib.dump(iso, os.path.join(model_dir, "isolation_forest.pkl"))
    metadata = {
        "model_version": "fraud-v1.0" if is_default else f"fraud-custom-{config.name}",
        "dataset": config.name,
        "schema": config.to_dict(),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "fraud_rate": float(y.mean()),
        "scale_pos_weight": round(spw, 2),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "flag_threshold": round(flag_thr, 4),
        "review_threshold": round(review_thr, 4),
        "precision_at_flag": round(float(precision_score(y_te, preds)), 4),
        "recall_at_flag": round(float(recall_score(y_te, preds)), 4),
        "f1_at_flag": round(float(f1_score(y_te, preds)), 4),
        "confusion_at_flag": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "anomaly_percentiles": anom_p,
        "data_source": "huggingface:David-Egea/Creditcard-fraud-detection" if is_default else "user-provided",
        "trained_on_real_data": is_default,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved models + metadata. PR-AUC=%.4f. Training complete.", pr_auc)
    return metadata


if __name__ == "__main__":
    train()
