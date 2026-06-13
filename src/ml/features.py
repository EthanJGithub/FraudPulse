"""Feature engineering for the fraud model.

Driven by a :class:`~src.ml.config.DatasetConfig` so the same pipeline works on
the ULB dataset (Time, V1–V28 PCA, Amount, Class) and on arbitrary labelled
transaction datasets onboarded by the agent. For any config we keep the chosen
numeric features, and — when those roles are present — add a log-scaled amount
and an hour-of-day derived from an elapsed-seconds column.

The module-level ``V_COLUMNS`` / ``FEATURE_COLUMNS`` / ``TARGET`` constants are
kept for the ULB default so existing imports continue to work.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.ml.config import DatasetConfig, DEFAULT_CONFIG, _V_COLUMNS

V_COLUMNS = _V_COLUMNS
FEATURE_COLUMNS = DEFAULT_CONFIG.model_feature_cols
TARGET = DEFAULT_CONFIG.target_col


def engineer_features(df: pd.DataFrame, config: Optional[DatasetConfig] = None) -> pd.DataFrame:
    """Build the model feature matrix for ``df`` according to ``config``."""
    config = config or DEFAULT_CONFIG
    df = df.copy()
    for col in config.feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if config.amount_col:
        amount = pd.to_numeric(df.get(config.amount_col, 0), errors="coerce").fillna(0.0)
        df["log_amount"] = np.log1p(amount.clip(lower=0))
    if config.time_col:
        time_col = pd.to_numeric(df.get(config.time_col, 0), errors="coerce").fillna(0)
        df["hour"] = ((time_col / 3600) % 24).astype(float)
    return df[config.model_feature_cols]


def engineer_single(raw: Dict[str, Any], config: Optional[DatasetConfig] = None) -> Dict[str, Any]:
    """Engineer features for one transaction dict (inference time)."""
    config = config or DEFAULT_CONFIG
    df = pd.DataFrame([dict(raw)])
    needed = set(config.feature_cols)
    if config.amount_col:
        needed.add(config.amount_col)
    if config.time_col:
        needed.add(config.time_col)
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0
    return engineer_features(df, config).iloc[0].to_dict()
