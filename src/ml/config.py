"""Dataset schema configuration.

A :class:`DatasetConfig` decouples the training / scoring pipeline from the
specific ULB column names, so FraudPulse can be trained on *any* labelled
transaction dataset. The onboarding agent produces one of these from a raw CSV;
the rest of the pipeline (feature engineering, training, scoring) is driven by
it. The ULB schema is expressed as :data:`DEFAULT_CONFIG`, so the committed
demo model and dashboard behave exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetConfig:
    """How to read a transaction dataset.

    Parameters
    ----------
    target_col:
        Column holding the binary fraud label.
    feature_cols:
        Base numeric columns used as model features (excludes derived ones).
    amount_col:
        Optional monetary-amount column. If set, a ``log_amount`` feature and
        amount-at-risk reporting are derived from it.
    time_col:
        Optional elapsed-seconds column. If set, an ``hour`` (intraday) feature
        is derived from it.
    positive_label:
        The value of ``target_col`` that means fraud (default ``1``).
    name:
        Human-readable dataset name, recorded in the model card.
    """

    target_col: str
    feature_cols: List[str]
    amount_col: Optional[str] = None
    time_col: Optional[str] = None
    positive_label: Any = 1
    name: str = "custom"

    @property
    def model_feature_cols(self) -> List[str]:
        """Full ordered feature list the model trains/scses on (base + derived)."""
        cols = list(self.feature_cols)
        if self.amount_col:
            cols.append("log_amount")
        if self.time_col:
            cols.append("hour")
        return cols

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_col": self.target_col,
            "feature_cols": list(self.feature_cols),
            "amount_col": self.amount_col,
            "time_col": self.time_col,
            "positive_label": self.positive_label,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        return cls(
            target_col=d["target_col"],
            feature_cols=list(d["feature_cols"]),
            amount_col=d.get("amount_col"),
            time_col=d.get("time_col"),
            positive_label=d.get("positive_label", 1),
            name=d.get("name", "custom"),
        )

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "DatasetConfig":
        """Reconstruct the config from a model's ``metadata.json``.

        Falls back to the ULB default for legacy metadata with no schema block,
        so previously-trained models keep scoring unchanged.
        """
        schema = meta.get("schema")
        if schema:
            return cls.from_dict(schema)
        return DEFAULT_CONFIG


# ── ULB Credit-Card Fraud dataset (the default / committed demo) ─────────────
_V_COLUMNS = [f"V{i}" for i in range(1, 29)]

DEFAULT_CONFIG = DatasetConfig(
    target_col="Class",
    feature_cols=_V_COLUMNS + ["Amount"],
    amount_col="Amount",
    time_col="Time",
    positive_label=1,
    name="ULB Credit-Card Fraud",
)
