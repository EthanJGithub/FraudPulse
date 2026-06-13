"""Dataset profiling — the onboarding agent's perception tool.

``profile_dataset`` produces a compact, machine-readable description of an
arbitrary transaction CSV (column types, cardinality, missingness, sample
values, binary-ness, id/leakage hints). The agent feeds this to the LLM to
reason about column roles.

``heuristic_mapping`` derives a column-role mapping from a profile using
deterministic rules. It serves two purposes: (1) the transparent
deterministic-fallback engine when an LLM is explicitly not required, and
(2) a sanity baseline the agent can compare the LLM's proposal against.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Name hints (substring match, case-insensitive) used by the heuristic.
_TARGET_HINTS = ("fraud", "class", "label", "is_fraud", "isfraud", "target", "y")
_AMOUNT_HINTS = ("amount", "amt", "transactionamount", "value", "price", "sum")
_TIME_HINTS = ("time", "timestamp", "seconds", "elapsed", "step")
_ID_HINTS = ("id", "uuid", "guid", "transactionid", "txn", "account", "card", "ref")

_MAX_UNIQUE_LISTED = 12


def _name_matches(name: str, hints) -> bool:
    low = re.sub(r"[^a-z0-9]", "", name.lower())
    return any(h in low for h in hints)


def profile_dataset(df: pd.DataFrame, max_sample: int = 5) -> Dict[str, Any]:
    """Return a compact JSON-serialisable profile of ``df``."""
    n = len(df)
    cols: List[Dict[str, Any]] = []
    for name in df.columns:
        s = df[name]
        nun = int(s.nunique(dropna=True))
        is_numeric = bool(pd.api.types.is_numeric_dtype(s))
        col: Dict[str, Any] = {
            "name": str(name),
            "dtype": str(s.dtype),
            "is_numeric": is_numeric,
            "n_unique": nun,
            "pct_missing": round(float(s.isna().mean()) * 100, 2),
            "is_binary": nun == 2,
            "looks_like_id": bool(_name_matches(str(name), _ID_HINTS) and nun >= 0.9 * max(n, 1)),
        }
        if nun <= _MAX_UNIQUE_LISTED:
            vc = s.dropna().value_counts()
            col["unique_values"] = [
                (v.item() if hasattr(v, "item") else v) for v in vc.index.tolist()
            ]
            col["value_counts"] = {str(k): int(v) for k, v in vc.items()}
        if is_numeric and nun:
            col["min"] = float(np.nanmin(s.values))
            col["max"] = float(np.nanmax(s.values))
            col["mean"] = round(float(np.nanmean(s.values)), 4)
        col["sample"] = [
            (v.item() if hasattr(v, "item") else v) for v in s.dropna().head(max_sample).tolist()
        ]
        cols.append(col)
    return {"n_rows": int(n), "n_cols": int(df.shape[1]), "columns": cols}


def _binary_is_01(col: Dict[str, Any]) -> bool:
    vals = col.get("unique_values")
    if not vals:
        return False
    try:
        return set(int(v) for v in vals) <= {0, 1}
    except (ValueError, TypeError):
        return False


def heuristic_mapping(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic column-role mapping from a profile.

    Returns a dict compatible with ``DatasetConfig.from_dict``.
    """
    columns = profile.get("columns", [])
    by_name = {c["name"]: c for c in columns}
    numeric = [c for c in columns if c.get("is_numeric")]
    binary = [c for c in columns if c.get("is_binary")]

    # ── target: a binary column, preferring a fraud-ish name, then a {0,1} col.
    target = None
    named_bin = [c for c in binary if _name_matches(c["name"], _TARGET_HINTS)]
    if named_bin:
        target = named_bin[0]["name"]
    else:
        zero_one = [c for c in binary if _binary_is_01(c)]
        if zero_one:
            # the rarer-positive one is the likely fraud flag; fall back to first
            target = zero_one[0]["name"]
        elif binary:
            target = binary[0]["name"]

    # ── amount / time by name among numeric columns
    amount = next((c["name"] for c in numeric if _name_matches(c["name"], _AMOUNT_HINTS)), None)
    time_col = next((c["name"] for c in numeric if _name_matches(c["name"], _TIME_HINTS)), None)

    # ── features: numeric columns minus target, raw time, and id-like columns
    exclude = {target, time_col}
    feature_cols = [
        c["name"] for c in numeric
        if c["name"] not in exclude and not c.get("looks_like_id")
    ]

    # ── positive label: the minority class of the target (fraud is rare)
    positive_label: Any = 1
    if target and by_name.get(target, {}).get("value_counts"):
        vc = by_name[target]["value_counts"]
        minority = min(vc, key=vc.get)
        positive_label = _coerce(minority)

    return {
        "target_col": target,
        "feature_cols": feature_cols,
        "amount_col": amount if (amount and amount in by_name) else None,
        "time_col": time_col,
        "positive_label": positive_label,
        "name": "custom",
    }


def _coerce(v: str):
    """Best-effort cast of a stringified label back to int/float when possible."""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return v
