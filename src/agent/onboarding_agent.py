"""Dataset Onboarding Agent.

Turns an arbitrary labelled transaction CSV into a trained FraudPulse model by
reasoning about its schema with an LLM. The loop is a genuine
perceive → reason → act → verify → report cycle with tool use and
self-correction:

1. **Perceive**  — profile the dataset (``profiler.profile_dataset``).
2. **Reason**    — the LLM proposes a column-role mapping (target / amount /
   time / features) as strict JSON, with a rationale.
3. **Verify**    — the proposal is validated against the real data; on failure
   the errors are fed back and the LLM retries (bounded self-correction).
4. **Act**       — train XGBoost + IsolationForest on the mapped schema.
5. **Report**    — the LLM explains the resulting model card in plain language.

Policy: the agent runs in its intended LLM-reasoned mode by default and will
NOT silently fall back to heuristics (see :func:`src.llm.resolve_llm`). The
deterministic engine is available only as an explicit, clearly-labelled opt-in.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src import llm as llm_mod
from src.agent import profiler
from src.ml.config import DatasetConfig

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

_SCHEMA_SYSTEM = (
    "You are a senior ML data engineer onboarding a transaction dataset into a "
    "fraud-detection pipeline. You are given a JSON PROFILE of the dataset's "
    "columns. Identify the column roles and respond with STRICT JSON ONLY (no "
    "prose, no markdown fences) with exactly these keys:\n"
    '  "target_col": the binary fraud/label column,\n'
    '  "positive_label": the value of target_col that means fraud (e.g. 1),\n'
    '  "amount_col": the monetary amount column, or null,\n'
    '  "time_col": the elapsed-seconds/timestamp column, or null,\n'
    '  "feature_cols": list of NUMERIC predictor columns,\n'
    '  "name": a short human dataset name,\n'
    '  "reasoning": one to three sentences explaining your mapping.\n'
    "Rules: NEVER put target_col in feature_cols. Exclude identifier columns "
    "(account/card/transaction ids) and any obvious label-leakage. Only numeric "
    "columns may be features. Do not invent column names that aren't in the "
    "profile."
)

_EXPLAIN_SYSTEM = (
    "You are a fraud-analytics lead explaining a freshly trained model to a "
    "stakeholder. Given the MODELCARD_JSON, write a concise (3-5 sentence) plain-"
    "language summary. Emphasise why PR-AUC (not accuracy) is the headline metric "
    "under class imbalance, what the decision thresholds mean operationally, and "
    "the role of the unsupervised IsolationForest. Do not invent numbers."
)

_INSIGHT_SYSTEM = (
    "You are a fraud-analytics lead briefing a colleague on a transaction dataset "
    "that was just profiled and mapped. You are given DATASET_FACTS (computed from "
    "the real data) and the proposed column mapping. Write a concise 3-4 sentence "
    "plain-language insight conveying the TRUTH of this dataset: what it appears to "
    "represent, the fraud prevalence and what that imbalance implies for modelling "
    "(why PR-AUC matters, why accuracy misleads), notable data-quality signals "
    "(missingness, suspected id/leakage columns, tiny size), and whether it is "
    "suitable for training a fraud model. Be candid about weaknesses. Use ONLY the "
    "numbers in DATASET_FACTS — never invent figures. No markdown, no bullet lists."
)


def _dataset_facts(profile: Dict[str, Any], config: DatasetConfig) -> Dict[str, Any]:
    """Compute objective, LLM-groundable facts about the dataset from its profile
    and the resolved mapping (so the insight statement cites real numbers)."""
    by_name = {c["name"]: c for c in profile.get("columns", [])}
    n_rows = profile.get("n_rows", 0)
    tcol = by_name.get(config.target_col, {})
    vc = tcol.get("value_counts") or {}
    pos = str(config.positive_label)
    n_pos = int(vc.get(pos, 0))
    fraud_rate = round(100 * n_pos / n_rows, 4) if n_rows else None
    amt = by_name.get(config.amount_col or "", {})
    suspected_ids = [c["name"] for c in profile.get("columns", []) if c.get("looks_like_id")]
    high_missing = [c["name"] for c in profile.get("columns", []) if c.get("pct_missing", 0) > 20]
    return {
        "n_rows": n_rows,
        "n_features": len(config.feature_cols),
        "target_col": config.target_col,
        "positive_label": config.positive_label,
        "n_fraud_rows": n_pos,
        "fraud_rate_pct": fraud_rate,
        "class_balance": vc,
        "amount_range": ({"min": amt.get("min"), "max": amt.get("max"), "mean": amt.get("mean")}
                         if amt else None),
        "suspected_id_columns_excluded": suspected_ids,
        "high_missingness_columns": high_missing,
        "tiny_dataset": n_rows < 200,
    }


def _quality_assessment(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic data-quality verdict from the computed facts. This is a
    transparent rules-based gate (NOT the schema-validation check) so the panel
    can show whether the data has integrity issues, not just whether columns
    mapped. Severity: high -> flagged, medium -> review, else ready."""
    reasons: List[Dict[str, str]] = []
    ids = facts.get("suspected_id_columns_excluded") or []
    miss = facts.get("high_missingness_columns") or []
    n_feat = facts.get("n_features") or 0
    if ids:
        reasons.append({"severity": "high",
                        "text": f"Identifier/leakage risk — {', '.join(ids)} excluded from features"})
    if miss:
        reasons.append({"severity": "medium",
                        "text": f"Missing values in: {', '.join(miss)}"})
    if n_feat < 3:
        reasons.append({"severity": "medium", "text": f"Few predictor features ({n_feat})"})
    if facts.get("tiny_dataset"):
        reasons.append({"severity": "info",
                        "text": f"Small sample ({facts.get('n_rows')} rows) — validate on more data before production"})
    if any(r["severity"] == "high" for r in reasons):
        verdict = "flagged"
    elif any(r["severity"] == "medium" for r in reasons):
        verdict = "review"
    else:
        verdict = "ready"
    return {"verdict": verdict, "reasons": reasons}


def _generate_insight(llm, profile: Dict[str, Any], config: DatasetConfig) -> Optional[str]:
    """Produce a plain-language, AI-authored insight about the dataset. Non-fatal:
    returns None if the LLM call fails, so analysis still succeeds."""
    facts = _dataset_facts(profile, config)
    try:
        messages = [
            llm_mod.SystemMessage(_INSIGHT_SYSTEM),
            llm_mod.HumanMessage(
                f"DATASET_FACTS: {json.dumps(facts, default=str)}\n"
                f"PROPOSED_MAPPING: {json.dumps(config.to_dict(), default=str)}\n\n"
                "Write the insight."),
        ]
        text = llm.invoke(messages).content.strip()
        return text or None
    except Exception as exc:  # pragma: no cover - insight is non-critical
        logger.warning("Dataset insight generation failed: %s", exc)
        return None


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of an LLM reply."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(m.group(0))


def validate_mapping(mapping: Dict[str, Any], df: pd.DataFrame) -> Tuple[Optional[DatasetConfig], List[str]]:
    """Validate a proposed mapping against the real data.

    Returns ``(config, [])`` when valid, else ``(None, [errors])``.
    """
    errors: List[str] = []
    cols = set(df.columns)
    target = mapping.get("target_col")
    feature_cols = mapping.get("feature_cols") or []
    amount_col = mapping.get("amount_col")
    time_col = mapping.get("time_col")
    positive_label = mapping.get("positive_label", 1)

    if not target or target not in cols:
        errors.append(f"target_col '{target}' is not a column in the dataset.")
    else:
        ncls = int(df[target].nunique(dropna=True))
        if ncls != 2:
            errors.append(f"target_col '{target}' must be binary; it has {ncls} distinct values.")
        else:
            vals = set(df[target].dropna().unique().tolist())
            coerced = profiler._coerce(positive_label) if isinstance(positive_label, str) else positive_label
            if positive_label not in vals and coerced not in vals:
                errors.append(
                    f"positive_label {positive_label!r} is not a value of '{target}' (values: {sorted(map(str, vals))})."
                )
            else:
                pos_rate = float((df[target] == (coerced if coerced in vals else positive_label)).mean())
                if pos_rate in (0.0, 1.0):
                    errors.append(f"positive_label {positive_label!r} selects {pos_rate:.0%} of rows — not a usable fraud label.")

    if not feature_cols:
        errors.append("feature_cols is empty; need at least one numeric predictor.")
    for c in feature_cols:
        if c not in cols:
            errors.append(f"feature column '{c}' is not in the dataset.")
        elif c == target:
            errors.append(f"feature column '{c}' is the target — remove it (leakage).")
        elif not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"feature column '{c}' is not numeric.")
    for role, name in (("amount_col", amount_col), ("time_col", time_col)):
        if name is not None:
            if name not in cols:
                errors.append(f"{role} '{name}' is not in the dataset.")
            elif not pd.api.types.is_numeric_dtype(df[name]):
                errors.append(f"{role} '{name}' is not numeric.")

    if errors:
        return None, errors

    coerced = profiler._coerce(positive_label) if isinstance(positive_label, str) else positive_label
    vals = set(df[target].dropna().unique().tolist())
    config = DatasetConfig(
        target_col=target,
        feature_cols=[c for c in feature_cols if c != amount_col] + ([amount_col] if amount_col else []),
        amount_col=amount_col,
        time_col=time_col,
        positive_label=coerced if coerced in vals else positive_label,
        name=str(mapping.get("name") or "custom")[:60],
    )
    return config, []


def onboard(
    df: pd.DataFrame,
    *,
    allow_heuristic: bool = False,
    do_train: bool = True,
    model_dir: str = "models",
) -> Dict[str, Any]:
    """Run the onboarding agent on ``df`` and return a structured report."""
    profile = profiler.profile_dataset(df)

    # Enforce the no-silent-fallback policy. Raises LLMUnavailableError when no
    # LLM is configured and heuristics weren't explicitly allowed.
    llm = llm_mod.resolve_llm(allow_heuristic=allow_heuristic)
    is_offline = isinstance(llm, llm_mod.OfflineLLM)
    provider = getattr(llm, "provider", "unknown")

    report: Dict[str, Any] = {
        "status": "failed",
        "mode": "deterministic-fallback" if is_offline else "llm",
        "llm_used": not is_offline,
        "provider": provider,
        "model": provider.split(":", 1)[1] if ":" in provider else provider,
        "attempts": 0,
        "profile": profile,
        "config": None,
        "reasoning": None,
        "validation": {"ok": False, "errors": []},
        "model_card": None,
        "explanation": None,
        "notice": None,
    }
    if is_offline:
        report["notice"] = (
            "No LLM was configured; FraudPulse used its transparent deterministic "
            "rule-based engine. This mapping was NOT produced by an LLM. Set "
            "GROQ_API_KEY or ANTHROPIC_API_KEY for LLM-reasoned onboarding."
        )

    # ── reason + verify, with bounded self-correction ───────────────────────
    profile_json = json.dumps(profile)
    feedback = ""
    config: Optional[DatasetConfig] = None
    reasoning = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        report["attempts"] = attempt
        human = f"PROFILE_JSON: {profile_json}\n\n{feedback}Return the JSON mapping."
        messages = [llm_mod.SystemMessage(_SCHEMA_SYSTEM), llm_mod.HumanMessage(human)]
        try:
            raw = llm.invoke(messages).content
            mapping = _extract_json(raw)
        except Exception as exc:
            feedback = f"Your previous response could not be parsed as JSON ({exc}). "
            logger.warning("Onboarding attempt %d: parse failed: %s", attempt, exc)
            continue
        reasoning = mapping.get("reasoning")
        config, errors = validate_mapping(mapping, df)
        if config is not None:
            report["validation"] = {"ok": True, "errors": []}
            break
        report["validation"] = {"ok": False, "errors": errors}
        feedback = "Your previous proposal failed validation: " + "; ".join(errors) + ". Fix and "
        logger.warning("Onboarding attempt %d failed validation: %s", attempt, errors)

    if config is None:
        report["error"] = "Could not derive a valid schema mapping after %d attempts." % MAX_ATTEMPTS
        return report

    report["config"] = config.to_dict()
    report["reasoning"] = reasoning
    # AI-authored, plain-language statement of what the dataset actually is —
    # grounded in computed facts, generated by the LLM (skipped when offline).
    report["dataset_facts"] = _dataset_facts(profile, config)
    report["quality"] = _quality_assessment(report["dataset_facts"])
    if not is_offline:
        report["insight"] = _generate_insight(llm, profile, config)

    if not do_train:
        report["status"] = "mapped"
        return report

    # ── act: train on the mapped schema ─────────────────────────────────────
    from src.ml.train import train

    metadata = train(config=config, df=df, model_dir=model_dir)
    card = {k: metadata[k] for k in (
        "model_version", "dataset", "n_features", "fraud_rate", "pr_auc", "roc_auc",
        "flag_threshold", "review_threshold", "precision_at_flag", "recall_at_flag",
        "confusion_at_flag",
    )}
    card["target"] = config.target_col
    report["model_card"] = card
    report["status"] = "trained"

    # ── report: explain the model card ──────────────────────────────────────
    try:
        modelcard_json = json.dumps(card)
        ex_messages = [
            llm_mod.SystemMessage(_EXPLAIN_SYSTEM),
            llm_mod.HumanMessage(f"MODELCARD_JSON: {modelcard_json}\n\nWrite the summary."),
        ]
        report["explanation"] = llm.invoke(ex_messages).content
    except Exception as exc:  # pragma: no cover - explanation is non-critical
        logger.warning("Explanation step failed: %s", exc)
        report["explanation"] = None

    return report
