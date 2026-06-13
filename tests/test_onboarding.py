"""Tests for the dataset onboarding agent (profiler, validation, agent loop).

These run fully offline (deterministic engine) by clearing any LLM keys, so the
suite is hermetic and needs no network or API account.
"""
import numpy as np
import pandas as pd
import pytest

from src.agent import profiler
from src.agent.onboarding_agent import onboard, validate_mapping
from src.llm import LLMUnavailableError


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Force the offline path so tests are deterministic and key-free."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FRAUDPULSE_ALLOW_HEURISTIC", raising=False)


def _make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.gamma(2, 1, n)
    label = (score > np.quantile(score, 0.9)).astype(int)
    return pd.DataFrame({
        "account_id": [f"A{i:05d}" for i in range(n)],   # id -> excluded
        "risk_a": score,
        "risk_b": rng.normal(0, 1, n),
        "amount": rng.exponential(50, n).round(2),
        "elapsed_seconds": rng.integers(0, 86400, n).astype(float),
        "is_fraud": label,
    })


def test_profile_shapes_and_flags():
    df = _make_df()
    prof = profiler.profile_dataset(df)
    assert prof["n_rows"] == len(df) and prof["n_cols"] == df.shape[1]
    by = {c["name"]: c for c in prof["columns"]}
    assert by["is_fraud"]["is_binary"] is True
    assert by["account_id"]["looks_like_id"] is True
    assert by["risk_a"]["is_numeric"] is True


def test_heuristic_mapping_picks_roles():
    df = _make_df()
    prof = profiler.profile_dataset(df)
    m = profiler.heuristic_mapping(prof)
    assert m["target_col"] == "is_fraud"
    assert m["amount_col"] == "amount"
    assert m["time_col"] == "elapsed_seconds"
    assert "account_id" not in m["feature_cols"]      # id excluded
    assert "is_fraud" not in m["feature_cols"]         # target excluded
    assert "elapsed_seconds" not in m["feature_cols"]  # raw time excluded


def test_validate_rejects_target_in_features():
    df = _make_df()
    bad = {"target_col": "is_fraud", "feature_cols": ["is_fraud", "risk_a"],
           "amount_col": None, "time_col": None, "positive_label": 1}
    cfg, errors = validate_mapping(bad, df)
    assert cfg is None and any("target" in e for e in errors)


def test_validate_rejects_unknown_column():
    df = _make_df()
    bad = {"target_col": "is_fraud", "feature_cols": ["does_not_exist"],
           "amount_col": None, "time_col": None, "positive_label": 1}
    cfg, errors = validate_mapping(bad, df)
    assert cfg is None and any("does_not_exist" in e for e in errors)


def test_onboard_requires_llm_by_default():
    df = _make_df()
    with pytest.raises(LLMUnavailableError):
        onboard(df, allow_heuristic=False, do_train=False)


def test_onboard_heuristic_maps_without_training():
    df = _make_df()
    r = onboard(df, allow_heuristic=True, do_train=False)
    assert r["status"] == "mapped"
    assert r["llm_used"] is False and r["mode"] == "deterministic-fallback"
    assert r["notice"] and "NOT" in r["notice"]
    assert r["config"]["target_col"] == "is_fraud"
    assert r["validation"]["ok"] is True


def test_onboard_heuristic_trains_to_tmpdir(tmp_path):
    df = _make_df(n=400)
    r = onboard(df, allow_heuristic=True, do_train=True, model_dir=str(tmp_path))
    assert r["status"] == "trained"
    card = r["model_card"]
    assert 0.0 <= card["pr_auc"] <= 1.0
    assert card["target"] == "is_fraud"
    assert (tmp_path / "xgb_fraud.pkl").exists()
    assert (tmp_path / "metadata.json").exists()
