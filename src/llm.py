"""Pluggable LLM layer for the FraudPulse onboarding agent.

The agent must run in three environments without code changes:

1. **Groq** (free, no credit card) — set ``GROQ_API_KEY`` (default model
   ``llama-3.1-8b-instant``).
2. **Anthropic** — set ``ANTHROPIC_API_KEY`` to use Claude instead.
3. **Offline** — no key set. A deterministic stand-in runs the *same* agent
   loop using heuristics, so dataset onboarding works end-to-end with zero
   external dependencies (CI, local dev, demos without any account).

Every caller uses the ``.invoke([SystemMessage(...), HumanMessage(...)])``
interface and reads ``.provider`` for telemetry, so the back-ends are
interchangeable. This mirrors the CredAgent LLM layer for consistency across
the two portfolio projects.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)


# ── lightweight message objects (LangChain-compatible ``.content``) ──────────
class SystemMessage:
    def __init__(self, content: str):
        self.content = content


class HumanMessage:
    def __init__(self, content: str):
        self.content = content


class _Reply:
    def __init__(self, content: str):
        self.content = content


class LLMClient:
    """Thin wrapper around a LangChain chat model, carrying a provider label."""

    def __init__(self, llm, provider: str):
        self._llm = llm
        self.provider = provider

    def invoke(self, messages):
        # Translate our universal message objects into LangChain messages.
        from langchain_core.messages import HumanMessage as LCHuman, SystemMessage as LCSystem

        conv = []
        for m in messages:
            content = getattr(m, "content", str(m))
            role = m.__class__.__name__.lower()
            conv.append(LCSystem(content=content) if "system" in role else LCHuman(content=content))
        return self._llm.invoke(conv)


class OfflineLLM:
    """Deterministic fallback used when no LLM API key is configured.

    For schema mapping it runs the same heuristic the live agent validates
    against; for the model-card narrative it fills a template. Both read a
    machine-readable JSON block embedded in the human prompt (our own format),
    so the result is reliable rather than parsed out of free text.
    """

    provider = "offline-heuristic"

    def invoke(self, messages: List) -> _Reply:
        system, human = "", ""
        for m in messages:
            content = getattr(m, "content", str(m))
            if "system" in m.__class__.__name__.lower():
                system = content
            else:
                human = content
        s = system.lower()
        if "schema" in s or "column" in s:
            return _Reply(self._map_schema(human))
        return _Reply(self._explain(human))

    @staticmethod
    def _extract_json(text: str, key: str) -> dict:
        m = re.search(rf"{key}:\s*(\{{.*\}})", text, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}

    def _map_schema(self, human: str) -> str:
        from src.agent.profiler import heuristic_mapping
        profile = self._extract_json(human, "PROFILE_JSON")
        cfg = heuristic_mapping(profile)
        return json.dumps(cfg)

    def _explain(self, human: str) -> str:
        card = self._extract_json(human, "MODELCARD_JSON")
        pr = card.get("pr_auc")
        roc = card.get("roc_auc")
        rate = card.get("fraud_rate")
        nfeat = card.get("n_features")
        target = card.get("target", "the label")
        lines = [
            f"FraudPulse trained a supervised XGBoost classifier and an unsupervised "
            f"IsolationForest on your dataset, using '{target}' as the fraud label and "
            f"{nfeat} numeric feature(s).",
        ]
        if rate is not None:
            lines.append(
                f"Your positive (fraud) rate is {rate*100:.3f}%. Because that is highly "
                f"imbalanced, the model is selected and reported on PR-AUC (average "
                f"precision), not accuracy — accuracy would look high while missing fraud."
            )
        if pr is not None:
            lines.append(
                f"Held-out PR-AUC is {pr:.3f} (ROC-AUC {roc:.3f}). The flag threshold was "
                f"set to the F1-optimal operating point; a lower review band routes "
                f"borderline cases to analysts, and the IsolationForest escalates novel "
                f"anomalies the supervised model hasn't seen."
            )
        lines.append("[Generated offline — set GROQ_API_KEY for LLM-authored text.]")
        return " ".join(lines)


def get_llm(temperature: float = 0.0):
    """Return an LLM client. Resolution order: Groq → Anthropic → offline."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("your_"):
        try:
            from langchain_groq import ChatGroq

            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            return LLMClient(ChatGroq(model=model, temperature=temperature), f"groq:{model}")
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Groq init failed (%s); trying next backend.", exc)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and not anthropic_key.startswith("your_"):
        try:
            from langchain_anthropic import ChatAnthropic

            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            return LLMClient(ChatAnthropic(model=model, temperature=temperature), f"anthropic:{model}")
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Anthropic init failed (%s); using offline LLM.", exc)

    logger.info("No LLM API key configured — using deterministic offline LLM.")
    return OfflineLLM()


def llm_is_live() -> bool:
    """True when a real (non-offline) LLM backend is configured."""
    return not isinstance(get_llm(), OfflineLLM)


class LLMUnavailableError(RuntimeError):
    """Raised when the onboarding agent requires an LLM but none is configured.

    Surfacing this (rather than silently using the deterministic engine) keeps
    the agent honest: an LLM result is only ever labelled as such when a real
    model actually produced it.
    """


_NO_LLM_GUIDANCE = (
    "No LLM backend is configured, so the onboarding agent cannot run in its "
    "intended (LLM-reasoned) mode. Set GROQ_API_KEY (free, no card — "
    "https://console.groq.com) or ANTHROPIC_API_KEY and retry. To proceed with "
    "the transparent rule-based engine instead, pass allow_heuristic=True "
    "(CLI: --allow-heuristic, env: FRAUDPULSE_ALLOW_HEURISTIC=1) — its output is "
    "clearly labelled mode='deterministic-fallback' and is not presented as LLM "
    "reasoning."
)


def resolve_llm(allow_heuristic: bool = False):
    """Return an LLM for the agent, enforcing the no-silent-fallback policy.

    - A live backend (Groq/Anthropic) is returned when configured.
    - Otherwise, if ``allow_heuristic`` (explicit opt-in), the deterministic
      :class:`OfflineLLM` is returned — callers MUST label its output as a
      fallback (provider ``offline-heuristic``).
    - Otherwise, :class:`LLMUnavailableError` is raised with guidance.
    """
    llm = get_llm()
    if isinstance(llm, OfflineLLM):
        if os.getenv("FRAUDPULSE_ALLOW_HEURISTIC", "0") == "1":
            allow_heuristic = True
        if not allow_heuristic:
            raise LLMUnavailableError(_NO_LLM_GUIDANCE)
        logger.warning("Running onboarding in deterministic-fallback mode (no LLM configured).")
    return llm
