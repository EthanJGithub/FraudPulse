# Compliance & Governance — FraudPulse

> How FraudPulse handles the governance concerns a fintech risk/compliance team
> cares about for an automated fraud system: model risk, PII minimization,
> customer-impact of false positives, auditability, and LLM guardrails. Every
> claim points at the code that backs it. Where something is **designed-for but
> not yet enforced**, it says so plainly.

Governance frame: model-risk management (in the spirit of **SR 11-7** model
governance), payment-data handling discipline, and the principle that an
**automated decision affecting a customer must be explainable and reviewable.**

---

## 1. Model risk — honest metrics under extreme imbalance

The real dataset is **0.17% fraud** ([models/metadata.json](models/metadata.json)).
FraudPulse reports the metrics that survive that imbalance and refuses the ones
that flatter it:

- **PR-AUC = 0.8798** is the headline (not accuracy, which is ~99.8% for a
  do-nothing model, and not ROC-AUC, which is optimistic under imbalance).
- **Operating-point precision/recall are pinned**: precision **0.942**, recall
  **0.827** at the FLAG threshold, so the precision/recall trade-off the business
  is accepting is explicit, not hidden behind a single AUC number.
- **Thresholds are versioned, not hand-tuned at runtime** (`flag_threshold`,
  `review_threshold` in model metadata; seeds pinned). The same input yields the
  same decision across runs and environments — a reproducibility requirement for
  model governance.

---

## 2. Customer impact — the REVIEW tier exists on purpose

A fraud false-positive is a **declined legitimate customer** — a real harm, not a
rounding error. FraudPulse never collapses to a binary block:

- The three-way decision **ALLOW / REVIEW / FLAG** ([src/ml/scorer.py](src/ml/scorer.py))
  routes ambiguous cases to **REVIEW** — an explicit human-in-the-loop band —
  rather than forcing a false accept or false decline.
- The dashboard surfaces the real decision mix (~12% flagged/review, not
  "everything blocked"), so operators see the true rate of customer-affecting
  actions, color-coded and filterable.

---

## 3. Data minimization & PII

- **The model features carry no direct PII.** The ULB dataset's predictive
  features are **PCA-anonymized components (V1–V28)** plus `Amount` and `Time` —
  by construction there are no names, card numbers, or identifiers in the feature
  space ([src/ml/config.py](src/ml/config.py)).
- **The store persists only scoring metadata** ([src/store.py](src/store.py)): a
  transaction id, amount, model outputs (probability, anomaly, decision), and the
  ground-truth label when replaying labelled data. No cardholder PII columns
  exist in the schema.
- **The onboarding agent profiles, it does not exfiltrate.** [profiler.py](src/agent/profiler.py)
  summarizes column *statistics/roles* for the LLM, and the agent is instructed to
  **exclude identifier columns** (account/card/transaction ids) and label-leakage
  from the feature set — a data-minimization rule enforced in the agent's prompt
  and validation.
- **Uploads are capped and sampled** — `/onboard/upload` rejects payloads over
  2 MB and samples to 3,000 rows for profiling ([src/api/routes.py](src/api/routes.py)),
  limiting how much user-supplied data is processed.

---

## 4. LLM guardrails — "as powerful as the guardrails around it"

The LLM in FraudPulse is **never** the scorer. It is confined to schema reasoning
and explanation, and it is fenced in three ways:

1. **No silent fallback.** Without an LLM the onboarding agent raises
   `LLMUnavailableError` → HTTP **503** with actionable guidance, unless the
   deterministic engine is *explicitly* opted into (`allow_heuristic`). Offline
   output is clearly labelled and never presented as LLM reasoning
   ([src/llm.py](src/llm.py), [src/agent/onboarding_agent.py](src/agent/onboarding_agent.py)).
2. **Validation-bounded self-correction.** The LLM's proposed schema is checked
   against the real data; invalid proposals (e.g. target in the feature set,
   non-numeric features, invented columns) are rejected and fed back for up to 3
   bounded retries. The deterministic checker, not the LLM, has the final say.
3. **Analyze-only on shared infrastructure.** The browser onboarding endpoint
   forces `do_train=False`, so an untrusted visitor cannot retrain and corrupt the
   shared model ([src/api/routes.py](src/api/routes.py)).

---

## 5. Auditability & observability

- **Every scored transaction is persisted** with its model version, probability,
  anomaly score, and decision ([src/store.py](src/store.py)) — decisions are
  reconstructable, not ephemeral.
- **Live detection quality** — a confusion matrix and live precision/recall are
  computed from labelled replays, so model performance is observable in operation.
- **Model version stamping** — the resolved model version is logged at load and
  attached to every score response, so any decision is traceable to a model build.
- **Onboarding runs are self-documenting** — the agent returns the resolved
  `provider:model`, attempt count, validation result, proposed schema, and
  reasoning, so the *agent's* decisions are auditable too.

> **Roadmap (Weeks 3–4 observability hardening):** per-request **latency**
> histograms and **PSI model-drift** monitoring. CredAgent already ships PSI
> drift ([../credagent/src/drift.py](../credagent/src/drift.py)); the same
> reference-window + stable/moderate/significant pattern ports directly here.

---

## 6. Governance scorecard

| Control | Status |
|---------|--------|
| Imbalance-honest metrics (PR-AUC headline, pinned operating point) | **Implemented** ([models/metadata.json](models/metadata.json)) |
| Human-in-the-loop REVIEW tier | **Implemented** ([src/ml/scorer.py](src/ml/scorer.py)) |
| Versioned/deterministic thresholds & seeds | **Implemented** |
| PII-free feature space + minimized store | **Implemented** ([src/ml/config.py](src/ml/config.py), [src/store.py](src/store.py)) |
| No-silent-fallback LLM degradation | **Implemented** ([src/llm.py](src/llm.py)) |
| Validation-bounded agent self-correction | **Implemented** ([src/agent/onboarding_agent.py](src/agent/onboarding_agent.py)) |
| Analyze-only / capped untrusted uploads | **Implemented** ([src/api/routes.py](src/api/routes.py)) |
| Per-request latency monitoring | **Roadmap** (Weeks 3–4) |
| PSI model-drift monitoring | **Roadmap** — port from CredAgent |
| Role-based access control on the API | **Designed-for, not enforced** (single-tenant demo) |
| Immutable/append-only audit log | **Roadmap** |

---

## 7. Testing the governance behaviors

Governance behaviors are covered by hermetic tests (no network/keys):
[tests/test_api.py](tests/test_api.py) verifies decision persistence, per-decision
histograms, the filterable feed, and the upload guards (empty / oversize);
[tests/test_onboarding.py](tests/test_onboarding.py) verifies the agent's
schema-mapping and self-correction. The principle is **test the contract** — the
guardrails are verified, not assumed.
