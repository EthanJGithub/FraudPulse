# Architecture — FraudPulse

> How FraudPulse is built and **why** each component was chosen, written for an
> engineering reviewer evaluating production-readiness. Governance and model-risk
> obligations are documented separately in [COMPLIANCE.md](COMPLIANCE.md).

---

## 1. System at a glance

FraudPulse is a **real-time transaction fraud-scoring service** plus an **agentic
dataset-onboarding** capability. A transaction is scored by two complementary
models and routed to an operational decision (ALLOW / REVIEW / FLAG); a live
React + Nivo dashboard shows the decision stream, score distribution, and
detection quality. Separately, an LLM agent can onboard an *arbitrary* labelled
CSV by reasoning about its schema.

```
Transaction ─▶ Feature engineering ─▶ XGBoost (P(fraud)) ─┐
                                       IsolationForest ────┴▶ decision ─▶ persisted + dashboard
                                       (anomaly)              ALLOW/REVIEW/FLAG

Arbitrary CSV ─▶ Onboarding Agent: profile ─▶ LLM maps schema ─▶ validate/retry ─▶ (train) ─▶ explain
```

| Layer | Component | Path |
|-------|-----------|------|
| Scoring | XGBoost + IsolationForest + decision logic | [src/ml/scorer.py](src/ml/scorer.py) |
| Schema abstraction | `DatasetConfig` (schema-agnostic pipeline) | [src/ml/config.py](src/ml/config.py) |
| Onboarding agent | profile → reason → validate → train → explain | [src/agent/onboarding_agent.py](src/agent/onboarding_agent.py), [src/agent/profiler.py](src/agent/profiler.py) |
| LLM abstraction | Groq → Anthropic → offline | [src/llm.py](src/llm.py) |
| Persistence | SQLite transaction/decision store | [src/store.py](src/store.py) |
| Serving | FastAPI scoring + monitoring API | [src/api/](src/api/) |
| Dashboard | Vite + React + Nivo live ops UI | [frontend/](frontend/) |
| Stream | Real-transaction replay simulator | [src/stream/simulator.py](src/stream/simulator.py) |

---

## 2. Why these choices (the defensible decisions)

### Why two models — XGBoost **and** IsolationForest
Fraud has two failure modes: patterns we have *seen labelled before*, and patterns
we *haven't*. A single supervised model catches the former and misses novel
attacks. FraudPulse pairs:

- **XGBoost (supervised)** — learns known fraud signal from labels; produces a
  calibrated fraud probability.
- **IsolationForest (unsupervised)** — flags statistical anomalies regardless of
  label, catching novel/unseen patterns.

The decision logic ([src/ml/scorer.py](src/ml/scorer.py)) combines them: a high
probability FLAGs; a moderate probability **or** an anomaly routes to REVIEW. This
defense-in-depth is the senior stance — no single model is trusted to be complete.

### Why PR-AUC is the headline metric (not accuracy or ROC-AUC)
The real ULB dataset is **0.17% fraud** ([models/metadata.json](models/metadata.json),
`fraud_rate` 0.001727). Under that imbalance, accuracy is meaningless (predict
"legit" always → 99.83% accurate, 0% useful) and ROC-AUC is optimistic. FraudPulse
reports **PR-AUC = 0.8798** as the headline, with operating-point precision/recall
(**precision 0.942 / recall 0.827 at the FLAG threshold**). Choosing the metric
that survives extreme imbalance is itself a model-governance signal.

### Why a three-way decision (ALLOW / REVIEW / FLAG), not a binary block
A binary fraud/not-fraud switch forces every borderline case into a false
accept or a false decline — both of which have real customer/loss cost. The
**REVIEW** tier is an explicit human-in-the-loop band for ambiguous cases. The two
thresholds are pinned in model metadata (`flag_threshold` 0.9387, `review_threshold`
0.05), so behavior is **deterministic and reproducible**, not hand-tuned at runtime.

### Why `DatasetConfig` (schema-agnostic pipeline)
The entire feature/train/score pipeline is driven by a [DatasetConfig](src/ml/config.py)
that names the target / amount / time / feature columns. The committed ULB model
is just `DEFAULT_CONFIG`. This decoupling is what makes the **onboarding agent**
possible: the agent's whole job is to *produce a valid `DatasetConfig` for an
unseen CSV*, after which the existing pipeline runs unchanged.

### Why the onboarding agent is a real agent (not an AutoML wrapper)
[src/agent/onboarding_agent.py](src/agent/onboarding_agent.py) runs a genuine
**perceive → reason → act → verify → report** loop with tool use and bounded
self-correction:
1. **Perceive** — profile the dataset's columns ([profiler.py](src/agent/profiler.py)).
2. **Reason** — the LLM proposes a column-role mapping as strict JSON.
3. **Verify** — the proposal is validated against the *real data*; on failure the
   errors are fed back and the LLM retries (up to `MAX_ATTEMPTS = 3`).
4. **Act** — train XGBoost + IsolationForest on the mapped schema (optional).
5. **Report** — the LLM explains the resulting model card in plain language.

The self-correction-against-validation loop is the differentiator: the agent is
**constrained by a deterministic checker**, not trusted to be right first time.

### Why the Groq → Anthropic → offline LLM ladder
[src/llm.py](src/llm.py) resolves a backend in a fixed order behind one interface,
so the service runs in CI, local dev, and the hosted demo without code changes.
Critically, the onboarding agent **refuses to silently fall back**: without an LLM
it raises `LLMUnavailableError` (HTTP 503 with guidance) unless the deterministic
engine is *explicitly* opted into via `allow_heuristic`. A degraded mapping is
never passed off as the LLM's reasoning.

---

## 3. Serving & API

FastAPI ([src/api/routes.py](src/api/routes.py)) exposes:

| Endpoint | Purpose |
|----------|---------|
| `POST /score` | Score one transaction → probability, anomaly, decision |
| `GET /monitoring/stats` | Decision mix, amount-at-risk, per-decision score histograms, confusion matrix |
| `GET /monitoring/transactions?decision=` | Filterable live decision feed |
| `GET /monitoring/alerts` | FLAG/REVIEW alert feed |
| `GET /sample` | Replay real transactions for the live stream |
| `POST /onboard` | Onboarding agent on a server-side CSV path |
| `POST /onboard/upload` | **Analyze-only** browser onboarding (CSV text, capped, never retrains) |

The dashboard ([frontend/](frontend/)) is the mandated **Vite + React + Nivo**
stack: live decision feed with decision filtering, a Nivo donut for decision mix,
a per-decision score-distribution histogram, and detection-quality KPIs.

---

## 4. Observability & monitoring

- **Detection quality, live** — when replaying labelled real data, the store
  computes a **confusion matrix** and the dashboard shows live precision/recall
  ([src/store.py](src/store.py) `stats`), so model quality is visible in
  operation, not just at train time.
- **Score distribution** — 20-bin fraud-probability histograms, overall and
  **per decision** (ALL / ALLOW / REVIEW / FLAG), so a shift in the score
  distribution is visually obvious.
- **Amount at risk** — summed FLAGged amount, the operational loss-exposure view.
- **Structured logging** — model load, scoring, and onboarding steps log via the
  standard `logging` module with model version stamped.
- **Latency / drift** — see [COMPLIANCE.md §5](COMPLIANCE.md) and the roadmap;
  per-request latency timing and PSI drift are the next observability hardening
  items (CredAgent already ships PSI; the same pattern ports here).

---

## 5. Known limitations / roadmap

- **Single shared model** on the server. The `/onboard/upload` endpoint is
  therefore **analyze-only** (`do_train=False` forced) and size-capped, so a
  visitor cannot retrain and clobber the shared demo for everyone.
- **No request-latency histogram / PSI drift endpoint yet** — present in CredAgent,
  scheduled to port here (Weeks 3–4 observability hardening).
- **No authn/authz** on the API (single-tenant demo); designed to accept an auth
  layer without restructuring the routes.
- **SQLite** persistence; production multi-writer load would move to Postgres (the
  store module isolates this).
