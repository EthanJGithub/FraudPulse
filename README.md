# FraudPulse ⚡

> **Real-Time Transaction Fraud Detection**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react)](https://react.dev)
[![Nivo](https://img.shields.io/badge/Charts-Nivo-ff6b6b)](https://nivo.rocks)
[![Tests](https://img.shields.io/badge/tests-13_passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A streaming fraud-detection service that scores card transactions in real time,
combining a supervised **XGBoost** classifier with an unsupervised
**IsolationForest** anomaly detector, behind a **FastAPI** API and a live
**React + Nivo** operations dashboard.

Trained on the real **ULB Credit Card Fraud** dataset — 284,807 European
card transactions, 492 frauds (**0.172%**), the standard benchmark for extreme
class imbalance.

---

## Why it's interesting

- **Extreme imbalance done right.** With 0.17% positives, accuracy and even
  ROC-AUC are misleading — the model is selected and reported on **PR-AUC
  (average precision)**, the correct metric. `scale_pos_weight` handles the skew.
- **Two complementary models.** XGBoost gives the primary fraud probability;
  IsolationForest adds an unsupervised anomaly score that can catch novel
  patterns the supervised model hasn't seen.
- **Real-time architecture.** A decoupled FastAPI scoring service + a stream
  simulator that replays real transactions + a live polling dashboard.
- **Operations focus.** Three-tier decisions (ALLOW / REVIEW / FLAG), an alert
  feed, amount-at-risk, and live precision/recall against ground-truth labels.
- **Agentic onboarding.** An LLM agent ingests *any* labelled transaction CSV,
  reasons about its schema, self-corrects against validation, retrains, and
  explains the model card — so the system is useful on real data out of the box,
  not just the benchmark. See [Bring your own data](#bring-your-own-data--the-onboarding-agent-).

---

## Model Performance (held-out test set)

| Metric | Value |
|---|---|
| Algorithm | XGBoost + IsolationForest on 22 features (V1–V28 PCA + Amount + derived) |
| **PR-AUC (average precision)** | **0.880** |
| ROC-AUC | 0.978 |
| Precision @ flag threshold | **0.94** |
| Recall @ flag threshold | **0.83** |
| Confusion @ flag | TP 81 · FP 5 · FN 17 · TN 56,859 |
| Fraud base rate | 0.172% |

Full model card: [`models/metadata.json`](models/metadata.json).

---

## Architecture

```
React + Nivo dashboard (Vite, :5174)         Stream simulator
   │  poll /monitoring/stats, /alerts            │  replay real txns → /score
   ▼                                             ▼
FastAPI  (:8000)  /score  /monitoring/stats  /monitoring/alerts  /model/info
   │
   ├─ XGBoost (fraud probability)  +  IsolationForest (anomaly score)
   └─ SQLite store (scored transactions, alerts, ground-truth labels)
```

Decision tiers (on fraud probability): `≥ flag_threshold → FLAG (block)`,
`≥ review_threshold → REVIEW (analyst)`, else `ALLOW`. An IsolationForest
anomaly also escalates to REVIEW.

---

## Quickstart

```bash
git clone <repo> && cd fraudpulse
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -r requirements.txt

# 1. Download the real dataset (~150 MB, no Kaggle account) and train
python -m src.ml.get_data
python -m src.ml.train          # PR-AUC ~0.88, ~1–2 min

# 2. Run the API (auto-seeds 2,000 real transactions for the dashboard)
uvicorn src.api.main:app --reload                    # http://localhost:8000/docs

# 3. Run the dashboard (Node 18+)
cd frontend && npm install && npm run dev            # http://localhost:5174

# Optional: stream live transactions at 5/sec
python -m src.stream.simulator --mode replay --rate 5

# Tests
pytest                                                # 13 passing
```

The repo ships with the trained `models/` and a committed
`data/sample_transactions.csv` (2,000 real transactions incl. 250 frauds), so
the API + dashboard run without the 150 MB dataset download.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`  | `/api/v1/health` | Service + model status |
| `POST` | `/api/v1/score` | Score one transaction → probability, anomaly, decision |
| `GET`  | `/api/v1/monitoring/stats` | Volume, decision mix, score histogram, live precision/recall |
| `GET`  | `/api/v1/monitoring/alerts` | Recent FLAG/REVIEW alerts |
| `GET`  | `/api/v1/model/info` | PR-AUC, ROC-AUC, thresholds, confusion |
| `POST` | `/api/v1/onboard` | **Onboarding agent**: analyse a CSV, map its schema, (optionally) retrain |

---

## Bring your own data — the Onboarding Agent 🤖

FraudPulse isn't hard-wired to the ULB columns. Point it at **any labelled
transaction CSV** and an LLM agent makes it usable — no manual config:

```bash
# LLM-reasoned onboarding (needs GROQ_API_KEY or ANTHROPIC_API_KEY)
python -m src.agent.onboard --data path/to/your_transactions.csv

# map the schema without retraining
python -m src.agent.onboard --data your_transactions.csv --no-train
```

The agent runs a genuine **perceive → reason → verify → act → report** loop:

1. **Perceive** — profiles every column (dtype, cardinality, missingness, sample
   values, id/leakage hints).
2. **Reason** — an LLM proposes the column roles (fraud label, amount, time,
   numeric features) as strict JSON, with a rationale.
3. **Verify** — the proposal is validated against the real data; on failure the
   errors are fed back and the LLM **self-corrects** (bounded retries).
4. **Act** — trains XGBoost + IsolationForest on the mapped schema; the schema
   is written into the model card so scoring reproduces it exactly.
5. **Report** — the LLM explains the resulting model card in plain language.

It handles real-world messiness — string labels (`"fraud"`/`"legit"`), arbitrary
column names, identifier columns to exclude — and picks the minority class as
the positive label automatically.

**No silent degradation.** By default the agent *requires* a real LLM; if none
is configured it stops with actionable guidance rather than quietly guessing.
A transparent deterministic engine is available only as an explicit opt-in
(`--allow-heuristic`), and its output is clearly labelled `mode:
"deterministic-fallback"` — never presented as LLM reasoning.

## Deployment

The repo ships a `Procfile` and `runtime.txt`, so the API deploys to Render or
Railway as a Python web service.

**API (Render / Railway)**
- Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` (Procfile).
- Environment variables:
  - `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`) — **required for the onboarding
    agent**. Without it the agent returns `503` by design (no silent fallback);
    scoring and the dashboard still work.
  - optional: `GROQ_MODEL=llama-3.1-8b-instant`, `FRAUDPULSE_AUTOSEED=1`.
- The SQLite store is ephemeral on these hosts and reseeds from the committed
  sample on startup — fine for a demo.

**Dashboard (Vercel)**
- Build dir `frontend/`, framework Vite.
- Environment variable: `VITE_API_BASE=https://<your-api-host>/api/v1`.

**Verify everything is online** — one call confirms models *and* the agent:

```bash
curl https://<your-api-host>/api/v1/health
# { "status":"ok", "models_loaded":true, "transactions_scored":2000,
#   "onboarding_agent_online":true, "llm_provider":"groq:llama-3.1-8b-instant" }
```

`onboarding_agent_online:false` (or `llm_provider:"offline"`) means the LLM key
isn't set in the host environment.

## Project Structure

```
fraudpulse/
├── src/
│   ├── ml/          # get_data, features, train (XGBoost+IsoForest), scorer
│   ├── api/         # FastAPI app, routes, schemas
│   ├── stream/      # transaction stream simulator (seed + live replay)
│   ├── agent/       # LLM onboarding agent (profiler + reason/verify loop)
│   ├── llm.py       # pluggable LLM (Groq/Anthropic) — no silent fallback
│   └── store.py     # SQLite store (transactions, alerts, labels)
├── frontend/        # React (Vite) + Nivo live ops dashboard
├── models/          # trained models + metadata (committed)
├── data/sample_transactions.csv   # 2,000 real txns for seeding (committed)
└── tests/           # pytest suite
```

## License

MIT
