"""Transaction stream simulator.

Replays REAL transactions (from a committed sample of the ULB dataset) through
the scoring pipeline to drive the live ops dashboard. Two modes:

- ``seed()`` — score the sample into the store in one pass (with ground-truth
  labels) so the dashboard shows real detection quality immediately. Used on
  API startup when the store is empty.
- ``replay()`` — post transactions to the running API at an interval, to
  demonstrate live streaming.

    python -m src.stream.simulator --mode replay --rate 5
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from src import store
from src.ml import scorer
from src.ml.features import V_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SAMPLE_PATH = "data/sample_transactions.csv"


def _row_to_txn(row) -> dict:
    raw = {"Amount": float(row.get("Amount", 0)), "Time": float(row.get("Time", 0))}
    for v in V_COLUMNS:
        if v in row:
            raw[v] = float(row[v])
    return raw


def seed() -> int:
    import os
    if not os.path.exists(SAMPLE_PATH) or not scorer.is_ready():
        logger.warning("Seed skipped: missing sample data or models.")
        return 0
    df = pd.read_csv(SAMPLE_PATH)
    # Vectorized scoring (two matrix calls) keeps startup fast.
    from src.ml.features import engineer_features
    X = engineer_features(df)
    results = scorer.score_batch(X)
    labels = df["Class"].tolist() if "Class" in df.columns else [None] * len(df)
    amounts = df["Amount"].tolist()
    rows = [(f"seed-{i:05d}", amounts[i], results[i], labels[i]) for i in range(len(df))]
    store.bulk_log(rows)
    logger.info("Seeded %d scored transactions.", len(rows))
    return len(rows)


def ensure_seeded() -> int:
    if store.count() > 0:
        return 0
    return seed()


_sample_df = None


def sample_payloads(n: int = 1) -> list:
    """Return ``n`` random REAL transactions from the committed sample, shaped as
    ``/score`` request payloads. Used by the dashboard's live stream so the feed
    shows varied real amounts and a realistic fraud/legit mix (no hardcoded
    vectors). The sample CSV is loaded once and cached."""
    global _sample_df
    import os
    if _sample_df is None:
        if not os.path.exists(SAMPLE_PATH):
            return []
        _sample_df = pd.read_csv(SAMPLE_PATH)
    n = max(1, min(int(n), 50))
    rows = _sample_df.sample(min(n, len(_sample_df)))
    out = []
    for _, row in rows.iterrows():
        feats = {v: float(row[v]) for v in V_COLUMNS if v in row and pd.notna(row[v])}
        out.append({
            "Amount": float(row.get("Amount", 0) or 0),
            "Time": float(row.get("Time", 0) or 0),
            "features": feats,
        })
    return out


def replay(rate: float, base_url: str) -> None:
    """Post sample transactions to the running API at ``rate`` per second."""
    import requests
    df = pd.read_csv(SAMPLE_PATH).sample(frac=1).reset_index(drop=True)
    delay = 1.0 / max(rate, 0.1)
    for i, row in df.iterrows():
        raw = _row_to_txn(row)
        payload = {"txn_id": f"live-{int(time.time()*1000)}-{i}",
                   "Amount": raw["Amount"], "Time": raw["Time"],
                   "features": {k: raw[k] for k in raw if k.startswith("V")}}
        try:
            r = requests.post(f"{base_url}/api/v1/score", json=payload, timeout=10).json()
            logger.info("%s -> %s (p=%.3f)", payload["txn_id"], r["decision"], r["fraud_probability"])
        except Exception as exc:
            logger.warning("post failed: %s", exc)
        time.sleep(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seed", "replay"], default="seed")
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    if args.mode == "seed":
        print(f"Seeded rows: {seed()} | store total: {store.count()}")
    else:
        replay(args.rate, args.url)


if __name__ == "__main__":
    main()
