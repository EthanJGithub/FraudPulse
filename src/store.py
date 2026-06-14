"""SQLite store for scored transactions + fraud alerts (live ops + audit)."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

DB_PATH = os.getenv("FRAUDPULSE_DB", "data/transactions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    txn_id            TEXT PRIMARY KEY,
    ts                TEXT,
    amount            REAL,
    fraud_probability REAL,
    anomaly_score     REAL,
    is_anomaly        INTEGER,
    decision          TEXT,
    label             INTEGER          -- ground-truth Class when replaying real data (else NULL)
);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def count() -> int:
    try:
        with _connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def log_txn(txn_id: str, amount: float, result: Dict[str, Any], label=None) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO transactions (txn_id, ts, amount, fraud_probability, anomaly_score, "
            "is_anomaly, decision, label) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(txn_id) DO UPDATE SET ts=excluded.ts, fraud_probability=excluded.fraud_probability, "
            "anomaly_score=excluded.anomaly_score, is_anomaly=excluded.is_anomaly, decision=excluded.decision",
            (txn_id, datetime.now().isoformat(timespec="seconds"), float(amount),
             result["fraud_probability"], result["anomaly_score"],
             int(result["is_anomaly"]), result["decision"],
             None if label is None else int(label)),
        )


def bulk_log(rows: List[tuple]) -> None:
    """Fast batch insert. Each row: (txn_id, amount, result_dict, label)."""
    init_db()
    ts = datetime.now().isoformat(timespec="seconds")
    params = [
        (tid, ts, float(amt), res["fraud_probability"], res["anomaly_score"],
         int(res["is_anomaly"]), res["decision"], None if lbl is None else int(lbl))
        for (tid, amt, res, lbl) in rows
    ]
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO transactions (txn_id, ts, amount, fraud_probability, "
            "anomaly_score, is_anomaly, decision, label) VALUES (?,?,?,?,?,?,?,?)",
            params,
        )


def recent_alerts(limit: int = 25) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM transactions WHERE decision IN ('FLAG','REVIEW') ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_transactions(limit: int = 25, decision: str | None = None) -> List[Dict[str, Any]]:
    """Most recent scored transactions, optionally filtered to one decision
    (ALLOW / REVIEW / FLAG). Powers the filterable live transaction feed."""
    init_db()
    with _connect() as conn:
        if decision in ("ALLOW", "REVIEW", "FLAG"):
            rows = conn.execute(
                "SELECT * FROM transactions WHERE decision=? ORDER BY ts DESC LIMIT ?",
                (decision, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM transactions ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def stats() -> Dict[str, Any]:
    init_db()
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        out: Dict[str, Any] = {
            "total": total,
            "decision_counts": {"ALLOW": 0, "REVIEW": 0, "FLAG": 0},
            "amount_at_risk": 0.0,
            "alerts": 0,
            "score_histogram": [],
            "confusion": None,
        }
        if not total:
            return out
        for d, c in conn.execute("SELECT decision, COUNT(*) FROM transactions GROUP BY decision"):
            out["decision_counts"][d] = c
        out["alerts"] = out["decision_counts"]["FLAG"] + out["decision_counts"]["REVIEW"]
        out["amount_at_risk"] = round(conn.execute(
            "SELECT COALESCE(SUM(amount),0) FROM transactions WHERE decision='FLAG'").fetchone()[0], 2)
        # Probability histograms (20 bins) for the score-distribution chart,
        # both overall and per-decision so the chart can follow the feed filter.
        hist_all = [0] * 20
        hist_by = {"ALLOW": [0] * 20, "REVIEW": [0] * 20, "FLAG": [0] * 20}
        for p, d in conn.execute("SELECT fraud_probability, decision FROM transactions"):
            b = min(int((p or 0) * 20), 19)
            hist_all[b] += 1
            if d in hist_by:
                hist_by[d][b] += 1

        def _bins(h):
            return [{"bin": f"{i*5}-{i*5+5}%", "count": h[i]} for i in range(20)]

        out["score_histogram"] = _bins(hist_all)
        out["histograms"] = {"ALL": _bins(hist_all), **{k: _bins(v) for k, v in hist_by.items()}}
        # If ground-truth labels exist (replaying real data), report detection quality.
        labeled = conn.execute("SELECT COUNT(*) FROM transactions WHERE label IS NOT NULL").fetchone()[0]
        if labeled:
            tp = conn.execute("SELECT COUNT(*) FROM transactions WHERE label=1 AND decision='FLAG'").fetchone()[0]
            fn = conn.execute("SELECT COUNT(*) FROM transactions WHERE label=1 AND decision!='FLAG'").fetchone()[0]
            fp = conn.execute("SELECT COUNT(*) FROM transactions WHERE label=0 AND decision='FLAG'").fetchone()[0]
            tn = conn.execute("SELECT COUNT(*) FROM transactions WHERE label=0 AND decision!='FLAG'").fetchone()[0]
            out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        return out
