"""In-process request metrics: latency, error rate, throughput.

A production fraud service is judged on operational health, not just model
quality. This module is a lightweight, dependency-free metrics collector that the
API middleware feeds and the monitoring endpoint reads. It keeps a bounded
rolling window of recent request latencies (so memory is constant) and running
counters, and exposes p50/p95/p99 latency, error rate, and throughput.

Thread-safe (the ASGI server may serve requests concurrently). Process-local by
design — for a multi-replica deployment this would be exported to Prometheus /
OpenTelemetry; the collection points are the same.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Tuple

_MAX_SAMPLES = 1000  # rolling window of recent requests

_lock = threading.Lock()
_started_at = time.time()
_total = 0
_errors = 0
# (path, latency_ms, is_error, monotonic_ts)
_samples: Deque[Tuple[str, float, bool, float]] = deque(maxlen=_MAX_SAMPLES)


def record(path: str, latency_ms: float, status_code: int) -> None:
    """Record one completed request."""
    global _total, _errors
    is_error = status_code >= 500
    with _lock:
        _total += 1
        if is_error:
            _errors += 1
        _samples.append((path, latency_ms, is_error, time.monotonic()))


def _percentile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def snapshot() -> Dict[str, Any]:
    """Current operational metrics over the rolling window."""
    now = time.monotonic()
    with _lock:
        samples = list(_samples)
        total, errors = _total, _errors
        uptime = time.time() - _started_at
    lats = sorted(s[1] for s in samples)
    window_errors = sum(1 for s in samples if s[2])
    # Throughput over the last 60s of the window.
    recent = [s for s in samples if now - s[3] <= 60]
    rpm = len(recent)
    return {
        "uptime_seconds": round(uptime, 1),
        "requests_total": total,
        "errors_total": errors,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "window_size": len(samples),
        "window_error_rate": round(window_errors / len(samples), 4) if samples else 0.0,
        "latency_ms": {
            "p50": round(_percentile(lats, 0.50), 2),
            "p95": round(_percentile(lats, 0.95), 2),
            "p99": round(_percentile(lats, 0.99), 2),
            "max": round(lats[-1], 2) if lats else 0.0,
        },
        "throughput_rpm": rpm,
    }


def reset() -> None:
    """Clear all metrics (used by tests for isolation)."""
    global _total, _errors, _started_at
    with _lock:
        _total = 0
        _errors = 0
        _samples.clear()
        _started_at = time.time()
