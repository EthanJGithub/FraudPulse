import os
import tempfile

# Isolate the store and skip startup seeding for fast, side-effect-free tests.
os.environ["FRAUDPULSE_DB"] = os.path.join(tempfile.gettempdir(), "fraudpulse_test.db")
os.environ["FRAUDPULSE_AUTOSEED"] = "0"
for _s in ("", "-wal", "-shm"):
    _p = os.environ["FRAUDPULSE_DB"] + _s
    if os.path.exists(_p):
        os.remove(_p)

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


# A real fraudulent transaction from the dataset (high V-feature signal).
FRAUD_TXN = {
    "txn_id": "test-fraud", "Amount": 0.0, "Time": 406.0,
    "features": {f"V{i}": v for i, v in enumerate([
        -2.3122, 1.9520, -1.6098, 3.9979, -0.5222, -1.4265, -2.5374, 1.3917,
        -2.7700, -2.7723, 3.2020, -2.8999, -0.5952, -4.2892, 0.3897, -1.1407,
        -2.8301, -0.0168, 0.4170, 0.1267, 0.5172, -0.0350, -0.4652, 0.3201,
        0.0445, 0.1779, 0.2611, -0.1432], start=1)},
}
LEGIT_TXN = {
    "txn_id": "test-legit", "Amount": 149.62, "Time": 0.0,
    "features": {"V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782, "V14": -0.3111},
}


@pytest.fixture
def fraud_txn():
    return dict(FRAUD_TXN)


@pytest.fixture
def legit_txn():
    return dict(LEGIT_TXN)
