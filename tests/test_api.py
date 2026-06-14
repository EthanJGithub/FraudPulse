def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["models_loaded"] is True


def test_model_info(client):
    m = client.get("/api/v1/model/info").json()
    assert m["model_version"] == "fraud-v1.0"
    assert m["pr_auc"] >= 0.7
    assert 0 < m["flag_threshold"] <= 1


def test_score_fraud_flagged(client, fraud_txn):
    r = client.post("/api/v1/score", json=fraud_txn)
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] in ("FLAG", "REVIEW")
    assert d["fraud_probability"] > 0.1


def test_score_legit_allowed(client, legit_txn):
    d = client.post("/api/v1/score", json=legit_txn).json()
    assert d["decision"] == "ALLOW"
    assert d["fraud_probability"] < 0.5


def test_score_persists_and_stats(client, fraud_txn, legit_txn):
    client.post("/api/v1/score", json=fraud_txn)
    client.post("/api/v1/score", json=legit_txn)
    s = client.get("/api/v1/monitoring/stats").json()
    assert s["total"] >= 2
    assert set(s["decision_counts"]) == {"ALLOW", "REVIEW", "FLAG"}
    assert len(s["score_histogram"]) == 20


def test_alerts_endpoint(client, fraud_txn):
    client.post("/api/v1/score", json=fraud_txn)
    a = client.get("/api/v1/monitoring/alerts").json()
    assert "alerts" in a


def test_transactions_feed_and_filter(client, fraud_txn, legit_txn):
    client.post("/api/v1/score", json=fraud_txn)
    client.post("/api/v1/score", json=legit_txn)
    # Unfiltered feed returns recent transactions of any decision.
    all_txns = client.get("/api/v1/monitoring/transactions?limit=50").json()["transactions"]
    assert all_txns and any(t["decision"] == "ALLOW" for t in all_txns)
    # Filtered feed only returns the requested decision.
    allow = client.get("/api/v1/monitoring/transactions?decision=ALLOW&limit=50").json()["transactions"]
    assert allow and all(t["decision"] == "ALLOW" for t in allow)


def test_stats_per_decision_histograms(client, fraud_txn, legit_txn):
    client.post("/api/v1/score", json=fraud_txn)
    client.post("/api/v1/score", json=legit_txn)
    s = client.get("/api/v1/monitoring/stats").json()
    assert set(s["histograms"]) == {"ALL", "ALLOW", "REVIEW", "FLAG"}
    for h in s["histograms"].values():
        assert len(h) == 20


def test_onboard_upload_rejects_empty(client):
    r = client.post("/api/v1/onboard/upload", json={"csv_text": "   "})
    assert r.status_code == 400


def test_onboard_upload_rejects_oversize(client):
    big = "a,b\n" + ("1,2\n" * 600_000)  # > 2 MB
    r = client.post("/api/v1/onboard/upload", json={"csv_text": big})
    assert r.status_code == 413
