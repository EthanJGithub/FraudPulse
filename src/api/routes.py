import os
import uuid

from fastapi import APIRouter, HTTPException

from src import store
from src.api.schemas import TransactionRequest, ScoreResponse, HealthResponse, OnboardRequest
from src.ml import scorer

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health():
    from src.llm import agent_status

    agent = agent_status()
    return HealthResponse(
        status="ok",
        models_loaded=scorer.is_ready(),
        transactions_scored=store.count(),
        onboarding_agent_online=agent["online"],
        llm_provider=agent["provider"],
    )


@router.get("/model/info")
async def model_info():
    if not scorer.is_ready():
        raise HTTPException(status_code=503, detail="Models not trained. Run: python -m src.ml.train")
    return scorer.metadata()


@router.post("/score", response_model=ScoreResponse)
async def score_txn(txn: TransactionRequest):
    if not scorer.is_ready():
        raise HTTPException(status_code=503, detail="Models not trained yet.")
    raw = {"Amount": txn.Amount, "Time": txn.Time, **(txn.features or {})}
    result = scorer.score(raw)
    txn_id = txn.txn_id or f"txn-{uuid.uuid4().hex[:12]}"
    store.log_txn(txn_id, txn.Amount, result)
    return ScoreResponse(txn_id=txn_id, **result)


@router.get("/monitoring/stats")
async def monitoring_stats():
    return store.stats()


@router.get("/monitoring/alerts")
async def monitoring_alerts(limit: int = 25):
    return {"alerts": store.recent_alerts(limit=limit)}


@router.get("/sample")
async def sample(n: int = 1):
    """Random REAL transactions (from the committed sample) shaped for /score.

    Drives the dashboard's live stream with varied real amounts and a realistic
    fraud/legit mix, instead of a few hardcoded vectors."""
    from src.stream.simulator import sample_payloads
    return {"transactions": sample_payloads(n)}


@router.post("/onboard")
async def onboard_dataset(req: OnboardRequest):
    """Run the LLM onboarding agent on a transaction CSV.

    Returns the agent's profile, proposed schema mapping, reasoning, and (when
    ``do_train``) a model card. Requires an LLM unless ``allow_heuristic`` is
    set — a missing LLM yields HTTP 503 with guidance rather than a silent
    fallback.
    """
    import pandas as pd
    from src.agent.onboarding_agent import onboard
    from src.llm import LLMUnavailableError

    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {req.csv_path}")
    try:
        df = pd.read_csv(req.csv_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")
    try:
        return onboard(df, allow_heuristic=req.allow_heuristic, do_train=req.do_train)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
