import os
import uuid

from fastapi import APIRouter, HTTPException

from src import store
from src.api.schemas import (
    TransactionRequest, ScoreResponse, HealthResponse, OnboardRequest, OnboardUploadRequest,
)
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


@router.get("/monitoring/transactions")
async def monitoring_transactions(limit: int = 25, decision: str = None):
    """Recent scored transactions for the live feed, optionally filtered by
    decision (ALLOW / REVIEW / FLAG)."""
    return {"transactions": store.recent_transactions(limit=limit, decision=decision)}


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


# Cap browser uploads: protect the free-tier box (512 MB) and keep profiling fast.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB of CSV text
_PROFILE_SAMPLE_ROWS = 3000


@router.post("/onboard/upload")
async def onboard_upload(req: OnboardUploadRequest):
    """ANALYZE-ONLY browser onboarding: the visitor pastes CSV text and the LLM
    agent profiles it, reasons about column roles, self-corrects, and returns the
    proposed schema + reasoning. It NEVER retrains (one shared demo model), so
    ``do_train`` is forced False. Requires an LLM unless ``allow_heuristic`` —
    a missing LLM yields HTTP 503 rather than a silent fallback.
    """
    import io

    import pandas as pd
    from src.agent.onboarding_agent import onboard
    from src.llm import LLMUnavailableError

    text = req.csv_text or ""
    if len(text.encode("utf-8")) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV too large (limit {_MAX_UPLOAD_BYTES // (1024*1024)} MB). "
                   "Upload a smaller sample for the demo.",
        )
    if not text.strip():
        raise HTTPException(status_code=400, detail="csv_text is empty.")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    if df.empty or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV needs a header and at least 2 columns.")
    # Sample for fast profiling; analyse-only so we never touch the shared model.
    if len(df) > _PROFILE_SAMPLE_ROWS:
        df = df.sample(_PROFILE_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
    try:
        return onboard(df, allow_heuristic=req.allow_heuristic, do_train=False)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
