from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class TransactionRequest(BaseModel):
    """A transaction to score. V1–V28 are the PCA features from the dataset;
    Amount and Time are raw. Missing V-features default to 0."""
    txn_id: Optional[str] = Field(None, description="Client transaction id")
    Amount: float = Field(..., ge=0)
    Time: float = Field(0.0, description="Seconds since first transaction")
    features: Dict[str, float] = Field(default_factory=dict, description="V1..V28 PCA features")

    model_config = {"json_schema_extra": {"example": {
        "txn_id": "txn-001", "Amount": 149.62, "Time": 0.0,
        "features": {"V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V14": -0.31}
    }}}


class ScoreResponse(BaseModel):
    txn_id: str
    fraud_probability: float
    anomaly_score: float
    is_anomaly: bool
    decision: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    transactions_scored: int
    onboarding_agent_online: bool = Field(
        False, description="True when an LLM backend is configured so the onboarding agent can run"
    )
    llm_provider: str = Field("offline", description="Resolved LLM provider:model, or 'offline'")


class OnboardRequest(BaseModel):
    """Ask the onboarding agent to analyse a transaction CSV on the server.

    ``do_train`` defaults to False so the endpoint returns the agent's schema
    mapping + reasoning quickly; set it true to also retrain (can be slow on
    large files). ``allow_heuristic`` opts into the deterministic engine when no
    LLM is configured."""
    csv_path: str = Field(..., description="Path to a labelled transaction CSV on the server")
    do_train: bool = Field(False, description="Also retrain the model on the mapped schema")
    allow_heuristic: bool = Field(False, description="Use the deterministic engine if no LLM is set")

    model_config = {"json_schema_extra": {"example": {
        "csv_path": "data/sample_transactions.csv", "do_train": False, "allow_heuristic": False,
    }}}


class OnboardUploadRequest(BaseModel):
    """Browser-side onboarding: the visitor pastes/uploads CSV *text* (not a
    server path). The endpoint is ANALYZE-ONLY — it profiles + reasons + returns
    the proposed schema, but never retrains (one shared demo model)."""
    csv_text: str = Field(..., description="Raw CSV content (header + rows) as text")
    allow_heuristic: bool = Field(False, description="Use the deterministic engine if no LLM is set")

    model_config = {"json_schema_extra": {"example": {
        "csv_text": "amount,time,v1,v2,is_fraud\n149.62,0,-1.36,-0.07,0\n",
        "allow_heuristic": False,
    }}}
