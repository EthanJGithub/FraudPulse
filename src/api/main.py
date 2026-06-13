import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

load_dotenv()
logging.basicConfig(level="INFO")

app = FastAPI(
    title="FraudPulse",
    description="Real-time transaction fraud detection (XGBoost + IsolationForest)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def _seed_on_startup():
    import os, threading
    if os.getenv("FRAUDPULSE_AUTOSEED", "1") == "0":
        return

    def _run():
        try:
            from src.stream.simulator import ensure_seeded
            n = ensure_seeded()
            if n:
                logging.getLogger(__name__).info("Seeded %d transactions on startup.", n)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).warning("Seed skipped: %s", exc)

    threading.Thread(target=_run, daemon=True).start()


@app.get("/")
async def root():
    return {"service": "FraudPulse", "docs": "/docs", "health": "/api/v1/health"}
