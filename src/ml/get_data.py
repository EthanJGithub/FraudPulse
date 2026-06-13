"""Acquire the real Credit Card Fraud Detection dataset.

Uses the ULB (Université Libre de Bruxelles) dataset of European cardholder
transactions from September 2013 — 284,807 transactions, 492 frauds (0.172%),
the standard real-world benchmark for highly-imbalanced fraud classification.
Features V1–V28 are PCA-anonymized; ``Time``, ``Amount`` and ``Class`` are raw.

Downloaded from a public HuggingFace mirror (no Kaggle account required) to
``data/creditcard.csv``. A real Kaggle download dropped at the same path is a
drop-in replacement.

    python -m src.ml.get_data
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/creditcard.csv"
HF_URL = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"
MIN_ROWS = 200_000


def _looks_real(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) < 50_000_000:
        return False
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    return "Class" in header and "Amount" in header


def ensure_dataset() -> str:
    if _looks_real(DATA_PATH):
        logger.info("Using existing fraud dataset at %s", DATA_PATH)
        return DATA_PATH

    import requests

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    logger.info("Downloading real credit-card fraud dataset (~150 MB) from HuggingFace...")
    with requests.get(HF_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    size_mb = os.path.getsize(DATA_PATH) / 1e6
    logger.info("Saved %s (%.1f MB).", DATA_PATH, size_mb)
    if not _looks_real(DATA_PATH):
        raise RuntimeError("Downloaded file does not look like the fraud dataset.")
    return DATA_PATH


if __name__ == "__main__":
    path = ensure_dataset()
    import pandas as pd
    head = pd.read_csv(path, nrows=3)
    print(f"\nDataset ready: {path}\nColumns: {list(head.columns)[:6]}... ({head.shape[1]} cols)")
