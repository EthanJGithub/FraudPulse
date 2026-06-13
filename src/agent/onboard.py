"""CLI for the Dataset Onboarding Agent.

    python -m src.agent.onboard --data path/to/transactions.csv
    python -m src.agent.onboard --data mydata.csv --no-train         # map only
    python -m src.agent.onboard --data mydata.csv --allow-heuristic  # no-LLM mode

By default an LLM (Groq/Anthropic) is required; without one the agent stops with
guidance rather than silently using heuristics. Pass --allow-heuristic to run the
transparent deterministic engine instead (clearly labelled in the output).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.agent.onboarding_agent import onboard
from src.llm import LLMUnavailableError


def _print_report(r: dict) -> None:
    line = "=" * 64
    print(line)
    print(f"FraudPulse Onboarding Agent  |  mode={r['mode']}  provider={r['provider']}")
    if r.get("notice"):
        print(f"\n[!] {r['notice']}")
    cfg = r.get("config")
    if cfg:
        print(f"\nProposed schema (validated in {r['attempts']} attempt(s)):")
        print(f"  target_col   : {cfg['target_col']}  (positive_label={cfg['positive_label']})")
        print(f"  amount_col   : {cfg['amount_col']}")
        print(f"  time_col     : {cfg['time_col']}")
        print(f"  feature_cols : {len(cfg['feature_cols'])} columns -> {cfg['feature_cols'][:8]}"
              + (" ..." if len(cfg['feature_cols']) > 8 else ""))
    if r.get("reasoning"):
        print(f"\nAgent reasoning: {r['reasoning']}")
    if not r["validation"]["ok"] and r["validation"]["errors"]:
        print("\nValidation errors:")
        for e in r["validation"]["errors"]:
            print(f"  - {e}")
    card = r.get("model_card")
    if card:
        print(f"\nModel card ({card['model_version']}):")
        print(f"  PR-AUC={card['pr_auc']}  ROC-AUC={card['roc_auc']}  "
              f"fraud_rate={card['fraud_rate']*100:.3f}%")
        print(f"  precision@flag={card['precision_at_flag']}  recall@flag={card['recall_at_flag']}")
        print(f"  confusion={card['confusion_at_flag']}")
    if r.get("explanation"):
        print(f"\nExplanation:\n  {r['explanation']}")
    print(f"\nstatus: {r['status']}")
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Onboard a transaction CSV into FraudPulse.")
    ap.add_argument("--data", required=True, help="Path to a labelled transaction CSV.")
    ap.add_argument("--no-train", action="store_true", help="Map the schema but don't train.")
    ap.add_argument("--allow-heuristic", action="store_true",
                    help="Run the deterministic engine if no LLM is configured.")
    ap.add_argument("--model-dir", default="models", help="Where to write the trained model.")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    df = pd.read_csv(args.data)
    try:
        report = onboard(
            df,
            allow_heuristic=args.allow_heuristic,
            do_train=not args.no_train,
            model_dir=args.model_dir,
        )
    except LLMUnavailableError as exc:
        print(f"\n[x] LLM required: {exc}\n", file=sys.stderr)
        return 2

    _print_report(report)
    return 0 if report["status"] in ("trained", "mapped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
