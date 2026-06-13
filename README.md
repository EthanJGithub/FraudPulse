# FraudPulse
Real-time credit-card fraud detection on the real ULB dataset (0.17% fraud). Supervised XGBoost paired with an unsupervised IsolationForest, selected on PR-AUC (0.88) for extreme imbalance. FastAPI scoring service with ALLOW/REVIEW/FLAG decisions, a stream simulator, and a live React + Nivo ops dashboard (KPIs, score histogram, alert feed).
