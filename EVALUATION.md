**Evaluation protocol — September 2026**

Training/validation/final-test are stratified 60/20/20 splits (seeds 42, 43). Only validation selects early stopping and thresholds. This is a within-dataset estimate, not prospective customer validation; the underlying public dataset has been explored previously. No claim of a never-before-seen benchmark dataset is made.

Uncalibrated class-weighted model scores must not be interpreted as validated real-world probabilities. Temporal/customer-disjoint evaluation and independent deployment validation remain future work.

```json
{
  "model_version": "fraud-v2.0",
  "dataset": "ULB Credit-Card Fraud",
  "n_features": 31,
  "n_train": 170883,
  "n_validation": 56962,
  "n_test": 56962,
  "evaluation_protocol": "stratified 60/20/20; early stopping and thresholds on validation only; final test untouched during selection",
  "split_seeds": [
    42,
    43
  ],
  "validation_pr_auc": 0.8814598224811538,
  "test_review_rate": 0.0019486675327411256,
  "fraud_rate": 0.001727485630620034,
  "scale_pos_weight": 578.26,
  "pr_auc": 0.8755,
  "roc_auc": 0.9829,
  "flag_threshold": 0.1365,
  "review_threshold": 0.05,
  "precision_at_flag": 0.8235,
  "recall_at_flag": 0.8571,
  "f1_at_flag": 0.84,
  "confusion_at_flag": {
    "tp": 84,
    "fp": 18,
    "fn": 14,
    "tn": 56846
  },
  "anomaly_percentiles": {
    "50": 0.3976099834318804,
    "90": 0.45428993901784814,
    "95": 0.4879669563518574,
    "99": 0.5844255499635442
  },
  "data_source": "huggingface:David-Egea/Creditcard-fraud-detection",
  "trained_on_real_data": true
}
```
