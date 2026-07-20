# Runbook: Concept Drift Detected

**Alert**: `CONCEPT_DRIFT`  
**Severity**: WARNING → CRITICAL (if sustained)  
**Auto-Resolve**: No

---

## Alert Meaning

The distribution of fraud labels has shifted significantly:
- **Label rate change > 20%** vs baseline (rolling 7d vs 30d)
- **Prediction rate change** (model predicting more/less fraud)
- **Chargeback reason codes** distribution shifted

This indicates the underlying fraud patterns have changed - the model may become less effective.

---

## Quick Diagnosis (5 min)

### 1. Check Drift Details
```bash
# Concept drift details
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb?type=concept" | jq .

# Expected output:
# {
#   "tenant_id": "bank_ng_gtb",
#   "label_rate_baseline": 0.015,
#   "label_rate_current": 0.022,
#   "rate_change": 0.467,
#   "threshold": 0.2,
#   "drift_detected": true,
#   "window_days": 7
# }
```

### 2. Check Feature Drift Too
```bash
# Check if feature drift accompanies concept drift
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb" | jq '.metrics[] | select(.drift_detected) | {feature: .feature, psi: .psi, kl: .kl_divergence}'
```

### 3. Check Recent Labels
```bash
# Label distribution
curl -s "http://localhost:8000/v1/labels/distribution/bank_ng_gtb?days=7" | jq .

# Chargeback reason codes
curl -s "http://localhost:8000/v1/labels/reason_codes/bank_ng_gtb?days=7" | jq 'group_by(.reason_code) | map({code: .[0].reason_code, count: length})'
```

---

## Root Cause Analysis

| Cause | Indicators | Action |
|-------|------------|--------|
| **New fraud pattern** | Specific feature drift + label shift | Accelerate retrain, add rules |
| **Seasonal change** | Periodic, predictable | Adjust baseline, add seasonal features |
| **Label pipeline issue** | Sudden label spike, no feature drift | Fix label pipeline |
| **Upstream data change** | Feature drift + label shift | Investigate upstream |
| **Adversarial attack** | New feature patterns + label spike | Block patterns, accelerate retrain |
| **Label pipeline bug** | Labels change, features stable | Fix label pipeline |

---

## Investigation Steps

### 1. Analyze Label Shift
```bash
# Label rate over time
curl -s "http://localhost:8000/v1/analytics/label_rate/bank_ng_gtb?days=30" | jq '.daily_rates[] | {date: .date, rate: .rate, count: .count}'

# Chargeback reason codes
curl -s "http://localhost:8000/v1/labels/reason_codes/bank_ng_gtb?days=7" | jq 'group_by(.reason_code) | map({code: .[0].reason_code, count: length}) | sort_by(.count) | reverse'
```

### 2. Check Prediction Distribution Shift
```bash
# Model prediction distribution
curl -s "http://localhost:8000/v1/analytics/prediction_dist/bank_ng_gtb?days=7" | jq '.buckets[] | {bucket: .range, count: .count, fraud_rate: .fraud_rate}'
```

### 3. Check Feature Drift Correlation
```bash
# Features with highest drift
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb" | jq '.metrics[] | select(.drift_detected) | {feature: .feature, psi: .psi, kl: .kl_divergence} | select(.psi > 0.2)' | sort_by(.psi) | reverse
```

### 4. Check New Fraud Patterns
```bash
# New fraud clusters
curl -s "http://localhost:8000/v1/analytics/fraud_clusters/bank_ng_gtb?days=7" | jq '.clusters[] | {cluster: .id, size: .size, fraud_rate: .fraud_rate, key_features: .top_features}'
```

---

## Resolution Actions

### 1. If New Fraud Pattern (Feature + Label Drift)
```bash
# 1. Accelerate retrain
docker compose run --rm dashboard python -m training.pipeline \
  --tenant bank_ng_gtb --phase 3 --immediate --reason "Concept drift detected"

# 2. Add temporary rules for new pattern
curl -X POST http://localhost:8000/v1/admin/rules \
  -d '{"id":"NEW_PATTERN_001","type":"expression","expression":"new_fraud_feature > 0.8","action":"hard_block","severity":"high"}'

# 3. Monitor
watch -n 60 'curl -s "http://localhost:8000/v1/lifecycle/bank_ng_gtb" | jq "{pr_auc: .pr_auc, recall: .recall, fraud_capture: .fraud_capture_rate}"'
```

### 2. If Seasonal Pattern
```bash
# Add seasonal features to model config
curl -X PUT http://localhost:8000/v1/admin/model-config/bank_ng_gtb \
  -d '{"features": {"add_seasonal": true, "seasonal_features": ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_holiday"]}}'

# Retrain with seasonal features
docker compose run --rm dashboard python -m training.pipeline --tenant bank_ng_gtb --phase 3
```

### 3. If Label Pipeline Issue
```bash
# Check label ingestion
docker compose logs label_ingestion | grep -i error | tail -20

# Check chargeback system
curl -s "http://localhost:8000/v1/admin/label-pipeline/status/bank_ng_gtb" | jq .

# Fix pipeline, then reprocess
docker compose run --rm label_ingestion python -m label_ingestion.reprocess --days 7
```

### 4. If Adversarial Attack
```bash
# Block attack patterns
curl -X POST http://localhost:8000/v1/admin/blocklist \
  -d '{"list":"accounts","values":["acct_attacker_1","acct_attacker_2"],"reason":"Coordinated attack"}'

curl -X POST http://localhost:8000/v1/admin/blocklist \
  -d '{"list":"devices","values":["dev_attacker_1"],"reason":"Coordinated attack"}'

# Accelerate retrain with attack data
docker compose run --rm dashboard python -m training.pipeline \
  --tenant bank_ng_gtb --phase 3 --include-attack-data
```

---

## Validation After Fix

### 1. Monitor Label Rate
```bash
watch -n 300 'curl -s "http://localhost:8000/v1/analytics/label_rate/bank_ng_gtb?window=1h" | jq .'
```

### 2. Monitor Model Performance
```bash
watch -n 300 'curl -s "http://localhost:8000/v1/lifecycle/bank_ng_gtb" | jq "{pr_auc: .pr_auc, recall: .recall, fraud_capture: .fraud_capture_rate, precision: .precision}"'
```

### 3. Monitor Drift Metrics
```bash
watch -n 600 'curl -s "http://localhost:8000/v1/drift/bank_ng_gtb" | jq ".metrics[] | select(.drift_detected) | {feature: .feature, psi: .psi}"'
```

---

## Post-Resolution

1. **Update baseline** after legitimate shift
```bash
curl -X POST http://localhost:8000/v1/admin/drift/baseline/update \
  -d '{"tenant_id": "bank_ng_gtb", "window_days": 30}'
```

2. **Document root cause** in incident report
3. **Update retrain triggers** if new pattern confirmed
4. **Add features/rules** for new fraud pattern

---

## Prevention

| Measure | Implementation |
|---------|----------------|
| **Adaptive baselines** | Rolling 30-day baseline with seasonal adjustment |
| **Automated retrain** | Trigger on concept drift + performance drop |
| **Label quality monitoring** | Alert on label rate anomalies |
| **Feature importance tracking** | Weekly feature importance comparison |

---

## Related Runbooks

- `drift-spike.md` - Feature drift
- `performance-drop.md` - Performance degradation
- `data-quality.md` - Label pipeline issues
- `chaos-test-model-missing.md` - Chaos testing

---

## Contacts

| Role | Contact |
|------|---------|
| On-call | [Name] - Slack/Phone |
| ML Engineer | [Name] - Model issues |
| Data Engineer | [Name] - Label pipeline |
| Infra | [Name] - Infrastructure |

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*