# Runbook: Performance Drop

**Alert**: `PERFORMANCE_DROP`  
**Severity**: CRITICAL  
**Auto-Resolve**: No

---

## Alert Meaning

Model performance metrics have degraded:
- **PR-AUC drop > 5%** vs 7-day rolling average
- **Recall @ 0.85 drop > 10%** 
- **Fraud capture rate drop > 15%**
- **F2 score drop > 5%**

---

## Quick Diagnosis (5 min)

### 1. Identify Dropped Metric
```bash
curl -s "http://localhost:8000/v1/lifecycle/bank_ng_gtb" | jq '{pr_auc: .pr_auc, recall: .recall, fraud_capture: .fraud_capture_rate, f2: .f2_score, review_rate: .decisions.REVIEW, fpr: .fpr}'
```

### 2. Check Trend
```bash
# Last 7 days PR-AUC
curl -s "http://localhost:8000/v1/metrics/pr_auc/bank_ng_gtb?days=7" | jq '.data[] | {date: .bucket_date, pr_auc: .metric_value}'
```

### 3. Check for Concurrent Drift
```bash
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb" | jq '.metrics[] | select(.drift_detected==true) | {feature: .feature, psi: .psi, kl: .kl_divergence}'
```

---

## Root Causes

| Cause | Indicators | Action |
|-------|------------|--------|
| **Data drift** | Drift alerts + perf drop | Check `/v1/drift`, retrain |
| **Concept drift** | Label rate shift + perf drop | Check label rates, retrain |
| **Model staleness** | Gradual decline, no drift | Retrain overdue |
| **Label quality** | Perf drop, stable features | Check label pipeline |
| **Adversarial** | Sudden drop, specific segments | Check new patterns |
| **Feature bug** | Sudden drop, specific features | Check feature engineering |
| **Model corruption** | Sudden drop, all metrics | Reload/reload model |

---

## Investigation

### 1. Check Drift Correlation
```bash
# Get drift metrics
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb" | jq '.metrics[] | select(.drift_detected) | {feature, psi, kl, mean_shift}'

# Correlation with perf drop
# If high PSI features = important features → drift cause
```

### 2. Check Label Quality
```bash
# Label rate
curl -s "http://localhost:8000/v1/metrics/label_rate/bank_ng_gtb?days=7" | jq '.data[] | {date: .bucket_date, rate: .rate}'

# Label quality
curl -s "http://localhost:8000/v1/labels/quality/bank_ng_gtb?days=7" | jq .
```

### 3. Check Model Artifacts
```bash
# Model version
curl -s "http://localhost:8000/v1/phase/bank_ng_gtb" | jq '{phase: .current_phase, version: .model_version}'

# Model age
curl -s "http://localhost:8000/v1/admin/models/history/bank_ng_gtb" | jq '.[0] | {version: .version, trained_at: .trained_at, metrics: .metrics}'
```

### 4. Check Recent Retrains
```bash
curl -s "http://localhost:8000/v1/admin/retrain/history/bank_ng_gtb" | jq '.[0:5] | .[] | {phase: .phase, version: .version, date: .completed_at, metrics: .metrics}'
```

---

## Resolution

### If Data/Concept Drift
```bash
# 1. Accelerate retrain
docker compose run --rm dashboard python -m training.pipeline \
  --tenant bank_ng_gtb --phase 3 --immediate --reason "Performance drop + drift"

# 2. Add temp rules for new patterns
curl -X POST http://localhost:8000/v1/admin/rules \
  -d '{"id":"TEMP_DRIFT_001","type":"expression","expression":"drifted_feature > threshold","action":"soft_boost","boost":0.15}'

# 3. Monitor
watch -n 60 'curl -s "http://localhost:8000/v1/lifecycle/bank_ng_gtb" | jq "{pr_auc: .pr_auc, recall: .recall}"'
```

### If Model Staleness
```bash
# Schedule immediate retrain
docker compose run --rm dashboard python -m training.pipeline \
  --tenant bank_ng_gtb --phase 3 --immediate
```

### If Label Quality Issue
```bash
# Check label pipeline
docker compose logs label_ingestion | grep -i error | tail -20

# Check chargeback codes
curl -s "http://localhost:8000/v1/labels/reason_codes/bank_ng_gtb?days=7" | jq 'group_by(.reason_code) | map({code: .[0].reason_code, count: length}) | sort_by(.count) | reverse'

# Fix pipeline, reprocess labels
docker compose run --rm label_ingestion python -m label_ingestion.reprocess --days 7
```

### If Adversarial/Novel Pattern
```bash
# Add emergency rules
curl -X POST http://localhost:8000/v1/admin/rules \
  -d '{"id":"EMERGENCY_ATTACK_001","type":"expression","expression":"new_attack_feature > 0.9","action":"hard_block","severity":"critical"}'

# Add to blocklist
curl -X POST http://localhost:8000/v1/admin/blocklist \
  -d '{"list":"accounts","values":["acct_attacker_1"],"reason":"Attack pattern"}'

# Accelerate retrain with attack data
docker compose run --rm dashboard python -m training.pipeline \
  --tenant bank_ng_gtb --phase 3 --immediate --include-attack-data
```

### If Model Corruption
```bash
# Reload model
curl -X POST http://localhost:8000/v1/admin/models/reload

# Verify
curl -s "http://localhost:8000/v1/phase/bank_ng_gtb" | jq .
```

---

## Validation After Fix

### 1. Immediate (5 min)
```bash
# Score test transaction
curl -X POST http://localhost:8000/v1/score \
  -d '{"tenant_id":"bank_ng_gtb","account_id":"test","amount":10000,"currency":"NGN","timestamp":"2026-07-15T12:00:00Z","transaction_type":"TRANSFER","channel":"API"}' | jq '{decision: .decision, score: .risk_score, rules: .triggered_rules}'
```

### 2. 30 min
```bash
curl -s "http://localhost:8000/v1/lifecycle/bank_ng_gtb" | jq '{pr_auc: .pr_auc, recall: .recall, fraud_capture: .fraud_capture_rate}'
```

### 3. 24h
```bash
curl -s "http://localhost:8000/v1/metrics/pr_auc/bank_ng_gtb?days=1" | jq '.data[-1].metric_value'
```

---

## Post-Resolution

1. **Root cause** in 24h
2. **Update retrain triggers** if new pattern
3. **Add monitoring** for new failure mode
4. **Update runbook** if new failure mode

---

## Related Runbooks

- `drift-spike.md` - Feature drift
- `concept-drift.md` - Label distribution shift
- `model-reload.md` - Model reload failure
- `chaos-test-model-missing.md` - Chaos test

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*