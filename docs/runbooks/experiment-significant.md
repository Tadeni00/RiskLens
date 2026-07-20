# Runbook: Experiment Statistically Significant

**Alert**: `EXPERIMENT_SIGNIFICANT`  
**Severity**: INFO  
**Auto-Resolve**: No (requires human decision)

---

## Alert Meaning

An A/B experiment has reached statistical significance:
- **Challenger significantly better** than champion (p < 0.05, lift > MDE)
- **Challenger significantly worse** than champion (p < 0.05, negative lift)
- **Experiment duration** exceeded maximum

This is a **decision point**, not an error condition.

---

## Alert Details

**Payload:**
```json
{
  "experiment_name": "supervised_v2_challenger",
  "tenant_id": "bank_ng_gtb",
  "champion": "supervised_ensemble_v1",
  "challenger": "supervised_ensemble_v2",
  "metric": "pr_auc",
  "champion_value": 0.78,
  "challenger_value": 0.82,
  "lift_pct": 5.1,
  "p_value": 0.012,
  "significant": true,
  "direction": "challenger_better",
  "samples_champion": 45000,
  "samples_challenger": 5200,
  "duration_days": 14
}
```

---

## Decision Framework

### If Challenger Significantly Better (p < 0.05, lift > MDE)

| Lift Range | Action |
|------------|--------|
| **MDE to 2%** | Extend experiment, gather more data |
| **2% - 5%** | Plan promotion, check guardrails |
| **5% - 10%** | Promote to champion |
| **> 10%** | **Immediate promotion** |

### If Challenger Significantly Worse
- **Stop experiment immediately**
- Rollback to champion
- Investigate root cause

### If No Significance (Duration Exceeded)
- Stop experiment
- Keep champion
- Document learnings

---

## Decision Process

### 1. Verify Statistical Validity
```bash
# Check experiment stats
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/stats" | jq .

# Verify:
# - Minimum sample size met (configurable, default 1000 per variant)
# - Experiment ran minimum duration (default 7 days)
# - No guardrail violations
# - p-value < significance_level (default 0.05)
```

### 2. Check Guardrails
```bash
# Verify no guardrail violations
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/guardrails" | jq '.[] | select(.violated) | {name: .name, value: .value, threshold: .threshold}'
```

### 3. Analyze Lift & Business Impact
```bash
# Get detailed comparison
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/comparison" | jq '.champion, .challenger, .lift_pct, .confidence_interval'

# Calculate business impact
# lift_pct * baseline_fraud_rate * avg_transaction_value * daily_volume
```

---

## Decision Matrix

| Scenario | Action |
|----------|--------|
| **Challenger better, all guardrails pass, lift > MDE** | **PROMOTE** |
| Challenger better, but guardrail violated | **STOP**, investigate, fix, restart |
| Challenger better, but lift < MDE | **EXTEND** experiment |
| Challenger worse, significant | **STOP**, rollback to champion |
| No significance, duration expired | **STOP**, keep champion |
| Challenger worse, not significant | **EXTEND** or **STOP** |

---

## Promotion Procedure

### 1. Pre-Promotion Checklist
```bash
# 1. Verify all guardrails pass
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/guardrails" | jq '.[] | select(.violated) | .name'
# Should return empty

# 2. Verify sample sizes
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/stats" | jq '.variants[] | {name: .name, samples: .samples}'

# 3. Verify duration
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger" | jq '{name, start_date, end_date, duration_days: (.end_date // now) - .start_date}'

# 4. Check business metrics
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/business_impact" | jq .
```

### 2. Promote Challenger
```bash
# Promote challenger to champion
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/promote

# Verify promotion
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger" | jq '.status, .champion_model'
# Should show: "COMPLETED", new champion model
```

### 3. Verify Promotion
```bash
# Check model registry
curl -s "http://localhost:8000/v1/admin/models/registry" | jq '.models[] | select(.tenant=="bank_ng_gtb") | {model_version: .version, phase: .phase, is_champion: .is_champion}'

# Verify traffic routing
curl -s "http://localhost:8000/v1/phase/bank_ng_gtb" | jq '{phase: .current_phase, version: .model_version}'

# Test scoring
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"bank_ng_gtb","account_id":"test","amount":1000,"currency":"NGN","timestamp":"2026-07-15T12:00:00Z","transaction_type":"PAYMENT","channel":"MOBILE"}' | jq .
```

---

## If Challenger Worse (Stop Experiment)

```bash
# 1. Stop experiment immediately
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/stop

# 2. Verify traffic back to champion
curl -s "http://localhost:8000/v1/phase/bank_ng_gtb" | jq .

# 3. Analyze why
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/postmortem" | jq .
```

### Investigation Questions
- Was model artifact corrupted?
- Different feature set?
- Calibration issue?
- Insufficient training data?
- Concept drift during experiment?

---

## Post-Promotion Monitoring

### 1. Monitor 30 min
```bash
watch -n 30 'curl -s "http://localhost:8000/v1/lifecycle/bank_ng_gtb" | jq "{pr_auc: .pr_auc, recall: .recall, fraud_capture: .fraud_capture_rate, latency: .p95_latency_ms}"'
```

### 2. Monitor 24h
```bash
# Next day check
curl -s "http://localhost:8000/v1/metrics/pr_auc/bank_ng_gtb?days=1" | jq '.data[-1].metric_value'
```

### 3. Compare Champion Metrics
```bash
# 7-day comparison
curl -s "http://localhost:8000/v1/metrics/compare/bank_ng_gtb?metric=pr_auc&days=7&baseline_days=7" | jq .
```

---

## Rollback Procedure (If Issues Post-Promotion)

```bash
# 1. Rollback to previous champion
curl -X POST http://localhost:8000/v1/admin/models/rollback \
  -d '{"tenant_id": "bank_ng_gtb", "target_version": "supervised_ensemble_v1"}'

# 2. Verify rollback
curl -s "http://localhost:8000/v1/phase/bank_ng_gtb" | jq .

# 3. Monitor
watch -n 30 'curl -s "http://localhost:8000/v1/lifecycle/bank_ng_gtb" | jq .pr_auc'
```

---

## Documentation

### Update Experiment Record
```bash
# Add decision to experiment record
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/decision \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "PROMOTE",
    "reason": "Challenger v2.1 showed 5.1% PR-AUC lift (p=0.012), all guardrails passed",
    "decided_by": "ml-engineer@example.com",
    "promoted_version": "supervised_ensemble_v2",
    "previous_champion": "supervised_ensemble_v1"
  }'
```

### Update Model Registry
```bash
# Tag promoted model
curl -X POST http://localhost:8000/v1/admin/models/supervised_ensemble_v2/tags \
  -d '{"tags": ["champion", "promoted-2026-07-15"]}'
```

---

## Related Runbooks

- `model-reload.md` - For model reload after promotion
- `sla-breach.md` - If promotion causes latency issues
- `drift-spike.md` - If drift detected post-promotion

---

## Contacts

| Role | Contact |
|------|---------|
| ML Engineer | [Name] - Slack/Phone |
| Product Owner | [Name] - Business approval |
| Engineering Lead | [Name] - Technical approval |

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*