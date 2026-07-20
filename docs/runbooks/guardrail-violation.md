# Runbook: Guardrail Violation (Experiment)

**Alert**: `GUARDRAIL_VIOLATION`  
**Severity**: CRITICAL  
**Auto-Resolve**: No

---

## Alert Meaning

An A/B experiment violated a safety guardrail:
- **Error rate** > 1%
- **Latency P95** > 200ms
- **Custom guardrail** exceeded (business metric)

The experiment must be paused immediately.

---

## Quick Diagnosis (2 min)

### 1. Identify Violated Guardrail
```bash
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/guardrails" | jq .

# Example output:
# {
#   "experiment": "supervised_v2_challenger",
#   "violations": [
#     {"metric": "error_rate", "value": 0.015, "threshold": 0.01, "variant": "challenger"}
#   ]
# }
```

### 2. Check Experiment Status
```bash
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger" | jq .
```

---

## Immediate Action (REQUIRED)

### STOP THE EXPERIMENT IMMEDIATELY
```bash
# Pause experiment - routes all traffic back to champion
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/pause

# Verify traffic routing
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger" | jq '.status, .variants[] | {name: .name, traffic_pct: .traffic_pct}'
```

---

## Investigation (After Pause)

### 1. Analyze Violation Details
```bash
# Get full violation details
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/violations" | jq .

# Check which variant violated
# Champion vs Challenger
```

### 2. Check Impact
```bash
# How many requests affected?
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/impact" | jq .

# Error rate by variant
curl -s "http://localhost:8000/v1/metrics/error_rate/supervised_v2_challenger?hours=1" | jq .

# Latency by variant
curl -s "http://localhost:8000/v1/metrics/latency/supervised_v2_challenger?hours=1" | jq .
```

### 3. Root Cause Analysis

| Violation | Likely Cause | Check |
|-----------|--------------|-------|
| **Error rate** | Model inference error, feature bug | Check model logs, feature engineering |
| **Latency P95** | Slow model, Redis/Kafka slow | Check model latency, dependencies |
| **Custom metric** | Business logic issue | Check metric definition |

---

## Root Cause Resolution

### 1. Challenger Model Errors
```bash
# Check challenger model logs
docker compose logs api | grep "supervised_v2_challenger" | tail -20

# Check model loading
curl -s "http://localhost:8000/v1/admin/models/supervised_v2_challenger" | jq .
```

**Fix**: Rollback challenger, investigate model artifact

### 2. Feature Engineering Bug
```bash
# Check feature engineering
docker compose logs feature_engineering | grep -i error | tail -20

# Test feature extraction
curl -X POST http://localhost:8000/v1/admin/features/test \
  -d '{"tenant_id": "bank_ng_gtb", "features": {...}}'
```

### 3. Model Inference Timeout
```bash
# Check model inference time
curl -s "http://localhost:8000/v1/metrics/model_latency?model=supervised_v2_challenger&hours=1" | jq .

# If > 200ms: optimize or rollback
```

### 4. Guardrail Misconfiguration
```bash
# Check guardrail config
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/config" | jq '.guardrails'

# Fix threshold if too aggressive
curl -X PUT http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/guardrails \
  -d '{"error_rate": {"threshold": 0.02}}'  # Was 0.01
```

---

## Resolution Decision Matrix

| Violation Type | Severity | Action |
|----------------|----------|--------|
| **Error rate > 1%** | CRITICAL | Stop experiment, fix model/code |
| **Latency P95 > 200ms** | CRITICAL | Stop, optimize or rollback |
| **Latency P95 100-200ms** | WARNING | Investigate, may continue with monitoring |
| **Custom metric** | Per definition | Per metric definition |

### Standard Resolution: Stop & Fix
```bash
# 1. Stop experiment
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/stop

# 2. Fix root cause (model, code, config)

# 3. Restart experiment (after fix)
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/start
```

### Alternative: Adjust Guardrail (If False Positive)
```bash
# Only if confirmed false positive
curl -X PUT http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/guardrails \
  -d '{"error_rate": {"threshold": 0.02}}'  # Relax threshold

# Resume experiment
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/resume
```

---

## Post-Resolution

### 1. Verify Fix
```bash
# Re-enable experiment
curl -X POST http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/resume

# Monitor 30 min
watch -n 60 'curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/guardrails" | jq .'
```

### 2. Verify Metrics Normal
```bash
# 30 min after resume
curl -s "http://localhost:8000/v1/admin/experiments/supervised_v2_challenger/guardrails" | jq '.violations | length'
# Should be 0
```

### 3. Post-Mortem
```bash
# Document
cat > incident_report.md << EOF
## Guardrail Violation: supervised_v2_challenger

**Date**: $(date)
**Violation**: error_rate=1.5% (threshold 1%)
**Duration**: 8 minutes
**Impact**: 10% of traffic to challenger

**Root Cause**: Challenger model v2.1 had division-by-zero bug in feature X

**Resolution**: Rolled back to v2.0, fixed feature, redeployed v2.2

**Action Items**:
- [ ] Add division-by-zero guard in feature X
- [ ] Add model validation test for feature X
- [ ] Add pre-deployment smoke test
EOF
```

---

## Prevention

| Measure | Implementation |
|---------|----------------|
| **Pre-deployment tests** | Smoke tests, integration tests for models |
| **Canary deployment** | 1% → 10% → 50% → 100% with auto-rollback |
| **Guardrail tuning** | Quarterly review of thresholds |
| **Model validation** | Schema validation, shape checks on load |
| **Feature validation** | Schema validation, range checks |

---

## Related Runbooks

- `model-reload.md` - Model reload procedures
- `chaos-test-model-missing.md` - Chaos testing
- `performance-drop.md` - If performance degrades

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*