# Runbook: Model Reload Failure / Manual Reload

**Alert**: `MODEL_RELOAD`  
**Severity**: INFO (success) / WARNING (failure)  
**Auto-Resolve**: Yes (success) / No (failure)

---

## Alert Meaning

Model artifacts reloaded from disk. Triggered by:
- **Watchdog** detecting file changes
- **Manual reload** via `/v1/admin/models/reload`
- **Scheduled reload** (configurable interval)

---

## Normal Reload Flow

```
1. File change detected (watchdog) OR manual trigger
   ↓
2. Load models into _staging (double-buffered)
   ↓
3. Validate models (warmup inference)
   ↓
4. Atomic swap: _active ← _staging
   ↓
5. Emit model_reloaded event
```

---

## Reload Failure Alert

### Symptoms
- Alert: `MODEL_RELOAD` with `success=false`
- Metrics: `model_reload_duration_ms` spikes
- Logs: `Model reload failed` in API logs

### Common Causes

| Cause | Symptoms | Fix |
|-------|----------|-----|
| **Missing artifact** | `FileNotFoundError` | Re-train or restore |
| **Corrupt pickle** | `pickle.UnpicklingError` | Re-train |
| **Schema mismatch** | `AttributeError` on load | Regenerate with current code |
| **OOM during load** | `MemoryError` | Increase memory, smaller model |
| **Version conflict** | Version hash mismatch | Force reload with `--force` |

---

## Resolution

### 1. Check Error Details
```bash
# Check logs
docker compose logs api | grep -i "model.*reload\|model.*load" | tail -20

# Check specific error
docker compose logs api | grep -A 10 "Model reload failed"
```

### 2. Verify Artifacts Exist
```bash
# Check model directory
ls -la artifacts/models/bank_ng_gtb/

# Should have:
# - phase1/ (cold start)
# - phase2/ (adaptive learning)
# - phase3/ (supervised)
# - simple_model.pkl (serving model)
# - version.txt
```

### 3. Manual Reload
```bash
# Force reload
curl -X POST http://localhost:8000/v1/admin/models/reload \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "bank_ng_gtb", "force": true}'

# Check status
curl -s http://localhost:8000/v1/phase/bank_ng_gtb | jq .
```

### 3. Retrain if Missing
```bash
# Full retrain pipeline
docker compose run --rm dashboard python scripts/train_simple_model.py \
  --tenant bank_ng_gtb --all-tenants

# Or specific phase
docker compose run --rm dashboard python -m training.pipeline \
  --tenant bank_ng_gtb --phase 3
```

---

## Manual Reload Procedure

### 1. Prepare New Model
```bash
# Train new model
docker compose run --rm dashboard python scripts/train_simple_model.py \
  --tenant bank_ng_gtb --epochs 500

# Verify artifact
ls -la artifacts/models/bank_ng_gtb/simple_model.pkl
```

### 2. Atomic Reload (Zero Downtime)
```bash
# Trigger reload (atomic swap)
curl -X POST http://localhost:8000/v1/admin/models/reload \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Verify
curl -s http://localhost:8000/v1/phase/bank_ng_gtb | jq .
```

### 3. Verify New Model
```bash
# Test scoring
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"bank_ng_gtb","account_id":"test","amount":1000,"currency":"NGN","timestamp":"2026-07-15T12:00:00Z","transaction_type":"PAYMENT","channel":"MOBILE"}' | jq .
```

### 4. Check Version
```bash
curl -s http://localhost:8000/v1/phase/bank_ng_gtb | jq '{phase: .current_phase, version: .model_version, features: .loaded_models.simple_model.feature_count}'
```

---

## Automated Reload (Watchdog)

### Configuration
```python
# In config/settings.py
model_reload_interval_seconds: 300  # 5 min poll
model_dir: "artifacts/models"
```

### How It Works
```python
# scoring/orchestrator.py
class ModelRegistry:
    def __init__(self):
        self._observer = Observer()
        self._observer.schedule(
            ModelFileHandler(self), 
            settings.model_dir, 
            recursive=True
        )
        self._observer.start()
    
    def load_from_disk(self, model_dir):
        # Double-buffered load
        staging = {}
        # ... load all models ...
        self._swap_active(staging)  # Atomic
```

---

## Troubleshooting

### Reload Takes Too Long (>30s)
```bash
# Check model sizes
du -sh artifacts/models/*/

# Check warmup time
grep "Model warmup" /var/log/fraudtrap/*.log | tail -5
```
**Fix**: Reduce model size, optimize warmup

### Version Hash Mismatch
```bash
# Check version files
cat artifacts/models/bank_ng_gtb/version.txt
cat artifacts/models/bank_ng_gtb/simple_model.pkl | head -c 100

# Force reload ignoring version
curl -X POST http://localhost:8000/v1/admin/models/reload \
  -d '{"force": true, "ignore_version": true}'
```

### Corrupt Model File
```bash
# Verify pickle integrity
python -c "
import pickle
with open('artifacts/models/bank_ng_gtb/simple_model.pkl', 'rb') as f:
    m = pickle.load(f)
    print('OK:', m.feature_names[:5])
"
```

---

## Monitoring

### Key Metrics
| Metric | Normal | Alert |
|--------|--------|-------|
| `model_reload_duration_ms` | < 5000 | > 30000 |
| `model_reload_total` | Increasing | Stuck |
| `model_reload_failures_total` | 0 | > 0 |
| `model_version` | Stable | Unexpected change |

### Alert Rules
```yaml
# Prometheus alerts
- alert: ModelReloadFailed
  expr: increase(model_reload_failures_total[5m]) > 0
  severity: critical

- alert: ModelReloadSlow
  expr: model_reload_duration_ms > 30000
  severity: warning

- alert: ModelReloadStuck
  expr: model_reload_duration_ms > 120000
  severity: critical
```

---

## Post-Reload Verification

```bash
# 1. Check phase
curl -s http://localhost:8000/v1/phase/bank_ng_gtb | jq .

# 2. Test scoring
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"bank_ng_gtb","account_id":"test","amount":1000,"currency":"NGN","timestamp":"2026-07-15T12:00:00Z","transaction_type":"PAYMENT","channel":"MOBILE"}' | jq .

# 3. Check metrics
curl -s http://localhost:8000/metrics | grep fraudtrap_model_version
```

---

## Related Runbooks

- `model-reload.md` - This file
- `sla-breach.md` - If reload causes latency spike
- `chaos-model-missing.md` - Chaos test for missing models

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*