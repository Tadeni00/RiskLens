# Runbook: Drift Spike

**Alert**: `DRIFT_SPIKE`  
**Severity**: WARNING  
**Auto-Resolve**: Yes (after 2 hours if resolved)

---

## Alert Meaning

A feature's Population Stability Index (PSI) > 0.25 or KL Divergence > 0.1
- Feature distribution has shifted significantly
- May indicate data quality issues, new user behavior, or upstream changes

---

## Quick Diagnosis (3 min)

### 1. Identify Drifting Feature
```bash
# Which feature triggered
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb" | jq '.metrics[] | select(.psi > 0.25 or .kl > 0.1) | {feature, psi, kl, mean_shift}'
```

### 2. Check Feature Distribution
```bash
# Compare baseline vs current
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb?feature=acct_v_1h_count&hours=24" | jq .
```

### 3. Check Upstream Source
```bash
# Is the feature computed correctly?
docker compose logs -f feature_engineering | grep "acct_v_1h_count" | tail -20
```

---

## Common Causes

| Feature Type | Common Causes | Investigation |
|--------------|---------------|---------------|
| **Velocity** (`acct_v_*`) | Redis outage, clock skew, new channel | Check Redis, new channels |
| **Amount** (`amount`, `amount_zscore`) | Currency change, new merchant, promo | Check merchants, currency rates |
| **Device** (`is_new_device`, `device_account_count`) | New device SDK, app update | Check app version rollout |
| **Geo** (`geo_speed_kmh`, `impossible_travel`) | VPN, proxy, travel | Check VPN detection |
| **Behavioral** (`typing_zscore`) | SDK change, new OS | Check SDK version |
| **All features** | Redis outage, schema change | Check feature engineering logs |

---

## Investigation Steps

### 1. Check Feature Time Series
```bash
# 24h trend
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb?feature=acct_v_1h_count&hours=24" | jq '.series'
```

### 2. Compare Distributions
```bash
# ClickHouse query
SELECT 
    toStartOfHour(transaction_timestamp) as hour,
    quantile(0.5)(acct_v_1h_count) as p50,
    quantile(0.95)(acct_v_1h_count) as p95,
    avg(acct_v_1h_count) as mean,
    count() as cnt
FROM transactions
WHERE tenant_id = 'bank_ng_gtb' AND transaction_timestamp >= now() - INTERVAL 24 HOUR
GROUP BY hour ORDER BY hour;
```

### 3. Check Feature Engineering
```bash
# Redis key health
redis-cli --scan --pattern "ft:*:*:acct_v_1h_count" | head -20

# Feature engineering logs
docker compose logs feature_engineering | grep -i "acct_v_1h_count" | tail -20
```

---

## Decision Matrix

| Drift Type | PSI | KL | Mean Shift | Likely Cause | Action |
|------------|-----|-----|------------|--------------|--------|
| **Sudden spike** | >0.5 | >0.5 | Large | Redis/schema issue | Check infra, feature eng |
| **Gradual shift** | 0.25-0.5 | 0.1-0.5 | Moderate | Behavior change | Monitor, expand baseline |
| **Seasonal** | 0.1-0.3 | <0.1 | Periodic | Holiday/sale | Expand baseline window |
| **New feature values** | >0.5 | >0.5 | New range | New merchant/channel | Update schema, retrain |

---

## Resolution Actions

### 1. Infrastructure Issue (Redis/Schema)
```bash
# Check Redis
redis-cli INFO stats | grep keyspace

# Check schema
curl -s "http://localhost:8000/v1/admin/schema/bank_ng_gtb" | jq '.feature_names | length'
```

### 2. Upstream Data Change
```bash
# Check ingestion
docker compose logs transaction_ingestion | grep -i "acct_v_1h_count" | tail -20

# Check Kafka
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group fraudtrap-ingestion --describe
```

### 3. Schema Evolution
```bash
# Check feature schema
curl -s "http://localhost:8000/v1/admin/schema/bank_ng_gtb" | jq '.features[] | select(.name=="acct_v_1h_count")'

# Register new schema version
curl -X POST http://localhost:8000/v1/admin/schema \
  -d '{"tenant_id":"bank_ng_gtb","features":[...]}'
```

### 4. Expand Baseline (Seasonal/Gradual)
```bash
# Expand baseline window
docker compose run --rm dashboard python -c "
from monitoring.drift import recompute_baseline
recompute_baseline('bank_ng_gtb', feature='acct_v_1h_count', window_days=60)
"
```

---

## Post-Resolution

1. **Verify** PSI/KL return to < 0.1 within 2h
2. **Update** baseline if legitimate shift
3. **Document** cause in incident tracker
4. **Alert** team if upstream data contract changed

---

## Related Runbooks

- `concept-drift.md` - Label distribution shift
- `data-quality.md` - Upstream data issues
- `redis-outage.md` - Redis infrastructure issues

---

## References

- [PSI Monitoring](https://www.listendata.com/2019/05/population-stability-index-psi.html)
- [KL Divergence Monitoring](https://medium.com/@mlengineer/kl-divergence-for-drift-detection-abc123)
- [Feature Drift Best Practices](https://ml-ops.org/content/drift-detection)

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*