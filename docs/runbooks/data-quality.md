# Runbook: Data Quality Issues

**Alert**: `DATA_QUALITY`  
**Severity**: WARNING → CRITICAL (if > 20% features affected)  
**Auto-Resolve**: No

---

## Alert Meaning

Data quality checks failed on incoming transactions:
- **Missing features**: > 10% features zero-filled (Redis unavailable)
- **Schema violations**: Unexpected types, out-of-range values
- **Staleness**: Features not updated within expected window
- **Schema drift**: New/removed features without schema update
- **Behavioral profile staleness**: Profiles not updated within expected window (24h for active entities)
- **Cold-start hierarchy degradation**: > 50% queries falling to Tenant/Global level (insufficient individual history)

---

## Quick Diagnosis (3 min)

### 1. Identify Issue Type
```bash
# Data quality dashboard
curl -s "http://localhost:8000/v1/admin/data-quality/bank_ng_gtb" | jq '.issues[] | {feature, issue, severity, count}'
```

### 2. Check Recent Transactions
```bash
# Recent scored transactions
curl -s "http://localhost:8000/v1/recent?limit=10&tenant=bank_ng_gtb" | jq '.items[] | .features'
```

### 3. Check Feature Store
```bash
# Redis health
redis-cli ping

# Key health
redis-cli --scan --pattern "ft:bank_ng_gtb:acct:*" | head -5 | xargs -I {} redis-cli GET {}
```

---

## Common Issues

| Issue | Detection | Root Cause |
|-------|-----------|------------|
| **All features zero** | All features = 0 | Redis down, circuit breaker open |
| **Partial zero** | Specific features = 0 | Redis key expiry, naming change |
| **Schema violation** | Type errors in scoring | Upstream schema change |
| **Stale features** | Features not updating | Redis TTL expired, ingestion lag |
| **Schema drift** | New/removed features | Upstream schema change |

---

## Investigation Steps

### 1. Check Feature Store Health
```bash
# Redis connectivity
redis-cli ping

# Key counts
redis-cli --scan --pattern "ft:bank_ng_gtb:*" | wc -l

# Sample keys
redis-cli --scan --pattern "ft:bank_ng_gtb:acct:*" | head -5 | xargs -I {} redis-cli HGETALL {}
```

### 2. Check Ingestion Pipeline
```bash
# Ingestion logs
docker compose logs transaction_ingestion | grep -i "error\|fail\|timeout" | tail -20

# Kafka consumer lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group fraudtrap-ingestion --describe
```

### 3. Feature Engineering Errors
```bash
# Feature engineering errors
docker compose logs feature_engineering | grep -i "error\|fail\|timeout" | tail -20

# Specific feature
docker compose logs feature_engineering | grep "acct_v_1h_count" | tail -10
```

### 4. Schema Registry Check
```bash
# Registered schema
curl -s "http://localhost:8000/v1/admin/schema/bank_ng_gtb" | jq .

# Live features vs registered
curl -s "http://localhost:8000/v1/admin/live-features/bank_ng_gtb" | jq '.features | length'
```

---

## Resolution Actions

### 1. Redis Down
```bash
# Check Redis
redis-cli ping

# Restart
docker compose restart redis

# Verify
redis-cli ping && echo "OK"
```

### 2. Circuit Breaker Open
```bash
# Check status
curl -s "http://localhost:8000/v1/admin/circuit-breakers" | jq .

# Reset if appropriate
curl -X POST http://localhost:8000/v1/admin/circuit-breakers/redis/reset
```

### 3. Schema Migration
```bash
# Register new schema
curl -X POST http://localhost:8000/v1/admin/schema \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "bank_ng_gtb",
    "version": 2,
    "features": ["amount", "amount_log", "amount_zscore", ...],
    "created_by": "migration_script"
  }'
```

### 4. Feature Store Recovery
```bash
# Rebuild velocity features from Kafka
docker compose run --rm dashboard python -c "
from features.engineering import rebuild_velocity_features
rebuild_velocity_features('bank_ng_gtb', hours=24)
"
```

### 5. Schema Migration (Breaking Change)
```bash
# Create migration script
cat > migrate_schema_v2.py << 'EOF'
from features.schema_registry import FeatureSchemaRegistry
registry = FeatureSchemaRegistry()
registry.migrate("bank_ng_gtb", 1, 2, transform_fn=lambda x: x)
EOF
python migrate_schema_v2.py
```

---

## Validation

```bash
# 1. Check data quality endpoint
curl -s "http://localhost:8000/v1/admin/data-quality/bank_ng_gtb" | jq '.issues | length'

# 2. Score a test transaction
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "bank_ng_gtb",
    "account_id": "test_acct",
    "amount": 10000,
    "currency": "NGN",
    "timestamp": "2026-07-15T12:00:00Z",
    "transaction_type": "PAYMENT",
    "channel": "MOBILE",
    "device_id": "dev_test",
    "country_code": "NG"
  }' | jq '{decision: .decision, score: .risk_score, rules: .triggered_rules}'

# 3. Check feature freshness
curl -s "http://localhost:8000/v1/admin/feature-freshness/bank_ng_gtb" | jq .
```

---

## Post-Resolution

1. **Monitor 1h** for recurrence
2. **Update schema registry** if new features added
3. **Update runbook** if new failure mode
4. **Alert data engineering** if upstream change

---

## Prevention

| Measure | Implementation |
|---------|----------------|
| **Schema validation** | Pydantic models on ingestion |
| **Redis health checks** | Every 10s, circuit breaker at 5 failures |
| **Schema evolution** | Backward compatible only, versioned |
| **Feature freshness** | Alert if feature age > 2x expected |
| **Upstream contract** | Schema registry with breaking change detection |

---

## Contacts

| Role | Contact |
|------|---------|
| On-call | [Name] - Slack/Phone |
| Data Engineer | [Name] - Schema changes |
| Infra | [Name] - Redis/Infra issues |

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*