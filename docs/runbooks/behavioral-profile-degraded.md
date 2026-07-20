# Runbook: Behavioral Profile Issues

**Alert**: `BEHAVIORAL_PROFILE_DEGRADED` / `COLD_START_FALLBACK_SPIKE`  
**Severity**: WARNING → CRITICAL (if > 50% cold-start fallback)  
**Auto-Resolve**: No

---

## Alert Meaning

Behavioral Intelligence Layer (Phase 7) profiles showing degradation:

- **Cold-start fallback spike**: > 50% of queries falling to Merchant/Tenant/Global level (insufficient individual history)
- **Profile corruption**: Serialization errors, missing fields, invalid data
- **Stale profiles**: Profiles not updated > 24h for active entities
- **Feature generation failures**: Behavioral features returning zeros/errors
- **Trust score anomalies**: Trust scores stuck at 0.0 or 1.0 unexpectedly

---

## Quick Diagnosis (3 min)

### 1. Check Behavioral Health Endpoint
```bash
# Behavioral profile health
curl -s "http://localhost:8000/v1/admin/behavioral-health/bank_ng_gtb" | jq .

# Expected output:
# {
#   "customer_profiles": {"total": 10000, "active_24h": 8500, "stale": 1500},
#   "merchant_profiles": {"total": 5000, "active_24h": 4200, "stale": 800},
#   "device_profiles": {"total": 12000, "active_24h": 10000, "stale": 2000},
#   "cold_start_fallback_rate": 0.15,  # 15% = OK
#   "profile_update_errors": 0,
#   "feature_generation_errors": 0
# }
```

### 2. Check Cold-Start Fallback Rate
```bash
# If fallback_rate > 0.5 (50%), investigate
curl -s "http://localhost:8000/v1/admin/cold-start-fallback/bank_ng_gtb" | jq .
```

### 3. Check Recent Profile Updates
```bash
# Redis profile keys
redis-cli --scan --pattern "ft:bank_ng_gtb:cust:*" | head -5 | xargs -I {} redis-cli HGETALL {}

# Check TTL
redis-cli --scan --pattern "ft:bank_ng_gtb:cust:*" | head -5 | xargs -I {} redis-cli TTL {}
```

---

## Common Issues

| Issue | Detection | Root Cause |
|-------|-----------|------------|
| **High cold-start fallback** | fallback_rate > 50% | New tenant, insufficient history, profile TTL too short |
| **Profile serialization error** | Errors in logs "Failed to deserialize profile" | Schema change, corrupted data |
| **Trust scores all 0.0/1.0** | trust_score not in (0,1) | Missing trusted_devices, frequency maps empty |
| **Behavioral features zero** | velocity/trust/similarity = 0 | Feature generation error, profile missing |
| **Profile not updating** | last_updated > 24h ago | Update pipeline failing, circuit breaker open |

---

## Investigation Steps

### 1. Check Profile Store Health
```bash
# Redis connectivity
redis-cli ping

# Profile counts by type
redis-cli --scan --pattern "ft:bank_ng_gtb:cust:*" | wc -l
redis-cli --scan --pattern "ft:bank_ng_gtb:merch:*" | wc -l
redis-cli --scan --pattern "ft:bank_ng_gtb:dev:*" | wc -l
redis-cli --scan --pattern "ft:bank_ng_gtb:ben:*" | wc -l
redis-cli --scan --pattern "ft:bank_ng_gtb:inst:*" | wc -l

# Sample profile structure
redis-cli --scan --pattern "ft:bank_ng_gtb:cust:*" | head -3 | xargs -I {} redis-cli HGETALL {}
```

### 2. Check Profile Update Pipeline
```bash
# Behavior engine logs
docker compose logs behavior_engine | grep -i "error\|fail\|update" | tail -20

# Feature generation errors
docker compose logs api | grep -i "behavioral.*error\|generate.*error" | tail -20

# Circuit breaker status
curl -s "http://localhost:8000/v1/admin/circuit-breakers" | jq '.behavioral'
```

### 3. Check Cold-Start Hierarchy
```bash
# Cold-start fallback breakdown
curl -s "http://localhost:8000/v1/admin/cold-start-breakdown/bank_ng_gtb" | jq .

# Expected:
# {
#   "customer_level": 0.65,   # 65% use customer profile
#   "merchant_level": 0.20,   # 20% fallback to merchant
#   "tenant_level": 0.10,     # 10% fallback to tenant
#   "global_level": 0.05      # 5% fallback to global
# }
```

### 4. Check Trust Score Distribution
```bash
# Sample trust scores
curl -s "http://localhost:8000/v1/admin/trust-score-distribution/bank_ng_gtb" | jq .

# Should show distribution, not all 0.0 or 1.0
```

---

## Resolution Actions

### 1. High Cold-Start Fallback (New Tenant)
```bash
# This is EXPECTED for new tenants (< 1000 transactions)
# No action needed - profiles need time to build

# Verify: check transaction volume
curl -s "http://localhost:8000/v1/admin/transaction-volume/bank_ng_gtb?days=7" | jq .
```

### 2. Profile Serialization Errors
```bash
# Check for schema mismatches
curl -s "http://localhost:8000/v1/admin/profile-schema/bank_ng_gtb" | jq .

# If schema changed, rebuild profiles
docker compose run --rm dashboard python -c "
from behavior.storage.redis_store import get_feature_store
from behavior.profiles.customer import CustomerBehaviorProfile
store = get_feature_store()
# Rebuild logic here
"
```

### 3. Trust Scores Stuck at Extremes
```bash
# Check profile has trusted_devices and frequency data
redis-cli --scan --pattern "ft:bank_ng_gtb:cust:*" | head -5 | xargs -I {} redis-cli HGETALL {} | grep -E "trusted_devices|device_fingerprint_frequency"

# If empty, profiles need transactions to build history
# Verify transaction ingestion working
docker compose logs transaction_ingestion | tail -20
```

### 4. Behavioral Features Returning Zeros
```bash
# Check feature generation pipeline
curl -s "http://localhost:8000/v1/admin/behavioral-features/bank_ng_gtb?sample=10" | jq .

# Check behavior engine
curl -s "http://localhost:8000/v1/admin/behavior-engine/status" | jq .

# Restart behavior engine if stuck
docker compose restart api
```

### 5. Profile Update Pipeline Stalled
```bash
# Check circuit breakers
curl -s "http://localhost:8000/v1/admin/circuit-breakers" | jq '.behavioral'

# If open, reset
curl -X POST "http://localhost:8000/v1/admin/circuit-breakers/behavioral/reset"

# Check update_profiles calls
docker compose logs api | grep "update_profiles" | tail -10
```

### 6. Full Profile Rebuild (Last Resort)
```bash
# Rebuild from Kafka transaction history (24h)
docker compose run --rm dashboard python -c "
from behavior.storage.redis_store import get_feature_store
from behavior.services.behavior_engine import get_behavior_engine
engine = get_behavior_engine()
engine.rebuild_profiles_from_kafka('bank_ng_gtb', hours=24)
"
```

---

## Validation

```bash
# 1. Check health endpoint
curl -s "http://localhost:8000/v1/admin/behavioral-health/bank_ng_gtb" | jq '.cold_start_fallback_rate'

# 2. Score test transaction - verify behavioral features present
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "bank_ng_gtb",
    "account_id": "test_acct_123",
    "amount": 50000,
    "currency": "NGN",
    "timestamp": "2026-07-18T12:00:00Z",
    "transaction_type": "PAYMENT",
    "channel": "MOBILE",
    "device_id": "dev_test_123",
    "country_code": "NG",
    "merchant_id": "merch_test"
  }' | jq '{decision, risk_score, features: .features | keys[] | select(. | contains("trust") or contains("velocity") or contains("novelty") or contains("similarity"))}'

# 3. Verify trust scores in response
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "bank_ng_gtb", "account_id": "test_acct", "amount": 1000, "currency": "NGN", "timestamp": "2026-07-18T12:00:00Z", "transaction_type": "PAYMENT", "channel": "MOBILE"}' | jq '.explanation.behavioral'

# 4. Check fallback rate decreasing
curl -s "http://localhost:8000/v1/admin/cold-start-fallback/bank_ng_gtb" | jq '.fallback_rate'
```

---

## Post-Resolution

1. **Monitor 2h** for fallback rate decrease
2. **Update runbook** if new failure mode discovered
3. **Alert data engineering** if upstream transaction quality issue
4. **Review TTL settings** if profiles expiring too quickly

---

## Prevention

| Measure | Implementation |
|---------|----------------|
| **Profile TTL** | 30 days for active, 7 days for inactive |
| **Circuit breaker** | On profile store, fallback to hierarchy |
| **Profile freshness alerts** | Alert if > 10% profiles stale > 24h |
| **Cold-start metrics** | Dashboard fallback rate by level |
| **Trust score monitoring** | Alert if > 90% scores at extremes |

---

## Contacts

| Role | Contact |
|------|---------|
| On-call | [Name] - Slack/Phone |
| ML Engineer | [Name] - Profile/Feature issues |
| Data Engineer | [Name] - Transaction ingestion |
| Infra | [Name] - Redis/Storage |

---

*Last reviewed: 2026-07-18 | Next review: 2026-10-18*