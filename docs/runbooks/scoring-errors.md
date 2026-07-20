# Runbook: Scoring Errors Spike

**Alert**: `SCORING_ERRORS`  
**Severity**: CRITICAL  
**Auto-Resolve**: No

---

## Alert Meaning

Scoring endpoint returning 5xx errors at rate > 1%:
- **Error rate > 1%** for 5 minutes
- **5xx rate spike** sudden increase
- **Timeout errors** > 5% of requests

---

## Quick Diagnosis (3 min)

### 1. Check Error Rate
```bash
# Prometheus
curl -s http://localhost:8000/metrics | grep fraudtrap_requests_total | grep '5..' | awk -F' ' '{sum+=$2} END {print sum}'

# Or logs
docker compose logs api --since=5m | grep -c "5.."
```

### 2. Categorize Errors
```bash
# Error breakdown
docker compose logs api --since=5m | grep "ERROR\|5.." | \
  awk '{print $NF}' | sort | uniq -c | sort -rn
```

### 3. Check Dependencies
```bash
# Redis
redis-cli ping

# Kafka
kafka-broker-api-versions --bootstrap-server localhost:9092

# ClickHouse
clickhouse-client --query "SELECT 1"
```

---

## Common Causes

| Cause | Symptoms | Fix |
|-------|----------|-----|
| **Redis timeout** | `redis.exceptions.TimeoutError` | Redis load, network |
| **Kafka send failure** | `KafkaError` on emit | Kafka down, quota |
| **Model inference error** | `AttributeError`, `ValueError` | Bad model artifact |
| **Feature computation** | `KeyError`, `TypeError` | Feature engineering bug |
| **Kafka backpressure** | `BufferError`, `QueueFull` | Kafka down/slow |
| **ClickHouse insert fail** | `DatabaseError` | ClickHouse down |
| **OOM** | `MemoryError` | Model too large, leak |
| **Circuit breaker open** | `CircuitOpenException` | Dependency failing |

---

## Resolution by Error Type

### 1. Redis Timeout
```bash
# Check Redis
redis-cli --latency-history -i 1

# Fix: scale Redis, add replica
docker compose up -d --scale redis=2
```

### 2. Kafka Failure
```bash
# Check Kafka
kafka-broker-api-versions --bootstrap-server localhost:9092

# Restart if needed
docker compose restart kafka
```

### 3. Model Inference Error
```bash
# Check model
curl -s http://localhost:8000/v1/admin/models/status | jq .

# Reload
curl -X POST http://localhost:8000/v1/admin/models/reload
```

### 4. Feature Engineering Bug
```bash
# Check logs
docker compose logs feature_engineering | grep -i error | tail -20

# Common: new feature name, division by zero
```

### 4. OOM
```bash
# Check memory
docker stats --no-stream

# Restart with more memory
docker compose up -d --scale api=4  # Horizontal scale
# Or increase memory limit in docker-compose.yml
```

### 5. Circuit Breaker Open
```bash
# Check status
curl -s http://localhost:8000/v1/admin/circuit-breakers | jq .

# Reset
curl -X POST http://localhost:8000/v1/admin/circuit-breakers/reset
```

---

## Validation

```bash
# Test scoring
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/v1/score \
    -H "Content-Type: application/json" \
    -d '{"tenant_id":"bank_ng_gtb","account_id":"test","amount":1000,"currency":"NGN","timestamp":"2026-07-15T12:00:00Z","transaction_type":"PAYMENT","channel":"MOBILE"}' | jq -r '.decision'
done
```

---

## Post-Incident

1. **Root cause** in 24h
2. **Add monitoring** for new error type
3. **Update runbook** if new error type
4. **Chaos test** the failure mode

---

## Contacts

| Role | Contact |
|------|---------|
| On-call | [Name] - Slack/Phone |
| Backend Engineer | [Name] - Code issues |
| ML Engineer | [Name] - Model issues |
| Infra | [Name] - Redis/Kafka/ClickHouse |

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*