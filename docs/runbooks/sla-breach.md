# Runbook: SLA Breach (P95 Latency > 90ms)

**Alert**: `SLA_BREACH`  
**Severity**: CRITICAL  
**Runbook Version**: 1.0  
**Last Updated**: 2026-07-15

---

## Overview
P95 scoring latency has exceeded 90ms for 5+ minutes. This impacts customer experience and may violate SLAs.

---

## Quick Diagnosis (2 minutes)

### 1. Check Current Latency
```bash
# Quick API check
curl -s http://localhost:8000/metrics | grep fraudtrap_latency_ms

# Or check recent scores
curl -s "http://localhost:8000/v1/recent?limit=100" | jq '.items[] | .latency_ms' | sort -n | tail -20
```

### 2. Check System Resources
```bash
# CPU/Memory
docker stats --no-stream

# Redis
redis-cli INFO stats | grep instantaneous_ops_per_sec
redis-cli --latency-history -i 1

# Kafka lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group fraudtrap-scoring-group --describe
```

### 3. Check Model Status
```bash
curl -s http://localhost:8000/v1/phase/bank_ng_gtb | jq .
```

---

## Common Causes & Fixes

### 1. Redis Latency/Outage (Most Common)
**Symptoms**: Feature engineering timeouts, fallback to zero features
**Diagnosis**: 
```bash
redis-cli --latency -h localhost -p 6379
redis-cli INFO | grep connected_clients
```
**Fix**:
- Check Redis memory: `redis-cli INFO memory`
- Check slowlog: `redis-cli SLOWLOG GET 10`
- Scale Redis or add replica
- Check network: `ping redis-host`

### 2. Model Inference Slowdown
**Symptoms**: Model phase shows SUPERVISED but latency high
**Diagnosis**:
```bash
# Check model load time
curl -s http://localhost:8000/v1/phase/bank_ng_gtb | jq .loaded_models
```
**Fix**:
- Model warmup: `POST /v1/admin/warmup`
- Check model size: `ls -lh artifacts/models/*/simple_model.pkl`
- Consider smaller model or quantization

### 3. Feature Engineering Bottleneck
**Symptoms**: High latency in feature assembly (< 5ms expected)
**Diagnosis**:
```bash
# Check feature assembly time in logs
grep "Feature assembly" /var/log/fraudtrap/*.log | tail -20
```
**Fix**:
- Check Redis pipeline usage
- Reduce feature count if possible
- Add Redis pipeline batching

### 4. Kafka Producer Backpressure
**Symptoms**: Audit emit latency spikes
**Diagnosis**:
```bash
kafka-producer-perf-test --topic fraudtrap.audit.decisions --num-records 10000 --record-size 1000 --throughput 10000
```
**Fix**:
- Increase `linger.ms` and `batch.size`
- Add more partitions
- Check broker disk I/O

### 5. Model Artifact Missing/Corrupt
**Symptoms**: Fallback to heuristic scoring, phase = UNSUPERVISED
**Fix**:
```bash
# Check model artifacts
ls -la artifacts/models/bank_ng_gtb/
# Re-train if needed
docker compose run --rm dashboard python scripts/train_simple_model.py --all-tenants
```

### 6. Kafka Consumer Lag
**Symptoms**: Labels not arriving, training data stale
**Fix**:
```bash
# Check consumer lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group fraudtrap-scoring-group --describe
# Restart consumer if lag > 10000
docker compose restart scoring_worker
```

---

## Escalation Path

| Time | Action | Owner |
|------|--------|-------|
| 0-5 min | Run quick diagnosis | On-call engineer |
| 5-15 min | Apply common fix | On-call engineer |
| 15-30 min | Engage ML team if model issue | ML Engineer |
| 30-60 min | Engage Infra team if Redis/Kafka | Infra Engineer |
| 60+ min | Page Manager, consider failover | Engineering Manager |

---

## Post-Incident

1. **Create Incident Report** within 24 hours
2. **Root Cause Analysis** within 48 hours
3. **Action Items** in Jira with owners
4. **Update Runbook** if new failure mode

---

## Useful Commands Cheatsheet

```bash
# Quick health
curl -s http://localhost:8000/health | jq .

# Recent scores with latency
curl -s "http://localhost:8000/v1/recent?limit=100" | jq '.items[] | {txn: .transaction_id, score: .risk_score, decision: .decision, latency: .latency_ms}'

# Model phases
curl -s "http://localhost:8000/v1/phase/bank_ng_gtb" | jq .

# Drift metrics
curl -s "http://localhost:8000/v1/drift/bank_ng_gtb" | jq .

# Prometheus metrics
curl -s http://localhost:8000/metrics | grep fraudtrap_

# Docker logs
docker compose logs -f --tail=100 api
docker compose logs -f --tail=100 scoring_worker

# Restart services
docker compose restart api
docker compose restart scoring_worker
```

---

## Contacts

| Role | Name | Contact |
|------|------|---------|
| On-call Engineer | [Name] | [Slack/Phone] |
| ML Engineer | [Name] | [Slack/Phone] |
| Infra Engineer | [Name] | [Slack/Phone] |
| Engineering Manager | [Name] | [Slack/Phone] |

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*