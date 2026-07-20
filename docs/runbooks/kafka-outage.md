# Runbook: Kafka Outage / High Consumer Lag

**Alert**: `KAFKA_DOWN` / `KAFKA_CONSUMER_LAG`  
**Severity**: HIGH (Kafka down) / WARNING (Lag)  
**Auto-Resolve**: No

---

## Impact

| Component | Behavior | Mitigation |
|-----------|----------|------------|
| **Label ingestion** | Labels queue in producer buffer | Buffered locally, flushed on recovery |
| **Audit events** | Events buffered in memory | Flushed on recovery, max 10k events |
| **Training pipeline** | Labels delayed | Retrain uses stale labels |
| **GNN embeddings** | Graph updates delayed | Stale embeddings for GNN |

**Expected**: Scoring continues normally, labels/events replay on recovery

---

## Quick Diagnosis (3 min)

### 1. Check Kafka Brokers
```bash
# List brokers
kafka-broker-api-versions --bootstrap-server localhost:9092

# Check topics
kafka-topics --bootstrap-server localhost:9092 --list | grep fraudtrap
```

### 2. Check Consumer Groups
```bash
# All consumer groups
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list | grep fraudtrap

# Specific group lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group fraudtrap-scoring-group --describe
```

### 3. Check Topics
```bash
# Topic details
kafka-topics.sh --bootstrap-server localhost:9092 \
  --topic fraudtrap.labels.incoming --describe
```

---

## Root Cause Categories

| Cause | Symptoms | Resolution |
|-------|----------|------------|
| **Broker down** | `kafka-broker-api-versions` fails | Restart broker |
| **Partition leader missing** | `kafka-topics --describe` shows `-1` leader | Wait for election |
| **Disk full** | `df -h` shows 100% on Kafka disks | Add disk, increase retention |
| **Consumer lag** | Lag > 10k messages | Scale consumers |
| **Rebalance storm** | Frequent rebalances in logs | Fix session.timeout.ms |
| **Broker OOM** | Broker logs show OOM | Increase heap |

---

## Resolution Steps

### 1. Broker Down
```bash
# Check which broker
kafka-broker-api-versions --bootstrap-server localhost:9092 | grep -v "id:"

# Restart broker (in Docker)
docker compose restart kafka-1  # or kafka-2, kafka-3

# Verify
kafka-broker-api-versions --bootstrap-server localhost:9092
```

### 2. Partition Leader Missing
```bash
# Check
kafka-topics.sh --bootstrap-server localhost:9092 \
  --topic fraudtrap.labels.incoming --describe

# If leader is -1, wait for controller election
# Or force preferred replica election
kafka-leader-election.sh --bootstrap-server localhost:9092 \
  --election-type PREFERRED --topic fraudtrap.labels.incoming
```

### 3. Disk Full
```bash
# Check disk
df -h /var/lib/kafka

# Clean old segments (adjust retention)
kafka-configs.sh --bootstrap-server localhost:9092 \
  --entity-type topics --entity-name fraudtrap.labels.incoming \
  --alter --add-config retention.bytes=10737418240  # 10GB

# Or increase disk
# Add volume, mount, update log.dirs
```

### 4. Consumer Lag
```bash
# Check lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group fraudtrap-scoring-group --describe

# If lag > 10k: scale consumer
docker compose up -d --scale scoring_worker=4

# Or increase fetch size
# In consumer config:
# fetch.max.bytes=52428800
# max.poll.records=500
```

### 5. Rebalance Storm
```bash
# Check logs for frequent rebalances
docker compose logs kafka-1 | grep -i rebalance | tail -20

# Fix: increase session timeout
# In consumer config:
# session.timeout.ms=45000
# heartbeat.interval.ms=15000
```

### 6. Broker OOM
```bash
# Check heap
jcmd <kafka-pid> VM.native_memory summary

# Increase heap in docker-compose
# KAFKA_HEAP_OPTS: "-Xmx4g -Xms4g"
```

---

## Verification After Fix

```bash
# 1. Brokers healthy
kafka-broker-api-versions --bootstrap-server localhost:9092

# 2. Topics healthy
kafka-topics.sh --bootstrap-server localhost:9092 --list | grep fraudtrap

# 3. Consumer lag < 1000
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group fraudtrap-scoring-group --describe | awk '$6 > 1000'

# 4. Producer working
kafka-producer-perf-test --topic fraudtrap.labels.incoming \
  --num-records 100 --record-size 1000 --throughput 100 \
  --producer-props bootstrap.servers=localhost:9092

# 5. Labels flowing
curl -s "http://localhost:8000/v1/labels/recent?tenant=bank_ng_gtb&limit=10" | jq .
```

---

## Post-Incident

1. **Check data integrity**: Verify no label loss
2. **Reprocess missed labels** if any
3. **Update alerting** if new failure mode
4. **Review retention policies**

---

## Prevention

| Measure | Implementation |
|---------|----------------|
| **Multi-AZ brokers** | 3 brokers across AZs |
| **Disk monitoring** | Alert at 70% disk |
| **Consumer lag alert** | Alert at 5k lag |
| **Auto-scaling consumers** | K8S HPA on lag metric |
| **Multi-AZ consumers** | Deploy consumers per AZ |

---

## Contacts

| Role | Contact |
|------|---------|
| On-call | [Name] - Slack/Phone |
| Kafka Admin | [Name] - Slack/Phone |
| Infra Lead | [Name] - Slack/Phone |

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*