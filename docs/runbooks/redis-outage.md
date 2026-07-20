# Runbook: Redis Outage

**Alert**: `REDIS_DOWN` / Circuit Breaker Open  
**Severity**: CRITICAL  
**Auto-Resolve**: No (requires manual verification)

---

## Impact

| Component | Behavior | Fallback |
|-----------|----------|----------|
| Velocity features | Fail → zeros | Heuristic scoring |
| Seen entities (device/merchant) | Fail → all "new" | Heuristic boost |
| Historical stats | Fail → defaults | Heuristic floor |
| Geo features | Fail → zeros | Heuristic |
| Blocklists | Fail open (allow) | Allow by default |
| **Behavioral profiles** | Fail → MockFeatureStore | Heuristic + cold-start hierarchy |
| **Customer profiles** | Fail → tenant/global fallback | Cold-start hierarchy (Merchant → Tenant → Global) |
| **Merchant profiles** | Fail → tenant/global fallback | Cold-start hierarchy (Tenant → Global) |
| **Device profiles** | Fail → tenant/global fallback | Cold-start hierarchy (Tenant → Global) |
| **Beneficiary profiles** | Fail → global fallback | Cold-start hierarchy (Global) |
| **Payment instrument profiles** | Fail → global fallback | Cold-start hierarchy (Global) |
| **Feature generation** | Fail → zeros | Heuristic scoring |

**Expected**: Scoring continues with heuristic model, P95 < 100ms

---

## Quick Diagnosis (2 min)

### 1. Confirm Redis Down
```bash
redis-cli ping
# Expected: PONG
# Actual: Connection refused / timeout
```

### 2. Check Circuit Breaker
```bash
curl -s http://localhost:8000/v1/admin/circuit-breakers | jq .
# Look for: {"redis": {"state": "open", "failures": 5, "last_failure": "..."}}
```

### 3. Check API Health
```bash
curl -s http://localhost:8000/health | jq .
# Should show "degraded" not "unhealthy"
```

---

## Root Cause Categories

| Cause | Symptoms | Resolution |
|-------|----------|------------|
| **Redis process dead** | `redis-cli ping` fails | Restart Redis |
| **OOM killed** | `dmesg | grep -i oom` shows redis | Increase memory, restart |
| **Network partition** | `ping redis-host` works, `redis-cli` fails | Check firewall, network |
| **Max clients reached** | `redis-cli info clients` shows max | Increase maxclients |
| **AOF rewrite blocking** | `INFO persistence` shows `aof_rewrite_in_progress:1` | Wait or disable AOF rewrite |

---

## Resolution Steps

### 1. Redis Process Dead
```bash
# Check process
systemctl status redis
# or
docker compose ps redis

# Restart
systemctl restart redis
# or
docker compose restart redis

# Verify
redis-cli ping && echo "Redis recovered"
```

### 2. OOM Killed
```bash
# Check logs
dmesg -T | grep -i redis | tail -5
# Look for: Out of memory: Kill process xxx (redis-server)

# Fix: Increase memory limit
# In docker-compose.yml:
#   redis:
#     deploy:
#       resources:
#         limits:
#           memory: 4G  # was 2G

# Restart with new limit
docker compose up -d redis
```

### 3. Max Clients Reached
```bash
# Check
redis-cli info clients
# Look for: connected_clients: 10000 (near maxclients)

# Quick fix: increase maxclients
redis-cli CONFIG SET maxclients 20000

# Permanent: in redis.conf
# maxclients 20000
```

### 4. AOF Rewrite Blocking
```bash
# Check status
redis-cli INFO persistence | grep aof_rewrite

# If in progress, wait or:
redis-cli CONFIG SET appendonly no
redis-cli CONFIG SET appendonly yes
```

### 5. Network Partition
```bash
# Test connectivity
ping redis-host
telnet redis-host 6379

# Check firewall
iptables -L -n | grep 6379

# Check DNS
dig redis-host
```

---

## Verification After Fix

```bash
# 1. Redis responsive
redis-cli ping

# 2. Circuit breaker closed
curl -s http://localhost:8000/v1/admin/circuit-breakers | jq '.redis.state'
# Should be "closed"

# 3. Scoring works
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"bank_ng_gtb","account_id":"test","amount":100,"currency":"NGN","timestamp":"2026-07-15T12:00:00Z","transaction_type":"PAYMENT","channel":"MOBILE"}'

# 4. Velocity features working
curl -s "http://localhost:8000/v1/score" -X POST -H "Content-Type: application/json" \
  -d '{"tenant_id":"bank_ng_gtb","account_id":"test","amount":100,"currency":"NGN","timestamp":"2026-07-15T12:00:00Z","transaction_type":"PAYMENT","channel":"MOBILE"}' | jq '.triggered_rules'
# Should show velocity rules if applicable
```

---

## Post-Incident

1. **Root Cause Analysis** within 24h
2. **Update runbook** if new failure mode
3. **Check Redis capacity** - plan capacity increase
4. **Review circuit breaker** thresholds

---

## Prevention

| Measure | Implementation |
|---------|----------------|
| **Redis HA** | Redis Sentinel or Cluster |
| **Memory monitoring** | Alert at 70% memory usage |
| **Connection pooling** | Tune pool size, idle timeout |
| **Circuit breaker tuning** | Adjust failure threshold/timeout |
| **Graceful degradation testing** | Monthly chaos test |

---

## Contacts

| Role | Contact |
|------|---------|
| On-call | [Name] - Slack/Phone |
| Redis Admin | [Name] - Slack/Phone |
| Infra Lead | [Name] - Slack/Phone |

---

*Last reviewed: 2026-07-15 | Next review: 2026-10-15*