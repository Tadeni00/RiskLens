# FraudTrap API Documentation v2.0

Production-grade fraud detection API for banks and fintechs, implementing a three-layer model lifecycle: Cold Start → Adaptive Learning → Supervised.

---

## Base URLs

| Environment | URL |
|-------------|-----|
| Production  | `https://api.fraudtrap.io/v1` |
| Development | `http://localhost:8000/v1` |

---

## Authentication

All requests require one of:

| Method | Header | Description |
|--------|--------|-------------|
| API Key | `X-API-Key: ft_live_xxxxxxxxxxxx` | Server-to-server |
| Bearer JWT | `Authorization: Bearer eyJhbG...` | Short-lived tokens |

---

## Rate Limits

| Limit | Scope |
|-------|-------|
| 1,000 req/min | Per tenant |

Exceeded limits return `429 Too Many Requests`.

---

## SLA

| Metric | Target |
|--------|--------|
| P95 latency (scoring) | < 100ms |
| Availability | 99.95% |

---

## Endpoints

### Scoring

#### `POST /v1/score` — Score a transaction

Scores a single transaction for fraud risk.

**Request Body:**

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "bank_ng_gtb",
  "account_id": "tok_acct_123",
  "session_id": "tok_sess_789",
  "amount": 45000.00,
  "currency": "NGN",
  "timestamp": "2026-07-15T14:23:11Z",
  "transaction_type": "PAYMENT",
  "channel": "MOBILE",
  "merchant_id": "tok_merch_456",
  "merchant_category_code": "5411",
  "merchant_country": "NG",
  "counterparty_account_id": null,
  "device_id": "tok_dev_001",
  "device_type": "MOBILE",
  "ip_address_hash": "a1b2c3d4",
  "user_agent_hash": "e5f6g7h8",
  "latitude": 6.5244,
  "longitude": 3.3792,
  "country_code": "NG",
  "typing_cadence_ms": 120.5,
  "session_duration_seconds": 45.0,
  "field_visit_count": 3,
  "extra_fields": {}
}
```

**Response (200):**

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "bank_ng_gtb",
  "risk_score": 0.87,
  "decision": "BLOCK",
  "model_phase": "SUPERVISED",
  "model_version": "v2026.07.15",
  "latency_ms": 42.3,
  "explanation": {
    "model_type": "supervised",
    "base_value": 0.12,
    "prediction_value": 0.87,
    "top_features": [
      {
        "feature": "amount_zscore",
        "value": 4.2,
        "contribution": 0.31,
        "method": "shap"
      },
      {
        "feature": "velocity_1h",
        "value": 12,
        "contribution": 0.24,
        "method": "shap"
      },
      {
        "feature": "geo_anomaly",
        "value": 1.0,
        "contribution": 0.18,
        "method": "shap"
      }
    ],
    "components": null,
    "latency_ms": 8.1
  },
  "explanation_type": "sync",
  "triggered_rules": ["RULE_HIGH_VELOCITY", "RULE_GEO_MISMATCH"],
  "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scored_at": "2026-07-15T14:23:11.042Z"
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200  | Scoring success |
| 400  | Validation error |
| 429  | Rate limit exceeded |
| 500  | Scoring service error |

**curl:**

```bash
curl -X POST https://api.fraudtrap.io/v1/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx" \
  -d '{
    "tenant_id": "bank_ng_gtb",
    "account_id": "tok_acct_123",
    "amount": 45000.00,
    "currency": "NGN",
    "timestamp": "2026-07-15T14:23:11Z",
    "transaction_type": "PAYMENT",
    "channel": "MOBILE"
  }'
```

---

#### `POST /v1/score/batch` — Batch score transactions

Scores up to 1,000 transactions per request.

**Request Body:**

```json
{
  "transactions": [
    {
      "transaction_id": "550e8400-e29b-41d4-a716-446655440001",
      "tenant_id": "bank_ng_gtb",
      "account_id": "tok_acct_123",
      "amount": 12000.00,
      "currency": "NGN",
      "timestamp": "2026-07-15T15:00:00Z",
      "transaction_type": "TRANSFER",
      "channel": "WEB"
    },
    {
      "transaction_id": "550e8400-e29b-41d4-a716-446655440002",
      "tenant_id": "bank_ng_gtb",
      "account_id": "tok_acct_456",
      "amount": 250000.00,
      "currency": "NGN",
      "timestamp": "2026-07-15T15:01:00Z",
      "transaction_type": "PAYMENT",
      "channel": "MOBILE"
    }
  ]
}
```

**Response (200):**

```json
[
  {
    "transaction_id": "550e8400-e29b-41d4-a716-446655440001",
    "tenant_id": "bank_ng_gtb",
    "risk_score": 0.12,
    "decision": "APPROVE",
    "model_phase": "SUPERVISED",
    "model_version": "v2026.07.15",
    "latency_ms": 38.7,
    "explanation": null,
    "explanation_type": "none",
    "triggered_rules": [],
    "trace_id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
    "scored_at": "2026-07-15T15:00:00.039Z"
  },
  {
    "transaction_id": "550e8400-e29b-41d4-a716-446655440002",
    "tenant_id": "bank_ng_gtb",
    "risk_score": 0.74,
    "decision": "REVIEW",
    "model_phase": "SUPERVISED",
    "model_version": "v2026.07.15",
    "latency_ms": 41.2,
    "explanation": null,
    "explanation_type": "none",
    "triggered_rules": ["RULE_HIGH_AMOUNT"],
    "trace_id": "c3d4e5f6-a7b8-9012-cdef-345678901234",
    "scored_at": "2026-07-15T15:01:00.041Z"
  }
]
```

**curl:**

```bash
curl -X POST https://api.fraudtrap.io/v1/score/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx" \
  -d '{
    "transactions": [
      {
        "tenant_id": "bank_ng_gtb",
        "account_id": "tok_acct_123",
        "amount": 12000.00,
        "currency": "NGN",
        "timestamp": "2026-07-15T15:00:00Z",
        "transaction_type": "TRANSFER",
        "channel": "WEB"
      }
    ]
  }'
```

---

### Labels

#### `POST /v1/labels` — Ingest ground-truth labels

Submits fraud/legitimate labels for the training pipeline. Accepted asynchronously.

**Request Body:**

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "bank_ng_gtb",
  "label": 1,
  "label_source": "CHARGEBACK",
  "chargeback_reason_code": "10.4",
  "labelled_at": "2026-07-18T09:30:00Z",
  "confidence": 0.95
}
```

**Response (202):**

```json
{
  "status": "accepted",
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| `label` | Meaning |
|---------|---------|
| 1 | Fraud |
| 0 | Legitimate |

| `label_source` | Description |
|----------------|-------------|
| `CHARGEBACK` | Chargeback received |
| `MANUAL_REVIEW` | Analyst decision |
| `DISPUTE_RESOLVED` | Dispute investigation outcome |

**curl:**

```bash
curl -X POST https://api.fraudtrap.io/v1/labels \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx" \
  -d '{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "tenant_id": "bank_ng_gtb",
    "label": 1,
    "label_source": "CHARGEBACK",
    "chargeback_reason_code": "10.4",
    "labelled_at": "2026-07-18T09:30:00Z",
    "confidence": 0.95
  }'
```

---

### Operations

#### `GET /health` — Health check

**Response (200):**

```json
{
  "status": "ok",
  "version": "2.0.0",
  "environment": "production",
  "timestamp": "2026-07-15T14:30:00Z"
}
```

**curl:**

```bash
curl https://api.fraudtrap.io/health
```

---

#### `GET /metrics` — Prometheus metrics

Returns metrics in Prometheus exposition format (`text/plain`).

**curl:**

```bash
curl https://api.fraudtrap.io/metrics
```

---

#### `GET /v1/recent` — Recent scored transactions

| Parameter | In | Type | Default | Description |
|-----------|----|------|---------|-------------|
| `limit` | query | integer | 500 | Max results |
| `tenant_id` | query | string | — | Filter by tenant |

**Response (200):**

```json
{
  "count": 2,
  "items": [
    {
      "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
      "tenant_id": "bank_ng_gtb",
      "risk_score": 0.87,
      "decision": "BLOCK",
      "model_phase": "SUPERVISED",
      "model_version": "v2026.07.15",
      "latency_ms": 42.3,
      "explanation": null,
      "explanation_type": "none",
      "triggered_rules": ["RULE_HIGH_VELOCITY"],
      "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "scored_at": "2026-07-15T14:23:11.042Z"
    }
  ]
}
```

**curl:**

```bash
curl "https://api.fraudtrap.io/v1/recent?limit=100&tenant_id=bank_ng_gtb" \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx"
```

---

#### `GET /v1/phase/{tenant_id}` — Model phase status

Returns the current model phase and loaded models for a tenant.

**Path Parameter:** `tenant_id` (string, required)

**Response (200):**

```json
{
  "tenant_id": "bank_ng_gtb",
  "current_phase": "SUPERVISED",
  "model_version": "v2026.07.15",
  "loaded_models": {
    "cold_start": true,
    "adaptive_learning": true,
    "supervised": true,
    "simple_model": true
  },
  "available_model_tenants": {
    "bank_ng_gtb": ["SUPERVISED"],
    "fintech_ke_mpesa": ["SEMI_SUPERVISED"],
    "bank_gh_absa": ["COLD_START"]
  }
}
```

**curl:**

```bash
curl https://api.fraudtrap.io/v1/phase/bank_ng_gtb \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx"
```

---

#### `GET /v1/drift/{tenant_id}` — Drift metrics

Returns PSI, KL divergence, and other drift metrics for a tenant.

**Path Parameter:** `tenant_id` (string, required)

**Response (200):**

```json
{
  "tenant_id": "bank_ng_gtb",
  "n_baseline": 50000,
  "n_current": 12000,
  "metrics": {
    "amount": {
      "psi": 0.034,
      "kl_divergence": 0.021,
      "mean_shift": 0.15,
      "std_shift": -0.08,
      "drift_detected": false,
      "baseline_stats": {"mean": 32000, "std": 18500},
      "current_stats": {"mean": 33200, "std": 17800}
    },
    "hour_of_day": {
      "psi": 0.12,
      "kl_divergence": 0.08,
      "mean_shift": 1.2,
      "std_shift": 0.4,
      "drift_detected": true,
      "baseline_stats": {"mean": 14.5, "std": 4.2},
      "current_stats": {"mean": 15.7, "std": 4.6}
    }
  }
}
```

**curl:**

```bash
curl https://api.fraudtrap.io/v1/drift/bank_ng_gtb \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx"
```

---

#### `GET /v1/lifecycle/{tenant_id}` — Model lifecycle metrics

Returns scoring history, label counts, transition readiness, and loaded models for a tenant.

**Path Parameter:** `tenant_id` (string, required)

**Response (200):**

```json
{
  "tenant_id": "bank_ng_gtb",
  "current_phase": "SUPERVISED",
  "model_version": "v2026.07.15",
  "total_scored": 1284750,
  "fraud_labels": 842,
  "legit_labels": 12450,
  "decisions": {
    "APPROVE": 1180200,
    "REVIEW": 82400,
    "BLOCK": 22150
  },
  "phase_counts": {
    "COLD_START": 50000,
    "SEMI_SUPERVISED": 284750,
    "SUPERVISED": 950000
  },
  "pr_auc": 0.92,
  "avg_latency_ms": 41.5,
  "p95_latency_ms": 78.2,
  "scoring_history": [
    {"date": "2026-07-09", "transactions": 142000, "fraud_labels": 98, "avg_score": 0.14},
    {"date": "2026-07-10", "transactions": 138500, "fraud_labels": 112, "avg_score": 0.15},
    {"date": "2026-07-11", "transactions": 145200, "fraud_labels": 87, "avg_score": 0.13},
    {"date": "2026-07-12", "transactions": 131000, "fraud_labels": 95, "avg_score": 0.14},
    {"date": "2026-07-13", "transactions": 98000, "fraud_labels": 71, "avg_score": 0.16},
    {"date": "2026-07-14", "transactions": 140000, "fraud_labels": 103, "avg_score": 0.15},
    {"date": "2026-07-15", "transactions": 146200, "fraud_labels": 108, "avg_score": 0.14}
  ],
  "transition_readiness": {
    "fraud_labels": {"current": 842, "target": 1000, "pct": 84.2},
    "pr_auc": {"current": 0.92, "target": 0.85, "pct": 100.0},
    "champion_challenger": {"current": "champion_wins", "pct": 78.5}
  },
  "loaded_models": {
    "cold_start": true,
    "adaptive_learning": true,
    "supervised": true,
    "simple_model": true
  },
  "available_tenants": ["bank_ng_gtb", "fintech_ke_mpesa", "bank_gh_absa"]
}
```

**curl:**

```bash
curl https://api.fraudtrap.io/v1/lifecycle/bank_ng_gtb \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx"
```

---

### Explainability

#### `GET /v1/explain/{trace_id}` — Retrieve SHAP explanation

Returns the SHAP explanation for a previously scored transaction.

**Path Parameter:** `trace_id` (UUID, required)

**Response (200):**

```json
{
  "model_type": "supervised",
  "base_value": 0.12,
  "prediction_value": 0.87,
  "top_features": [
    {"feature": "amount_zscore", "value": 4.2, "contribution": 0.31, "method": "shap"},
    {"feature": "velocity_1h", "value": 12, "contribution": 0.24, "method": "shap"},
    {"feature": "geo_anomaly", "value": 1.0, "contribution": 0.18, "method": "shap"},
    {"feature": "device_risk_score", "value": 0.72, "contribution": 0.09, "method": "shap"},
    {"feature": "merchant_category_risk", "value": 0.65, "contribution": 0.07, "method": "shap"}
  ],
  "components": {"unsupervised": 0.35, "supervised": 0.52},
  "latency_ms": 8.1
}
```

| Status Code | Description |
|-------------|-------------|
| 200 | Explanation found |
| 404 | Trace ID not found |

**curl:**

```bash
curl https://api.fraudtrap.io/v1/explain/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "X-API-Key: ft_live_xxxxxxxxxxxx"
```

---

### Admin

#### `POST /v1/admin/models/reload` — Force model reload

Reloads models from disk for a specific tenant.

**Request Body:**

```json
{
  "tenant_id": "bank_ng_gtb",
  "force": true
}
```

**Response (200):**

```json
{
  "status": "reloaded",
  "tenant_id": "bank_ng_gtb",
  "models_loaded": ["cold_start", "adaptive_learning", "supervised"],
  "model_version": "v2026.07.15"
}
```

**curl:**

```bash
curl -X POST https://api.fraudtrap.io/v1/admin/models/reload \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ft_admin_xxxxxxxxxxxx" \
  -d '{"tenant_id": "bank_ng_gtb", "force": true}'
```

---

## Error Handling

All errors follow a consistent JSON structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "amount",
        "message": "must be greater than or equal to 0.01",
        "type": "validation_error"
      }
    ]
  }
}
```

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `VALIDATION_ERROR` | Request body validation failed |
| 401 | `UNAUTHORIZED` | Invalid or missing API key |
| 404 | `NOT_FOUND` | Resource not found |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Server-side failure |

---

## Webhooks (Future)

Planned Kafka topic events for async consumption:

| Topic | Event | Description |
|-------|-------|-------------|
| `fraudtrap.score.completed` | `ScoreCompleted` | Fired after each scoring request |
| `fraudtrap.labels.ingested` | `LabelIngested` | Fired after label acceptance |
| `fraudtrap.model.transitioned` | `ModelTransitioned` | Fired on phase change |
| `fraudtrap.drift.detected` | `DriftDetected` | Fired when PSI > threshold |

Kafka cluster: `kafka.fraudtrap.io:9092`, TLS required.

---

## Changelog

### v2.0.0 (2026-07-15)

- Initial release of FraudTrap API v2.
- Scoring endpoints: single and batch.
- Three-layer model lifecycle (Cold Start → Adaptive Learning → Supervised).
- SHAP-based explainability.
- Drift monitoring (PSI, KL divergence).
- Label ingestion pipeline.
- Admin model reload endpoint.
- Rate limiting (1,000 req/min/tenant).
- API Key and Bearer JWT authentication.
