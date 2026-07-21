# FraudTrap

A production-grade, real-time fraud detection platform for banks, fintechs, and financial institutions. Accepts transaction payloads via API, returns risk scores and decisions, stores recent decisions for dashboard use, ingests fraud labels, and includes a Streamlit dashboard plus a live traffic simulator.

The system implements a **three-phase model lifecycle** that evolves from unsupervised anomaly detection to fully supervised fraud classification as labeled data accumulates:

| Phase | Name | Models | Labels Required |
|-------|------|--------|-----------------|
| 1 | UNSUPERVISED (Cold Start) | VAE + Isolation Forest + Empirical Tail Detector | None |
| 2 | SEMI_SUPERVISED | Pseudo-labeling + XGBoost calibration | 5,000+ fraud labels |
| 3 | SUPERVISED | Stacked Ensemble (XGBoost + LightGBM + CatBoost → Logistic Regression) | 50,000+ fraud labels |

A **Champion-Challenger** architecture sits on top of Phase 3: CatBoost serves as the production champion while XGBoost, LightGBM, FT-Transformer, and TabNet run as offline challengers with automated promotion on statistical significance.

**Phase 7 (Behavioral Intelligence Layer)** provides online behavioral profiles for customers, merchants, devices, beneficiaries, and payment instruments — real-time velocity, trust scores, similarity, and novelty detection with Redis hot storage and PostgreSQL durability.

## What The System Does

FraudTrap receives transaction payloads from client institutions, computes risk signals, applies deterministic rules, scores with the appropriate ML model for the tenant's phase, applies a conservative policy floor, and returns a decision:

```
APPROVE  - low risk (score < 0.40)
REVIEW   - suspicious (0.40 ≤ score < 0.85)
BLOCK    - high risk or hard rule fired (score ≥ 0.85)
```

The dashboard shows transaction volume, decision mix, latency, fraud signals, model lifecycle state, Champion-Challenger status, explainability (SHAP), drift views, and compliance views.

---

## Architecture

```
Client / Simulator
        |
        v
FastAPI /v1/score
        |
        +-- Feature Assembly (Redis)
        +-- Rules Engine (Tier 1, <1ms)
        +-- Behavioral Engine (Phase 7 features)
        +-- ML Model (Tier 2, 10-50ms)
        +-- Champion-Challenger routing
        +-- GNN Score (Tier 3, async)
        |
        +-- Recent scores → Redis (dashboard cache)
        +-- Scored transactions + audit events → Kafka
        +-- Labels via /v1/labels → Kafka
        v
Dashboard reads /v1/recent
```

### Key Components

| Layer | Components |
|-------|------------|
| **API** | FastAPI with Swagger docs, batch scoring, label ingestion, lifecycle endpoints |
| **Scoring Pipeline** | Feature engineering → Rules engine → Behavioral features → ML model → Policy floor → Decision |
| **ML Models** | Cold Start Ensemble, SemiSupervisedBridge, Supervised Ensemble, Champion (CatBoost), Challengers (XGBoost, LightGBM, FT-Transformer, TabNet) |
| **Behavioral Intelligence** | Customer, Merchant, Device, Beneficiary, Payment Instrument profiles with velocity/trust/similarity/novelty features |
| **Storage** | Redis (online features + cache), Kafka (event backbone), ClickHouse (analytics + drift), PostgreSQL (metadata + labels) |
| **Monitoring** | Drift detection (PSI/KL), metrics collector, alert manager, 12 operational runbooks |
| **MLOps** | MLflow experiment tracking, Champion-Challenger evaluation with statistical significance, automated promotion |

### Docker Services (11)

| Service | Purpose |
|---------|---------|
| `api` | FastAPI scoring API on `:8000` |
| `dashboard` | Streamlit dashboard on `:8501` |
| `live_simulator` | Continuous synthetic transaction generator |
| `training_worker` | Periodic model retraining loop |
| `label_worker` | Label ingestion from Kafka |
| `redis` | Online feature store + recent score cache |
| `kafka` | Event backbone for transactions, labels, audit |
| `zookeeper` | Kafka coordination |
| `clickhouse` | Analytical rollups and drift metrics |
| `postgres` | Metadata and label persistence |
| `mlflow` | Experiment tracking on `:5000` |

---

## Project Structure

```
fraudtrap/
├── api/                    # FastAPI app and HTTP endpoints
├── behavior/               # Behavioral Intelligence Layer (Phase 7)
│   ├── profiles/           # Online behavioral profiles
│   ├── feature_generation/ # Velocity, trust, similarity, novelty
│   ├── storage/            # RedisFeatureStore, InMemoryFeatureStore, MockFeatureStore
│   ├── services/           # BehaviorEngine orchestration
│   └── tests/              # Unit tests (17 passing)
├── config/                 # Environment-driven settings (Pydantic)
├── dashboard/              # Streamlit app (8 pages)
├── docker/                 # Dockerfiles and compose file
├── features/               # Feature engineering (Redis pipelines)
├── ingestion/              # Kafka producer/consumer, schemas
├── migrations/             # SQL migrations
├── mlops/                  # Drift + registry integration
├── models/
│   ├── cold_start/         # VAE + Isolation Forest + Tail Detector
│   ├── supervised/         # Champion, Challengers, Evaluator, Registry, Promotion
│   └── gnn/                # Graph Neural Network scorer
├── monitoring/             # Drift, metrics, alerts, rollup
├── scoring/                # Orchestrator, rules, experiments, calibration, model router
├── scripts/                # Data gen, training, simulation
├── tests/                  # Unit, integration, load tests
├── training/               # Dataset builder, pipeline
├── docs/
│   └── runbooks/           # 12 operational runbooks
├── .github/workflows/      # CI/CD: drift-monitor, model-ci, retrain-scheduler
├── FraudTrap_End_to_End_Notebook.ipynb    # Executive ML walkthrough (42 cells)
├── FraudTrap_API.postman_collection.json   # Postman collection (22 requests)
├── FraudTrap_Complete_Study_Guide.docx     # Platform study guide (11 chapters)
├── API_DOCUMENTATION_v2.md                 # Full API reference   
├── openapi.yaml                           # OpenAPI spec
└── requirements.txt                        # Core dependencies
```

---

## Quick Start

### Prerequisites

- Docker Desktop (Linux engine)
- Python 3.11+ (for local scripts outside Docker)

### Start the System

```powershell
cd C:\Users\Tommie-YV\Downloads\fraudtrap
docker compose -f docker\docker-compose.yml up -d api dashboard live_simulator
```

Check status:

```powershell
docker compose -f docker\docker-compose.yml ps
```

Expected:

```
NAME                  STATUS
api                   healthy
dashboard             running
live_simulator        running
redis                 healthy
kafka                 healthy
zookeeper             healthy
clickhouse            healthy
postgres              healthy
mlflow                running
```

Open:

- **API**: `http://localhost:8000`
- **API docs**: `http://localhost:8000/docs`
- **Dashboard**: `http://localhost:8501`
- **MLflow**: `http://localhost:5000`

---

## Generate Sample Training Data

```powershell
docker compose -f docker\docker-compose.yml run --rm dashboard python scripts/generate_sample_data.py --rows 50000 --fraud-rate 0.025
```

Outputs:

```
artifacts/data/bank_ng_gtb/features.parquet
artifacts/data/bank_ke_equity/features.parquet
artifacts/data/fintech_za_yoco/features.parquet
```

---

## Train Lightweight Tenant Models

```powershell
docker compose -f docker\docker-compose.yml run --rm dashboard python -m scripts.train_simple_model --all-tenants
```

Rebuild the API so it loads the new artifacts:

```powershell
docker compose -f docker\docker-compose.yml build api
docker compose -f docker\docker-compose.yml up -d api live_simulator dashboard
```

Verify a tenant model loaded:

```powershell
Invoke-RestMethod http://localhost:8000/v1/phase/bank_ng_gtb
```

---

## Simulate Live Traffic

The `live_simulator` service continuously posts synthetic transactions to `/v1/score`.

```powershell
docker compose -f docker\docker-compose.yml logs live_simulator --tail=50
```

---

## Test The API Manually

Health check:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

Score a transaction:

```powershell
$payload = @{
  tenant_id = "bank_ng_gtb"
  account_id = "tok_acct_123"
  amount = 45000
  currency = "NGN"
  timestamp = "2026-07-08T14:23:11Z"
  transaction_type = "PAYMENT"
  channel = "MOBILE"
  device_id = "tok_dev_001"
  country_code = "NG"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/v1/score -Method Post -ContentType "application/json" -Body $payload
```

Send a label:

```powershell
$label = @{
  transaction_id = "txn_demo_001"
  tenant_id = "bank_ng_gtb"
  label = 1
  label_source = "MANUAL_REVIEW"
  labelled_at = "2026-07-08T15:50:00Z"
  confidence = 0.95
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/v1/labels -Method Post -ContentType "application/json" -Body $label
```

---

## How Scoring Works

1. Redis-backed feature assembly (velocity, device, geo, behavioral)
2. Rules engine (Tier 1, <1ms) — hard blocks + soft boosts
3. Behavioral features from Phase 7 engine
4. Tenant-specific model selection (SimpleFraudModel / ColdStart / SemiSupervised / Supervised / Champion)
5. Policy floor: `risk_score = max(model_score, heuristic_score)`
6. Decision thresholds (`config/settings.py`):
   - `score < 0.40` → APPROVE
   - `0.40 ≤ score < 0.85` → REVIEW
   - `score ≥ 0.85` → BLOCK
7. Redis recent-score cache for dashboard
8. Audit emit to Kafka (scored txn + decision)

---

## Champion-Challenger Architecture

The system supports automated model promotion:

- **Champion**: CatBoost (production inference, native categorical handling)
- **Challengers**: XGBoost, LightGBM, FT-Transformer, TabNet (offline evaluation)
- **Routing**: `scoring/model_router.py` splits traffic (configurable `challenger_traffic_pct`, default 10%)
- **Evaluation**: Statistical significance testing (chi-squared) with PR-AUC gating
- **Promotion**: Automated when challenger is significantly better, with rollback support
- **Alerts**: Fires when challenger significantly beats or underperforms champion

---

## Model Artifacts & Serving

1. `scripts/generate_sample_data.py` → labeled feature rows (Parquet)
2. `scripts/train_simple_model.py` → trains one model per tenant
3. Each model saved as `artifacts/models/<tenant>/simple_model.pkl`
4. `api/main.py` calls `registry.load_from_disk(settings.model_dir)` at startup
5. `scoring/orchestrator.py` selects model for `txn.tenant_id`
6. `/v1/phase/{tenant}` reports loaded model status

---

## Root-Level Documentation

| File | Description |
|------|-------------|
| `FraudTrap_End_to_End_Notebook.ipynb` | Executive ML walkthrough — 42 cells covering all three phases with real model inference |
| `FraudTrap_API.postman_collection.json` | Postman collection with 22 requests across 4 folders |
| `FraudTrap_Complete_Study_Guide.docx` | Platform study guide — 11 chapters covering architecture, models, and operations |
| `API_DOCUMENTATION_v2.md` | Full API reference with auth, rate limits, SLA, all endpoints |
| `openapi.yaml` | OpenAPI specification |

---

## Common Commands

```powershell
# Rebuild changed services
docker compose -f docker\docker-compose.yml build api dashboard live_simulator
docker compose -f docker\docker-compose.yml up -d api dashboard live_simulator

# Stop/restart simulator
docker compose -f docker\docker-compose.yml stop live_simulator
docker compose -f docker\docker-compose.yml restart live_simulator

# View logs
docker compose -f docker\docker-compose.yml logs api --tail=100
docker compose -f docker\docker-compose.yml logs dashboard --tail=100

# Compile-check Python
python -m compileall api dashboard features ingestion mlops models scoring scripts training
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Docker cannot connect | Start Docker Desktop, wait for Linux engine |
| Dashboard shows no live data | `docker compose logs live_simulator --tail=50` + `Invoke-RestMethod http://localhost:8000/v1/recent?limit=5` |
| Generated data missing | `docker compose run --rm dashboard python scripts/generate_sample_data.py --rows 50000 --fraud-rate 0.025` |
| API returns `model_version: unloaded` | Train models, rebuild/recreate API, check `/v1/phase/{tenant}` |

---

## CI/CD

Three GitHub Actions workflows:

- **drift-monitor.yml** — Continuous drift monitoring
- **model-ci.yml** — Model validation pipeline
- **retrain-scheduler.yml** — Scheduled retraining

---

*FraudTrap — Real-time fraud detection for banks and fintechs.*
