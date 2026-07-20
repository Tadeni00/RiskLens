# FraudTrap — Project Plan

## Vision

Real-time fraud detection platform for African banks and fintechs.

A production-grade system that accepts transaction payloads from client institutions, computes risk signals, applies deterministic rules, scores with the appropriate ML model per tenant phase, and returns a decision — all within a 100ms P95 latency SLA.

---

## Architecture

### Core Principles

- **Three-phase model lifecycle**: Cold Start → Semi-Supervised → Supervised, evolving as labeled data accumulates per tenant
- **Champion-Challenger model evaluation**: CatBoost champion replaces full stacking ensemble for production inference
- **Real-time scoring**: <100ms P95 latency with graceful degradation on dependency failures
- **Multi-tenant isolation**: Per-tenant models, Redis key namespacing, Kafka topic isolation, ClickHouse partitioning

### Infrastructure Stack

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| **FastAPI** | Scoring API (`/v1/score`), admin endpoints, health checks | `config/settings.py` |
| **Redis** | Online feature store, recent score cache, behavioral profiles | TTL 86,400s, namespaced keys |
| **Kafka** | Event backbone — transactions, labels, audit, drift alerts | 5 topics: `raw`, `scored`, `labels`, `audit`, `drift` |
| **ClickHouse** | Analytical rollups, drift metrics, feature distributions | Columnar storage for fast aggregation |
| **PostgreSQL** | Metadata, model registry mirror, label persistence | Row-level security per tenant |
| **MLflow** | Experiment tracking, model versioning | `mlflow-tracking-uri` |
| **Streamlit** | Dashboard (7 pages) | `http://localhost:8501` |
| **Docker Compose** | Local development stack (10 services) | `docker/docker-compose.yml` |

### Scoring Pipeline

```
HTTP Request
    │
    ▼
Validate & Parse Payload
    │
    ▼
Feature Assembly (Redis) ──── ~5–15ms
    │
    ▼
Rules Engine (Tier 1) ─────── <1ms
    │
    ├── Hard Block? ──▶ BLOCK (1.0) — return early
    │
    ▼
Behavioral Engine (Velocity, Trust, Similarity, Novelty)
    │
    ▼
ML Model (Tier 2) ──────────── 10–50ms
    │
    ├── Cold Start Ensemble (Phase 1)
    ├── Semi-Supervised Bridge (Phase 2)
    ├── Champion CatBoost (Phase 3)
    └── Supervised Stacking Ensemble (Phase 3)
    │
    ▼
Policy Floor: max(model_score, heuristic_score)
    │
    ▼
Soft Rule Boost (if triggered)
    │
    ▼
Decision: APPROVE (<0.40) | REVIEW (0.40–0.85) | BLOCK (≥0.85)
    │
    ├── Explainability (SHAP / feature importance / component weights)
    ├── Async GNN Scoring (Tier 3, non-blocking)
    └── Audit Emit → Kafka + Redis
```

### Resilience Patterns

- **Circuit breaker** on Redis, Kafka, ClickHouse, Model Inference, Behavioral Engine
- **Graceful degradation**: Redis down → heuristic scoring; Model missing → rules-only; Behavioral down → cold-start hierarchy
- **Model hot-reload**: Watchdog filesystem observer → atomic double-buffered swap → zero-downtime reload
- **Feature validation**: NaN/Inf checks, schema compatibility validation, auto-register missing schemas

---

## Current Status (Phase 4: Production-Ready)

### Completed

| Component | Details |
|-----------|---------|
| Core Scoring API | FastAPI with `/v1/score`, `/v1/health`, `/v1/admin/*`, Swagger docs |
| Rules Engine | 13 production rules across 5 tiers (velocity, amount, device, geo, behavioral) |
| Cold Start (Phase 1) | Isolation Forest + VAE + Empirical Tail Detector, weighted ensemble (0.4/0.35/0.25) |
| Semi-Supervised (Phase 2) | Isolation Forest + VAE ensemble with pseudo-labeling and XGBoost calibration |
| Supervised (Phase 3) | CatBoost champion with native categorical handling; XGBoost/LightGBM/FTTransformer/TabNet challengers |
| Champion-Challenger | Evaluation pipeline with PR-AUC gating (0.65 cold→semi, 0.78 semi→supervised) |
| Model Registry | Promote/rollback/archive with version pinning (model_version, training_hash, feature_hash, dataset_hash) |
| Probability Calibration | Isotonic and Platt scaling on validation set |
| Dashboard (Streamlit) | 7 pages: Overview, Live Monitoring, EDA, Drift, Lifecycle, Compliance, Behavioral |
| Behavioral Intelligence | Customer/merchant/device/beneficiary profiles with velocity, trust, similarity, novelty features |
| Feature Engineering | Redis pipeline with 5 velocity windows (1/5/60/1440/10080 minutes) |
| Docker Containerization | 10-service stack with health checks, graceful shutdown, resource limits |
| Uncertainty Estimation | MC Dropout, Temperature Scaling, Conformal Prediction |
| Explainability | SHAP (supervised), component weights (cold-start), feature importance (champion), rules explanation |

### In Progress

| Item | Status | Notes |
|------|--------|-------|
| Dashboard container rebuild | Fixing Dockerfile for production readiness | Ensuring all 7 pages render correctly |
| End-to-end notebook fixes | Resolving import paths and data flow issues | For local development and demo |

### Planned

| Item | Priority | Description |
|------|----------|-------------|
| GNN anomaly detection integration | High | Graph Neural Network scorer for account-device-merchant-IP relationship anomalies |
| Active learning loop | High | Prioritize labeling uncertain transactions for faster phase progression |
| A/B testing framework | Medium | Compare model variants in production with statistical significance |
| Multi-tenant isolation hardening | Medium | Resource quotas, rate limits, RLS policies |
| Real-time drift monitoring dashboard | Medium | PSI/KL trends per feature, concept drift alerts |
| Alerting system | High | PagerDuty (critical), Slack (warning), 15-min deduplication, runbook links |
| Model explainability dashboard | Medium | Per-decision SHAP waterfall, feature importance trends |

---

## Roadmap

### Phase 1: Foundation (Complete)

- Transaction ingestion schema (Pydantic models)
- Kafka producer/consumer for event streaming
- Redis online feature store
- Rules engine with deterministic fraud signals
- Heuristic scorer (day-zero fallback)
- Health checks and basic API endpoints

### Phase 2: ML Models (Complete)

- Cold-start ensemble (Isolation Forest + VAE + Empirical Tail Detector)
- Semi-supervised bridge with pseudo-labeling
- Feature engineering pipeline (5 velocity windows)
- Model training scripts and data generation
- MLflow experiment tracking integration

### Phase 3: Supervised Learning (Complete)

- Champion-Challenger model evaluation
- CatBoost champion with native categorical handling
- XGBoost/LightGBM/FTTransformer/TabNet challengers
- Probability calibration (Isotonic, Platt)
- Model registry with promote/rollback/archive
- Version pinning for reproducibility

### Phase 4: Production Hardening (Current)

- Streamlit dashboard (7 pages)
- Behavioral intelligence layer (Phase 7)
- Customer, merchant, device, beneficiary, payment instrument profiles
- Velocity, trust, similarity, novelty, heuristic feature generation
- Cold-start hierarchy fallback (Customer → Merchant → Tenant → Global)
- Docker containerization (10 services)
- Circuit breaker and graceful degradation patterns
- Model hot-reload with watchdog and atomic swap
- Uncertainty estimation (MC Dropout, Temperature Scaling, Conformal Prediction)

### Phase 5: Scale (Future)

- Multi-region deployment (Nigeria, Kenya, South Africa)
- GPU inference for deep models (VAE, FTTransformer, TabNet)
- Real-time model retraining pipeline
- Streaming feature computation (Kafka Streams / Flink)
- Cross-tenant pattern sharing (federated learning foundation)
- Compliance certifications (PCI DSS, SOC 2)

---

## Deployment

### Docker Compose (Local Development)

10 services managed via `docker/docker-compose.yml`:

```
api                 FastAPI scoring API (port 8000)
dashboard           Streamlit dashboard (port 8501)
live_simulator      Continuous synthetic transaction generator
redis               Online feature store + cache
kafka               Event streaming backbone
zookeeper           Kafka coordination
clickhouse          Analytical storage
postgres            Metadata + label persistence
mlflow              Experiment tracking (port 5000)
```

Start command:
```powershell
docker compose -f docker\docker-compose.yml up -d api dashboard live_simulator
```

### Kubernetes (Production-Ready Path)

- Horizontal Pod Autoscaler for `api` pods based on P95 latency
- StatefulSet for Redis, Kafka, ClickHouse, PostgreSQL
- ConfigMap/Secrets for environment-driven settings (`config/settings.py`)
- Ingress controller for API and dashboard endpoints

### CI/CD (GitHub Actions)

- Lint (ruff) + typecheck (mypy) on push
- Unit + integration tests on PR
- Docker image build + push to registry on merge to main
- Automated training pipeline trigger on label accumulation

---

## Team

Solo developer.

---

## Project Structure

```
fraudtrap/
├── api/                    # FastAPI app and HTTP endpoints
├── behavior/               # Behavioral Intelligence Layer
│   ├── profiles/           # Online behavioral profiles
│   ├── feature_generation/ # Velocity, trust, similarity, novelty features
│   ├── storage/            # RedisFeatureStore, InMemory, Mock
│   └── services/           # BehaviorEngine orchestration
├── config/                 # Environment-driven settings (Pydantic)
├── dashboard/              # Streamlit app and pages
├── docker/                 # Dockerfiles and compose file
├── features/               # Feature engineering (Redis pipelines)
├── ingestion/              # Kafka producer/consumer, schemas
├── models/                 # Model implementations
│   ├── cold_start/         # Isolation Forest + VAE ensemble
│   ├── supervised/         # CatBoost champion, stacking ensemble, semi-supervised
│   └── gnn/                # Graph Neural Network scorer
├── monitoring/             # Drift detection, metrics, alerts, rollup
├── scoring/                # Orchestrator, rules engine, calibration, validation
├── scripts/                # Data generation, training, simulation
├── training/               # Dataset builder, pipeline
├── artifacts/              # Generated data + model artifacts
├── tests/                  # Unit, integration, chaos tests
└── docs/runbooks/          # Operational runbooks
```

---

*Last updated: 2026-07-20*
