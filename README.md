# FraudTrap

FraudTrap is a production-grade, real-time fraud detection platform for banks, fintechs, and financial institutions. It provides a scoring API that accepts transaction payloads, returns risk scores and decisions, stores recent decisions for dashboard use, accepts fraud labels, and includes a Streamlit dashboard plus a live traffic simulator.

The system implements a **three-phase model lifecycle** that evolves from unsupervised anomaly detection to fully supervised fraud classification as labeled data accumulates:

- **Phase 1 (UNSUPERVISED / Cold Start)**: VAE + Isolation Forest + Empirical Tail Detector — no labels required
- **Phase 2 (SEMI_SUPERVISED)**: SemiSupervisedBridge — pseudo-labeling + XGBoost calibration
- **Phase 3 (SUPERVISED)**: Stacked Ensemble (XGBoost + LightGBM + CatBoost → Logistic Regression meta-learner) with isotonic calibration, temperature scaling, conformal prediction

**Phase 7 (Behavioral Intelligence Layer)**: Online behavioral profiles for customers, merchants, devices, beneficiaries, and payment instruments. Real-time feature generation (velocity, trust scores, similarity, novelty detection) with Redis hot storage and PostgreSQL durability. Multi-tenant isolation with cold-start hierarchy (Customer → Merchant → Tenant → Global).

The current local demo stack includes:

- FastAPI scoring API on `http://localhost:8000` with Swagger docs at `/docs`
- Streamlit dashboard on `http://localhost:8501`
- Redis online feature store + recent score cache
- Kafka event backbone for transactions, labels, audit, drift alerts
- ClickHouse for analytical rollups and drift metrics
- PostgreSQL for metadata and label persistence
- MLflow for experiment tracking
- Live transaction simulator for continuous demo traffic
- Lightweight trained tenant models (SimpleFraudModel) for fast serving

---

## 1. What The System Does

FraudTrap receives transaction payloads from client institutions, computes risk signals, applies deterministic rules, scores the transaction with the appropriate ML model for the tenant's phase, applies a conservative policy floor, and returns a decision:

```
APPROVE  - low risk (score < 0.40)
REVIEW   - suspicious (0.40 ≤ score < 0.85)
BLOCK    - high risk or hard rule fired (score ≥ 0.85)
```

The dashboard shows transaction volume, decision mix, latency, fraud signals, model lifecycle state, explainability (SHAP), drift views, and compliance views. During local development, a live simulator continuously posts synthetic transactions so the dashboard has moving data.

---

## 2. High-Level Architecture

```text
Client / Simulator
        |
        v
FastAPI /v1/score
        |
        +-- Feature Assembly (Redis)
        +-- Rules Engine (Tier 1, <1ms)
        +-- ML Model (Tier 2, 10-50ms)
        +-- GNN Score (Tier 3, async)
        |
        +-- Recent scores → Redis (dashboard cache)
        +-- Scored transactions + audit events → Kafka
        +-- Labels via /v1/labels → Kafka
        v
Dashboard reads /v1/recent
```

---

## 3. System Architectural Design

### 3.1 Component Architecture

```mermaid
graph TB
    subgraph "External"
        Client[Client / Bank / Fintech]
        Simulator[Live Simulator]
    end

    subgraph "API Layer"
        API[FastAPI /v1/score]
        Health[Health /v1/health]
        Admin[Admin /v1/admin/*]
        Dashboard[Streamlit Dashboard]
    end

    subgraph "Scoring Pipeline"
        FE[Feature Engineering\n(features/engineering.py)]
        RE[Rules Engine\n(scoring/rules_engine.py)]
        HE[Heuristic Scorer\n(scoring/heuristic.py)]
        BE[Behavioral Engine\n(behavior/services/behavior_engine.py)]
        MO[Model Orchestrator\n(scoring/orchestrator.py)]
    end

    subgraph "ML Models"
        CS[Cold Start Ensemble\n(models/cold_start/ensemble.py)]
        SS[Semi-Supervised Bridge\n(models/supervised/semi_supervised.py)]
        SV[Supervised Ensemble\n(models/supervised/ensemble.py)]
        GNN[GNN Scorer\n(models/gnn/gnn_scorer.py)]
        SM[Simple Model\n(scoring/simple_model.py)]
    end

    subgraph "Behavioral Intelligence (Phase 7)"
        BP[Behavioral Profiles\n(behavior/profiles/*)]
        FG[Feature Generation\n(behavior/feature_generation/*)]
        FS[Feature Store\n(behavior/storage/redis_store.py)]
    end

    subgraph "Storage & Messaging"
        Redis[(Redis\nOnline Features + Cache)]
        Kafka[(Kafka\nEvent Backbone)]
        CH[(ClickHouse\nAnalytics + Drift)]
        PG[(PostgreSQL\nMetadata + Labels)]
    end

    subgraph "Monitoring & Ops"
        MT[Metrics Collector\n(monitoring/metrics_collector.py)]
        DR[Drift Detection\n(monitoring/drift.py)]
        AL[Alert Manager\n(monitoring/alerts.py)]
        EX[Experiment Runner\n(scoring/experiments.py)]
    end

    Client --> API
    Simulator --> API
    API --> FE
    API --> RE
    API --> HE
    API --> BE
    API --> MO
    
    MO --> CS
    MO --> SS
    MO --> SV
    MO --> GNN
    MO --> SM
    
    BE --> BP
    BP --> FG
    FG --> FS
    FS --> Redis
    
    FE --> Redis
    RE --> Redis
    HE --> Redis
    
    API --> Kafka
    API --> CH
    API --> PG
    
    MT --> CH
    DR --> CH
    AL --> Kafka
    EX --> Kafka
    
    Dashboard --> API
    Dashboard --> CH
```

### 3.2 Data Flow Architecture

#### Scoring Request Flow

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TRANSACTION SCORING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │  HTTP    │───▶│  Validate &     │───▶│  Assemble        │───▶│  Rules    │  │
│  │  Request │    │  Parse Payload  │    │  Features        │    │  Engine   │  │
│  └──────────┘    └─────────────────┘    └──────────────────┘    └─────┬─────┘  │
│                                                                        │        │
│                          ┌─────────────────────────────────────────────┘        │
│                          ▼                                                      │
│                 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐  │
│                 │ Hard Block?   │────▶│  BLOCK (1.0)  │     │  Continue     │  │
│                 │  (Tier 1)     │     │  Return Early │     │  to Tier 2    │  │
│                 └───────────────┘     └───────────────┘     └───────┬───────┘  │
│                                                                       │          │
│                          ┌──────────────────────────────────────────┘          │
│                          ▼                                                     │
│                 ┌─────────────────┐     ┌──────────────────┐                  │
│                 │ Behavioral      │────▶│  Get ML Model    │                  │
│                 │ Engine          │     │  (Phase-based)   │                  │
│                 └─────────────────┘     └──────────────────┘                  │
│                          │                        │                             │
│                          ▼                        ▼                             │
│                 ┌─────────────────┐     ┌──────────────────┐                  │
│                 │ Behavioral      │     │  Score Model     │                  │
│                 │ Features        │     │  (Cold/Semi/     │                  │
│                 │ (Velocity,      │     │  Supervised)     │                  │
│                 │  Trust, Similar)│     └────────┬─────────┘                  │
│                 └─────────────────┘              │                             │
│                                                  ▼                             │
│                                         ┌──────────────────┐                  │
│                                         │ Policy Floor:    │                  │
│                                         │ max(model,       │                  │
│                                         │  heuristic)      │                  │
│                                         └────────┬─────────┘                  │
│                                                  │                             │
│                          ┌───────────────────────┘                             │
│                          ▼                                                     │
│                 ┌─────────────────┐     ┌──────────────────┐                  │
│                 │ Soft Rule       │────▶│  Final Decision  │                  │
│                 │ Boost?          │     │  (APPROVE/       │                  │
│                 └─────────────────┘     │  REVIEW/BLOCK)   │                  │
│                                         └────────┬─────────┘                  │
│                                                  │                             │
│                          ┌───────────────────────┘                             │
│                          ▼                                                     │
│                 ┌─────────────────┐     ┌──────────────────┐     ┌────────┐  │
│                 │ Explainability  │     │  Async GNN       │     │ Audit  │  │
│                 │ (SHAP/Rule      │     │  Scoring         │     │ Emit   │  │
│                 │  Contributions) │     │  (non-blocking)  │     │ Kafka  │  │
│                 └─────────────────┘     └──────────────────┘     └────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Behavioral Intelligence Data Flow (Phase 7)

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    BEHAVIORAL INTELLIGENCE LAYER                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Transaction                                                                   │
│       │                                                                         │
│       ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    BEHAVIOR ENGINE (Orchestrator)                        │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│   │  │  Customer   │  │  Merchant   │  │   Device    │  │ Beneficiary │    │   │
│   │  │  Profile    │  │  Profile    │  │  Profile    │  │  Profile    │    │   │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │   │
│   │         │                │                │                │            │   │
│   │         ▼                ▼                ▼                ▼            │   │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │  │              FEATURE GENERATION MODULES                         │   │   │
│   │  │  ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌───────────┐  │   │   │
│   │  │  │Velocity │ │Similarity│ │ Trust  │ │Novelty │ │ Heuristic │  │   │   │
│   │  │  └────┬────┘ └────┬────┘ └────┬───┘ └────┬───┘ └─────┬─────┘  │   │   │
│   │  │       │          │           │           │           │         │   │   │
│   │  └───────┼──────────┼───────────┼───────────┼───────────┼─────────┘   │   │
│   │          │          │           │           │           │             │   │
│   └──────────┼──────────┼───────────┼───────────┼───────────┼─────────────┘   │
│              │          │           │           │           │                 │
│              ▼          ▼           ▼           ▼           ▼                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                    UNIFIED FEATURE VECTOR (Dict[str, float])            │  │
│   │  velocity: acct_v_1h_count, acct_v_24h_mean_amt, dev_v_1h_count...     │  │
│   │  trust: device_trust_score, merchant_trust_score, customer_reputation..│  │
│   │  similarity: merchant_similarity, device_similarity, geo_speed_kmh...  │  │
│   │  novelty: is_new_device, is_new_merchant, is_new_country, is_new_ip    │  │
│   │  heuristic: heuristic_score                                             │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                           │
│                                    ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    COLD-START HIERARCHY (Fallback Chain)                │   │
│   │                                                                         │   │
│   │   Customer Profile (individual history)                                 │   │
│   │        │                                                                │   │
│   │        ▼ (fallback if insufficient history)                            │   │
│   │   Merchant Profile (merchant-level patterns)                           │   │
│   │        │                                                                │   │
│   │        ▼ (fallback)                                                    │   │
│   │   Tenant Profile (tenant-wide baselines)                               │   │
│   │        │                                                                │   │
│   │        ▼ (fallback)                                                    │   │
│   │   Global Profile (platform-wide defaults)                              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    FEATURE STORE (Multi-backend)                        │   │
│   │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │   │
│   │  │ RedisFeature   │  │ InMemoryFeature  │  │ MockFeatureStore       │  │   │
│   │  │ Store (Prod)   │  │ Store (Dev/Test) │  │ (Graceful Degradation) │  │   │
│   │  └────────────────┘  └──────────────────┘  └────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Three-Phase Model Lifecycle

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MODEL LIFECYCLE PHASES                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PHASE 1: UNSUPERVISED (Cold Start)                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • No labeled data required                                                │   │
│  │ • VAE (Variational Autoencoder) - reconstruction error                   │   │
│  │ • Isolation Forest - path length anomaly score                           │   │
│  │ • Empirical Tail Detector - robust z-score per feature                   │   │
│  │ • Ensemble: weighted combination (VAE: 0.4, IF: 0.35, Tail: 0.25)        │   │
│  │ • Score calibration via training percentiles                             │   │
│  │ • Explainability: per-component contribution                             │   │
│  │ • Gating: min 5 fraud labels + 500k txns + 8 weeks + PR-AUC ≥ 0.65       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  PHASE 2: SEMI_SUPERVISED                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Pseudo-labeling from Phase 1 scores (high confidence only)            │   │
│  │ • XGBoost trained on: confirmed labels + pseudo-labels (3:1 weight)     │   │
│  │ • CalibratedClassifierCV with isotonic calibration                       │   │
│  │ • Blended score: w_cold * cold_score + w_xgb * xgb_proba                │   │
│  │ • Graph-based label propagation (account-device-merchant-IP)             │   │
│  │ • Gating: min 5000 fraud labels + PR-AUC ≥ 0.78                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  PHASE 3: SUPERVISED                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Stacked ensemble: XGBoost + LightGBM + CatBoost → Logistic Regression │   │
│  │ • 5-fold out-of-fold stacking for meta-features                         │   │
│  │ • SMOTEENN for class imbalance                                          │   │
│  │ • Isotonic + Temperature scaling (blended 0.7/0.3)                      │   │
│  │ • Conformal prediction for uncertainty quantification                   │   │
│  │ • SHAP explanations for REVIEW band decisions (sync)                    │   │
│  │ • Behavioral features enable interaction learning                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Multi-Tenant Isolation

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-TENANT ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         TENANT NAMESPACING                              │   │
│  │                                                                         │   │
│  │  Redis Keys:     ft:{tenant_id}:{entity}:{id}:{feature}                │   │
│  │  Kafka Topics:   fraudtrap.{topic}.{tenant_id}                         │   │
│  │  ClickHouse:     tenant_id column + partition                           │   │
│  │  PostgreSQL:     tenant_id foreign key + RLS policies                  │   │
│  │  Model Artifacts: artifacts/models/{tenant_id}/                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         RESOURCE QUOTAS                                 │   │
│  │                                                                         │   │
│  │  bank_ng_gtb:     max_keys=10M, max_memory=2GB, rate_limit=1000/min    │   │
│  │  bank_ke_equity:  max_keys=5M,  max_memory=1GB, rate_limit=500/min     │   │
│  │  fintech_za_yoco: max_keys=3M,  max_memory=512MB, rate_limit=300/min   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         PHASE PER TENANT                                │   │
│  │                                                                         │   │
│  │  Each tenant independently progresses through phases:                  │   │
│  │  bank_ng_gtb      → SUPERVISED (has models)                            │   │
│  │  bank_ke_equity   → SEMI_SUPERVISED (some labels)                      │   │
│  │  fintech_za_yoco  → UNSUPERVISED (cold start only)                     │   │
│  │  new_tenant       → UNSUPERVISED (zero history)                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Observability & Monitoring Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       OBSERVABILITY STACK                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  METRICS (ClickHouse)                    LOGS (Structured JSON)               │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐   │
│  │ • Request latency (p50/p95/p99) │    │ • Transaction scored           │   │
│  │ • Decision distribution         │    │ • Model version loaded         │   │
│  │ • Fraud capture rate            │    │ • Rule triggered               │   │
│  │ • Model PR-AUC / ROC-AUC        │    │ • Drift alert fired            │   │
│  │ • Drift PSI / KL per feature    │    │ • Circuit breaker state        │   │
│  │ • Feature freshness             │    │ • Label received               │   │
│  │ • Behavioral fallback rates     │    │ • Profile updated              │   │
│  └─────────────────────────────────┘    └─────────────────────────────────┘   │
│              │                                          │                     │
│              ▼                                          ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      ALERT MANAGER (monitoring/alerts.py)              │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │ ALERT RULES                                                     │  │   │
│  │  ├─────────────────────────────────────────────────────────────────┤  │   │
│  │  │ SLA Breach        │ P95 > 90ms for 5min     │ Critical │ PD    │  │   │
│  │  │ Drift Spike       │ PSI > 0.25 any feature  │ Warning  │ Slack │  │   │
│  │  │ Concept Drift     │ Label rate change > 20% │ Warning  │ Slack │  │   │
│  │  │ Perf Drop         │ PR-AUC drop > 5%        │ Critical │ PD    │  │   │
│  │  │ Data Quality      │ >10% features zero      │ Warning  │ Slack │  │   │
│  │  │ Scoring Errors    │ 5xx rate > 1%           │ Critical │ PD    │  │   │
│  │  │ Behavioral Fallback│ Cold-start > 50%        │ Warning  │ Slack │  │   │
│  │  │ Profile Staleness  │ >10% profiles > 24h old  │ Warning  │ Slack │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                         │   │
│  │  Channels: PagerDuty (critical), Slack (warning)                       │   │
│  │  Deduplication: 15-min cooldown per alert type                          │   │
│  │  Runbook Links: Auto-attached to alert payload                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      DASHBOARD (Streamlit)                              │   │
│  │                                                                         │   │
│  │  • Overview: Volume, fraud rate, decision mix, latency                 │   │
│  │  • Live Monitoring: Real-time scoring, recent transactions             │   │
│  │  • EDA: Feature distributions, correlations, data quality              │   │
│  │  • Drift: PSI/KL trends, feature importance, embedding drift           │   │
│  │  • Lifecycle: Phase status, model versions, gating criteria            │   │
│  │  • Compliance: Explainability, audit trail, GDPR/PCI                   │   │
│  │  • Behavioral: Cold-start hierarchy, trust scores, profile freshness   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 Resilience Patterns

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       RESILIENCE & GRACEFUL DEGRADATION                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  CIRCUIT BREAKER PATTERN (per dependency)                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   CLOSED (Normal) ──5 failures──▶ OPEN (Fail Fast) ──30s timeout──▶    │   │
│  │        ▲                                                              │   │
│  │        │              3 successes                                      │   │
│  │        └──────────── HALF-OPEN ───────────────────────────────────────┘   │
│  │                                                                         │   │
│  │   Applied to: Redis, Kafka, ClickHouse, Model Inference, Behavioral     │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  GRACEFUL DEGRADATION FALLBACKS                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Component Failure          │ Fallback Behavior                          │   │
│  │  ────────────────────────── │ ────────────────────────────────────────  │   │
│  │  Redis (velocity)           │ In-memory zeros, heuristic scoring       │   │
│  │  Redis (seen entities)      │ Treat all as "new" (novelty = 1.0)       │   │
│  │  Redis (profiles)           │ MockFeatureStore → zeros/None            │   │
│  │  Behavioral Engine          │ Cold-start hierarchy (Merchant→Tenant)   │   │
│  │  ML Model missing/corrupt   │ Heuristic score + Rules only             │   │
│  │  Kafka (audit)              │ Local buffer, flush on recovery          │   │
│  │  ClickHouse (drift/rollup)  │ Queue locally, batch write on recovery   │   │
│  │  GNN Scorer                 │ Skip async, no GNN features              │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  MODEL HOT RELOAD (Zero-Downtime)                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  FileSystemEventHandler (watchdog) detects model file changes           │   │
│  │       │                                                                 │   │
│  │       ▼                                                                 │   │
│  │  Load into _staging dict (tenant_id → model)                           │   │
│  │       │                                                                 │   │
│  │       ▼                                                                 │   │
│  │  Validate: feature hash matches, model loads, warmup inference          │   │
│  │       │                                                                 │   │
│  │       ▼                                                                 │   │
│  │  Atomic swap: with RLock, _active = _staging; _staging = {}            │   │
│  │       │                                                                 │   │
│  │       ▼                                                                 │   │
│  │  Emit structured log: model_reloaded tenant=bank_ng_gtb                │   │
│  │  version=simple_123 duration_ms=42                                     │   │
│  │       │                                                                 │   │
│  │       ▼                                                                 │   │
│  │  Zero failed requests during reload (< 500ms)                          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.7 Security Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY LAYERS                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TRANSPORT                                                                      │
│  • TLS 1.2+ enforced in Docker Compose                                        │
│  • mTLS for service-to-service (future)                                        │
│                                                                                 │
│  AUTHENTICATION & AUTHORIZATION                                                 │
│  • API Key (X-API-Key header) per tenant                                       │
│  • Bearer Token (JWT) for admin endpoints                                      │
│  • Rate limiting: 1000 req/min per tenant (slowapi)                           │
│                                                                                 │
│  INPUT VALIDATION                                                               │
│  • Pydantic strict mode on all request models                                  │
│  • Max payload size: 50KB                                                      │
│  • Tokenized identifiers only (no raw PAN/BVN/SSN)                             │
│                                                                                 │
│  SECRETS MANAGEMENT                                                             │
│  • Docker secrets / Vault for production                                       │
│  • No plaintext credentials in config or images                                │
│  • Rotated API keys, DB passwords, Kafka SASL                                  │
│                                                                                 │
│  AUDIT LOGGING                                                                  │
│  • Immutable structured JSON to Kafka                                         │
│  • Every decision: transaction_id, tenant, score, decision, rules, model       │
│  • Label events: transaction_id, label, source, confidence, timestamp          │
│  • Model reloads, config changes, admin actions                                │
│  • Retention: 7 years (compliance)                                             │
│                                                                                 │
│  DATA PRIVACY                                                                   │
│  • Tokenization at ingestion (account_id, device_id, merchant_id)              │
│  • Field-level encryption for PII in PostgreSQL                                │
│  • GDPR: Right to erasure via /v1/admin/data/delete/{account_id}              │
│  • PCI DSS: No card data stored, only BIN/last4                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Prerequisites

```
fraudtrap/
  api/                    # FastAPI app and HTTP endpoints
  behavior/               # Behavioral Intelligence Layer (Phase 7)
    profiles/             # Online behavioral profiles (customer, merchant, device, beneficiary, instrument)
    feature_generation/   # Velocity, trust, similarity, novelty, heuristic features
    storage/              # RedisFeatureStore, InMemoryFeatureStore, MockFeatureStore
    services/             # BehaviorEngine orchestration
    integration.py        # ML pipeline integration (Cold Start → Semi-Supervised → Supervised)
    tests/                # Unit tests (17 passing)
  config/                 # Environment-driven settings (Pydantic)
  dashboard/              # Streamlit app and pages
  docker/                 # Dockerfiles and compose file
  features/               # Feature engineering (Redis pipelines)
  ingestion/              # Kafka producer/consumer, schemas
  models/                 # Cold-start, semi-supervised, supervised, GNN
  monitoring/             # Drift, metrics, alerts, rollup
  scoring/                # Orchestrator, rules, simple model, validation
  scripts/                # Data gen, trainer, simulator, retrain
  training/               # Dataset builder, pipeline
  artifacts/              # Generated data + model artifacts
  tests/                  # Unit, integration, chaos tests
  docs/
    runbooks/             # Operational runbooks
```

---

## 4. Prerequisites

- Docker Desktop (Linux engine)
- Python 3.11+ (for local scripts outside Docker)
- PowerShell (Windows) or bash (Linux/macOS)

Docker Desktop must be running before starting the stack.

---

## 5. Start The System

From the project root:

```powershell
cd C:\Users\Tommie-YV\Downloads\fraudtrap
docker compose -f docker\docker-compose.yml up -d api dashboard live_simulator
```

Check status:

```powershell
docker compose -f docker\docker-compose.yml ps
```

Expected core services:

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

- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`
- MLflow: `http://localhost:5000`

---

## 6. Generate Sample Training Data

```powershell
docker compose -f docker\docker-compose.yml run --rm dashboard python scripts/generate_sample_data.py --rows 50000 --fraud-rate 0.025
```

Outputs:

```
artifacts/data/bank_ng_gtb/features.parquet
artifacts/data/bank_ke_equity/features.parquet
artifacts/data/fintech_za_yoco/features.parquet
```

Each file contains labeled feature rows (`label=0` legitimate, `label=1` fraud).

---

## 7. Simulate Live Traffic

The `live_simulator` service continuously posts synthetic transactions to `/v1/score`.

View logs:

```powershell
docker compose -f docker\docker-compose.yml logs live_simulator --tail=50
```

Example log line:

```
bank_ng_gtb sim_... label=1 score=0.580 decision=REVIEW latency=13.16ms
```

The simulator also sends some labels to `/v1/labels`, exercising the label pipeline.

---

## 7. Train Lightweight Tenant Models

The fast serving path uses a lightweight NumPy logistic regression per tenant.

First generate sample data (if not done):

```powershell
docker compose -f docker\docker-compose.yml run --rm dashboard python scripts/generate_sample_data.py --rows 50000 --fraud-rate 0.025
```

Train models for all demo tenants:

```powershell
docker compose -f docker\docker-compose.yml run --rm dashboard python -m scripts.train_simple_model --all-tenants
```

Output:

```
artifacts/models/bank_ng_gtb/simple_model.pkl
artifacts/models/bank_ke_equity/simple_model.pkl
artifacts/models/fintech_za_yoco/simple_model.pkl
```

Rebuild/recreate the API so it loads the new artifacts:

```powershell
docker compose -f docker\docker-compose.yml build api
docker compose -f docker\docker-compose.yml up -d api live_simulator dashboard
```

Verify a tenant model loaded:

```powershell
Invoke-RestMethod http://localhost:8000/v1/phase/bank_ng_gtb
```

Expected:

```json
{
  "current_phase": "SUPERVISED",
  "loaded_models": { "simple_model": true }
}
```

---

## 9. Test The API Manually

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

Read recent live scores:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/v1/recent?limit=10"
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

## Data Contract For Banks

### Minimum Scoring Payload

```json
{
  "tenant_id": "bank_ng_gtb",
  "account_id": "tok_acct_123",
  "amount": 45000,
  "currency": "NGN",
  "timestamp": "2026-07-08T14:23:11Z",
  "transaction_type": "PAYMENT",
  "channel": "MOBILE"
}
```

### Recommended Extra Fields

```
transaction_id, session_id, merchant_id, merchant_category_code,
counterparty_account_id, device_id, device_type, ip_address_hash,
user_agent_hash, latitude, longitude, country_code,
typing_cadence_ms, session_duration_seconds, field_visit_count, extra_fields
```

**Privacy rule**: send tokenized/hashed identifiers only. No raw PANs, raw account numbers, BVNs, SSNs, or plaintext passwords.

---

## How Scoring Works Today

1. Redis-backed feature assembly (velocity, device, geo, behavioral)
2. Rules engine (Tier 1, <1ms) — hard blocks + soft boosts
3. Tenant-specific model selection (SimpleFraudModel / ColdStart / SemiSupervised / Supervised)
3. Policy floor: `risk_score = max(model_score, heuristic_score)` — never trust model alone
4. Decision thresholds (configurable in `config/settings.py`):
   - `score < 0.40` → APPROVE
   - `0.40 ≤ score < 0.85` → REVIEW
   - `score ≥ 0.85` → BLOCK
5. Redis recent-score cache for dashboard
6. Audit emit to Kafka (scored txn + decision)

Decision thresholds in `config/settings.py`:

```text
score_review_low = 0.40
score_block_threshold = 0.85
```

---

## Model Artifacts & Serving

Trained serving path:

1. `scripts/generate_sample_data.py` → labeled feature rows (Parquet)
2. `scripts/train_simple_model.py` → trains one model per tenant
3. Each model saved as `artifacts/models/<tenant>/simple_model.pkl`
4. `api/main.py` calls `registry.load_from_disk(settings.model_dir)` at startup
5. `scoring/orchestrator.py` selects model for `txn.tenant_id`
6. `/v1/phase/{tenant}` reports loaded model status

If a model file is missing for a tenant, `/v1/phase/{tenant}` shows `"simple_model": false` and the scorer falls back to conservative heuristic scoring.

---

## Common Commands

Rebuild changed services:

```powershell
docker compose -f docker\docker-compose.yml build api dashboard live_simulator
docker compose -f docker\docker-compose.yml up -d api dashboard live_simulator
```

Stop the simulator:

```powershell
docker compose -f docker\docker-compose.yml stop live_simulator
```

Restart simulator:

```powershell
docker compose -f docker\docker-compose.yml restart live_simulator
```

Show API logs:

```powershell
docker compose -f docker\docker-compose.yml logs api --tail=100
```

Show dashboard logs:

```powershell
docker compose -f docker\docker-compose.yml logs dashboard --tail=100
```

Compile-check Python:

```powershell
python -m compileall api dashboard features ingestion mlops models scoring scripts training
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Docker cannot connect | Start Docker Desktop, wait for Linux engine |
| Dashboard shows no live data | `docker compose logs live_simulator --tail=50` + `Invoke-RestMethod http://localhost:8000/v1/recent?limit=5` |
| Dashboard crashes on missing columns | Rebuild dashboard image (live loader backfills columns) |
| Generated data missing | `docker compose run --rm dashboard python scripts/generate_sample_data.py --rows 50000 --fraud-rate 0.025` |
| API returns `model_version: unloaded` | Train models, rebuild/recreate API, check `/v1/phase/{tenant}` |

---

## Recommended Build Order For A Beginner

1. Start Docker Desktop
2. Start API, dashboard, simulator
3. Open dashboard (`http://localhost:8501`)
4. Check `/v1/recent`
5. Generate sample Parquet data
6. Train simple tenant models
7. Rebuild/recreate API
8. Check `/v1/phase/bank_ng_gtb`
9. Read API schema at `/docs`
9. Send one manual transaction
10. Send one manual label
10. Review logs

---

*FraudTrap — Real-time fraud detection for banks and fintechs.*