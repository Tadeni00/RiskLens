<div align="center">

# FraudTrap

### Production-Grade · Real-Time · Multi-Tenant Fraud Detection Platform

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-FFC107?style=for-the-badge&logoColor=black)](https://catboost.ai)
[![Redis](https://img.shields.io/badge/Redis-7.0+-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![Kafka](https://img.shields.io/badge/Kafka-3.5+-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-23.8+-FFCC00?style=for-the-badge&logoColor=black)](https://clickhouse.com)
[![Docker](https://img.shields.io/badge/Docker-24.0+-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-0199E1?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

An end-to-end fraud detection platform — not a model, not a notebook, not a prototype.

A production system that scores transactions in real-time, adapts to each tenant's fraud patterns,
explains every decision, and evolves through three stages of ML maturity as labeled data accumulates.

</div>

---

## What FraudTrap Is

FraudTrap is a **multi-tenant, real-time fraud detection platform** built for:

| Sector | Examples |
|--------|----------|
| Banks | GTBank, Access Bank, Zenith Bank |
| Fintechs | OPay, Kuda, Yoco |
| Payment Processors | Flutterwave, Paystack |
| Lending Companies | Carbon, FairMoney |
| Wallets | PalmPay, OPay Wallet |
| Digital Commerce | Jumia Pay, KongaPay |

It performs:

- **Online inference** — sub-100ms transaction scoring
- **Behavioural profiling** — real-time entity profiles (customer, merchant, device, beneficiary, payment instrument)
- **Fraud detection** — three-phase ML lifecycle from zero labels to fully supervised
- **Drift monitoring** — PSI, KL divergence, concept drift detection
- **Model lifecycle management** — champion-challenger, promotion, rollback
- **Explainability** — SHAP attributions, counterfactual explanations, analyst-friendly reports

It supports:

- **Multi-tenancy** — each bank gets its own model lifecycle, thresholds, and profiles
- **Real-time scoring** — Redis-backed feature serving, <100ms P95 latency
- **Continuous learning** — online profile updates, semi-supervised label propagation

> **FraudTrap is not a machine learning model.**
> It is an end-to-end fraud detection platform with production infrastructure, monitoring, and operational tooling.

---

## Architecture

```mermaid
graph TD
    TXN[Transaction] --> FE[Feature Store<br/>Redis]
    FE --> BE[Behaviour Engine<br/>5 Entity Profiles]
    BE --> RE[Rules Engine<br/>Tier 1 · &lt;1ms]
    RE --> MR[ML Model Router<br/>Phase Detection]
    
    MR --> CS[Cold Start Layer<br/>VAE + IF + Tail]
    MR --> SS[Semi-Supervised Layer<br/>TabPFN]
    MR --> SV[Supervised Layer<br/>CatBoost Champion]
    
    SV --> CE{Confidence<br/>Check}
    CE -->|High| DEC[Decision Engine]
    CE -->|Low| FT[FT-Transformer<br/>Specialist]
    FT --> MF[Meta Fusion]
    MF --> DEC
    
    CS --> DEC
    SS --> DEC
    
    DEC --> APPROVE[APPROVE]
    DEC --> REVIEW[REVIEW]
    DEC --> BLOCK[BLOCK]
    
    DEC --> REDIS[(Redis<br/>Score Cache)]
    DEC --> KAFKA[Kafka<br/>Audit Events]
    DEC --> CH[(ClickHouse<br/>Analytics)]
    
    KAFKA --> LABELS[Label Worker]
    LABELS --> SS
    LABELS --> SV
    
    CH --> DRIFT[Drift Monitor]
    DRIFT --> RETRAIN[Retrain Trigger]
    RETRAIN --> SV
```

### Component Inventory

| Layer | Component | Technology | Purpose |
|-------|-----------|------------|---------|
| **API** | Scoring API | FastAPI | Transaction scoring, <90ms P95 |
| **Features** | Feature Store | Redis | Online feature serving |
| **Rules** | Rules Engine | Python | Sub-millisecond deterministic checks |
| **Behaviour** | Profile Engine | Redis + Python | Real-time entity profiling |
| **ML Phase 1** | Cold Start | VAE + IF + Tail | Zero-label anomaly detection |
| **ML Phase 2** | Semi-Supervised | TabPFN | Foundation model for limited labels |
| **ML Phase 3** | Champion | CatBoost | Production fraud classifier |
| **ML Phase 3** | Specialist | FT-Transformer | Low-confidence edge cases |
| **ML Phase 3** | Meta Fusion | Logistic Regression | Combines champion + specialist |
| **Explainability** | SHAP + Counterfactual | SHAP + FAISS | Decision explanations |
| **Monitoring** | Drift Detection | PSI + KL Divergence | Model stability monitoring |
| **Storage** | Analytics | ClickHouse | Offline analytics, drift metrics |
| **Storage** | Metadata | PostgreSQL | Model registry, audit logs |
| **MLOps** | Experiment Tracking | MLflow | Training metadata, versioning |
| **Dashboard** | Operations Console | Streamlit | 9-page enterprise dashboard |

---

## The Three-Phase ML Lifecycle

FraudTrap implements a **gated progression** from unsupervised to supervised learning. Each tenant advances independently based on accumulated labels.

```mermaid
graph LR
    A["New Tenant<br/>0 Labels"] -->|"Phase 1"| B["Cold Start<br/>VAE + IF + Tail"]
    B -->|"100+ Labels"| C["Semi-Supervised<br/>TabPFN"]
    C -->|"5000+ Labels"| D["Supervised<br/>CatBoost + FT-Transformer"]

    style A fill:#D69E2E,color:#000
    style B fill:#D69E2E,color:#000
    style C fill:#2D6CDF,color:#fff
    style D fill:#17A673,color:#fff
```

| Phase | Name | Models | Labels Required | Latency |
|-------|------|--------|-----------------|---------|
| 1 | Cold Start | VAE + Isolation Forest + Empirical Tail Detector | 0 | ~15ms |
| 2 | Semi-Supervised | TabPFN (Prior Labs) | 100+ | ~10ms |
| 3 | Supervised | CatBoost Champion + FT-Transformer Specialist | 5,000+ | ~4ms |

---

## Layer 1: Cold Start Intelligence

**Purpose**: Protect new tenants from day one — no fraud labels required.

When a bank joins FraudTrap, it has zero historical fraud data. Traditional supervised models cannot score a single transaction. Cold Start solves this with an ensemble of three complementary anomaly detectors.

### Models

| Model | What It Detects | Why It Exists |
|-------|-----------------|---------------|
| **VAE** (Variational Autoencoder) | Distributional anomalies | Learns the shape of "normal" transactions; anomalies have high reconstruction error |
| **Isolation Forest** | Point anomalies | Isolates outliers by random partitioning; no density estimation needed |
| **Empirical Tail Detector** | Statistical extremes | Generalised Pareto distribution on tail probabilities |

### Score Fusion

```
risk_score = 0.55 × VAE_reconstruction_error
           + 0.30 × IsolationForest_anomaly_score
           + 0.15 × TailDetector_zscore
```

### Why This Works

- **VAE** catches distributional anomalies (unusual spending patterns) but misses local outliers
- **Isolation Forest** catches point anomalies (single suspicious transactions) but misses collective fraud
- **Tail Detector** catches extreme values (massive transfers) but misses subtle patterns
- **Combined**: broader coverage with lower false positive rate

### Capabilities

- ✅ Works with zero labels
- ✅ Protects new tenants from day one
- ✅ Protects new customers within existing tenants
- ✅ Component-level attribution (explainable anomaly scores)
- ✅ Automatic calibration from training distribution

---

## Layer 2: Semi-Supervised Intelligence

**Purpose**: Bridge the gap between zero-label cold start and fully supervised learning.

As fraud labels accumulate from analyst reviews, chargebacks, and customer reports, the semi-supervised layer activates. It uses **TabPFN** (Tabular Prior-data Fitted Network) — a pretrained tabular foundation model by Prior Labs that excels on small-to-medium labelled datasets.

### Architecture

```
Transaction Features
        │
        ▼
┌─────────────────────────┐
│ TabPFN Foundation Model  │  (pretrained transformer)
│  Distribution Embedder   │
│  Row/Cross-row Attention │
│  In-context Learning     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Calibrated Probabilities │  (fraud probability + uncertainty)
└─────────────────────────┘
```

### Label Sources

| Source | Quality | Weight |
|--------|---------|--------|
| Analyst manual review | High | 1.0 |
| Chargeback confirmation | High | 0.9 |
| Customer report | Medium | 0.7 |
| Cold Start pseudo-labels | Low-Medium | 0.5 |

### How It Works

1. **Pseudo-label generation** — Cold Start scores unlabeled transactions
2. **Confidence filtering** — Only high-confidence pseudo-labels are used (threshold: 0.8)
3. **In-context fitting** — TabPFN stores the labelled dataset for in-context learning
4. **Calibrated prediction** — TabPFN produces well-calibrated probabilities via transformer attention
5. **Uncertainty estimation** — Prediction entropy quantifies model uncertainty per transaction

### Why TabPFN Instead of XGBoost

| Aspect | XGBoost Bridge | TabPFN |
|--------|---------------|--------|
| Label efficiency | Needs 500+ labels | Works with 100+ |
| Training | Gradient boosting | In-context learning (no training) |
| Uncertainty | Not naturally available | Entropy-based uncertainty |
| Calibration | Requires post-hoc calibration | Naturally well-calibrated |
| Adaptability | Retrain from scratch | Add new data to context |

---

## Layer 3: Supervised Intelligence

**Purpose**: Maximum accuracy when sufficient labeled data exists.

Once 5,000+ fraud labels accumulate with PR-AUC ≥ 0.78, the supervised layer activates. This layer implements a **confidence-aware routing** strategy.

### Champion Model: CatBoost

CatBoost is the **sole production model** serving live traffic.

**Why CatBoost:**

| Property | Benefit |
|----------|---------|
| Native categorical handling | No target encoding leakage; handles `channel`, `country_code`, `merchant_category_code` natively |
| Ordered boosting | Reduces overfitting on imbalanced fraud data |
| Built-in class imbalance handling | `auto_class_weights: Balanced` parameter |
| Fast inference | ~4ms per transaction |
| GPU support | Training acceleration |
| Built-in feature importance | No need for permutation importance |
| Robust to missing values | Handles missing features gracefully |

### Confidence-Aware Routing

```
Transaction
    │
    ▼
┌──────────────────┐
│ CatBoost Champion│  (fast, handles categoricals natively)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│Confidence Estim. │  (conformal prediction + distance-based)
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 High Conf  Low Conf
    │         │
    │         ▼
    │   ┌──────────────────┐
    │   │ FT-Transformer   │  (tabular attention specialist)
    │   └────────┬─────────┘
    │            │
    │            ▼
    │   ┌──────────────────┐
    │   │  Meta Fusion     │  (logistic regression combiner)
    │   └────────┬─────────┘
    │            │
    └────┬───────┘
         │
         ▼
  Final Probability
```

### Selective Inference Strategy

| Component | Invocation Rate | Latency | Purpose |
|-----------|----------------|---------|---------|
| CatBoost | 100% of transactions | ~4ms | Primary fraud detection |
| FT-Transformer | ~10-15% of transactions | ~15ms | Edge cases where CatBoost lacks confidence |
| Meta Fusion | When FT-Transformer is invoked | <1ms | Combines both predictions |

**Key design decision**: FT-Transformer is **not** a challenger. It is a **production specialist** that handles difficult transactions where CatBoost's confidence is low. This balances accuracy against latency.

### Meta Fusion

When FT-Transformer is invoked, the final prediction is:

```
P(fraud) = σ(w₁ × P(catboost) + w₂ × P(ft_transformer) + bias)
```

The fusion weights are learned via logistic regression on a held-out validation set.

### Offline Challengers

| Model | Role | Status |
|-------|------|--------|
| LightGBM | Offline benchmark | Never in production |
| XGBoost | Offline benchmark | Never in production |

These models are trained offline and evaluated continuously against the champion. They are **never used for production inference**. If a challenger consistently outperforms the champion, it is recommended for promotion through the model registry.

---

## Behaviour Engine

**Purpose**: Real-time entity profiling that makes the system smarter with every transaction.

### Entity Profiles

| Profile | What It Tracks | Key Features |
|---------|----------------|--------------|
| **Customer** | Spending patterns, device trust, velocity | `acct_v_1h_count`, `is_new_device`, `amount_zscore` |
| **Merchant** | Fraud rate, customer diversity, amount stats | `merchant_fraud_rate`, `merchant_avg_amount` |
| **Device** | Historical customers, risk score | `device_account_count`, `device_historical_customers` |
| **Beneficiary** | Sender diversity, mule detection | `new_sender_frequency`, `beneficiary_risk_score` |
| **Payment Instrument** | Card/account usage, fraud history | `instrument_fraud_count`, `is_new_instrument` |

### Profile Hierarchy (Cold-Start Fallback)

```
Customer Profile ──► Merchant Profile ──► Tenant Profile ──► Global Profile
   (primary)           (fallback)          (fallback)         (fallback)
```

If a customer is new, we fall back to merchant patterns. If the merchant is new, we use tenant baselines. If the tenant is new, we use global defaults.

### How Profiles Update

Every transaction triggers **incremental profile updates**. No batch recomputation. No model retraining.

```
Transaction arrives
        │
        ▼
Generate features from profiles
        │
        ▼
Score transaction
        │
        ▼
Update all five profiles:
  • Customer: velocity windows, amount stats, device trust
  • Merchant: customer diversity, fraud rate, amount patterns
  • Device: historical customers, risk score
  • Beneficiary: sender diversity, velocity
  • Instrument: usage patterns, fraud history
        │
        ▼
Transaction N+1 benefits from updated profiles
```

### Feature Families

| Family | Key Features | Source |
|--------|--------------|--------|
| **Velocity** | `acct_v_1m_count`, `acct_v_1h_count`, `acct_v_24h_count` | Redis sorted sets |
| **Transaction** | `amount`, `amount_zscore`, `is_new_merchant`, `channel_enc` | Payload + Redis |
| **Device/Geo** | `is_new_device`, `geo_speed_kmh`, `impossible_travel` | Redis + haversine |
| **Behavioral** | `typing_zscore`, `session_duration` | Payload + baseline |

---

## Multi-Tenant Design

**Purpose**: Each bank gets its own model lifecycle without cross-tenant interference.

### Tenant Isolation

Each tenant maintains:

- ✅ Independent models (per-phase)
- ✅ Independent thresholds (configurable per tenant)
- ✅ Independent drift monitoring
- ✅ Independent feature statistics
- ✅ Independent behavioural profiles

### Tenant Maturity Routing

```
New Tenant
    │
    ▼  (zero labels, zero history)
Phase 1: Cold Start
    │  VAE + Isolation Forest + Tail
    │  No labels required
    │
    ▼  (100+ fraud labels)
Phase 2: Semi-Supervised
    │  TabPFN (Prior Labs foundation model)
    │  Pseudo-labels + confidence-aware routing
    │
    ▼  (5000+ labels, PR-AUC ≥ 0.78)
Phase 3: Supervised
       CatBoost Champion
       + FT-Transformer Specialist
       + Meta Fusion
       Champion-Challenger evaluation
```

Each tenant can be on a **different phase simultaneously**. A new bank starts at Phase 1 while an established bank runs Phase 3.

---

## Explainability

**Purpose**: Every decision includes a regulator-friendly explanation.

### Explanation Stack

```
┌─────────────────────────────────────────────────────────┐
│              ExplainabilityEngine                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ SHAP        │  │ Counterfactual│  │  Formatter   │  │
│  │ Explainer   │  │ Engine       │  │              │  │
│  │             │  │              │  │  Analyst-    │  │
│  │ TreeExplainer│ │  Nearest     │  │  friendly    │  │
│  │ (CatBoost)  │  │  Neighbor   │  │  natural     │  │
│  │             │  │  + DiCE     │  │  language    │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ SHAP Cache  │  │ Explanation  │  │  Monitoring  │  │
│  │ (LRU+TTL)   │  │ Cache        │  │  Latency,    │  │
│  │             │  │ (LRU+TTL)    │  │  cache hits  │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Explanation Components

| Component | Purpose | Latency Impact |
|-----------|---------|----------------|
| **SHAP Attributions** | Feature contribution scores | +5-10ms |
| **Counterfactual** | "What would need to change" | +10-20ms |
| **Nearest Neighbor** | Most similar past transactions | +2-5ms (FAISS) |
| **Formatted Report** | Analyst-friendly natural language | +1ms |
| **Confidence Info** | Model certainty and prediction intervals | +1ms |

### Example Output

```json
{
  "model_type": "supervised",
  "base_value": 0.02,
  "prediction_value": 0.87,
  "confidence": {"level": "high", "distance": 0.12},
  "top_features": [
    {"feature": "amount", "value": 500000, "contribution": 0.35},
    {"feature": "is_new_device", "value": 1.0, "contribution": 0.22},
    {"feature": "acct_v_1h_count", "value": 12.0, "contribution": 0.18}
  ],
  "counterfactual": {
    "nearest_neighbor_id": "txn_abc123",
    "distance": 0.15,
    "changed_features": [
      {"feature": "amount", "from": 500000, "to": 45000}
    ]
  },
  "formatted_report": "High risk: large amount (500k NGN) from new device."
}
```

### Regulatory Compliance

- **Audit trail**: Every score logged with `trace_id`, `model_version`, features
- **Reason codes**: Human-readable explanations for each decision
- **Model versioning**: Every model stores `training_hash`, `feature_hash`, `dataset_hash`
- **PII safety**: Sensitive features never logged in explanations

---

## Drift Detection

**Purpose**: Detect when fraud patterns change and the model needs retraining.

### Monitoring Stack

| Monitor | Metric | Alert Threshold |
|---------|--------|-----------------|
| **Data drift** | PSI per feature | PSI > 0.2 |
| **Concept drift** | Prediction distribution shift | KL divergence > 0.1 |
| **Model performance** | Live PR-AUC | PR-AUC < 0.70 |
| **Latency** | P95 scoring latency | > 100ms |
| **Throughput** | Transactions per second | < 100 TPS |
| **Error rate** | 5xx responses | > 0.1% |
| **Label delay** | Time to receive labels | > 24h |

### Drift Response Protocol

```
PSI > 0.1  →  Warning alert (logged, monitored)
PSI > 0.2  →  Automatic retraining triggered
PSI > 0.4  →  Emergency rollback to previous champion
PR-AUC < 0.70  →  Champion demoted, challenger evaluation
Latency > 100ms  →  Specialist invocation rate reduced
```

---

## Model Registry

**Purpose**: Version control, promotion, and rollback for production models.

### Operations

| Operation | Description |
|-----------|-------------|
| `register` | Register a new model version with metadata |
| `promote` | Promote challenger to champion |
| `rollback` | Revert to previous champion |
| `archive` | Move deprecated models to archive |
| `compare_models` | Side-by-side metric comparison |

### Metadata Tracked

- Model version (semver)
- Training date and duration
- Training dataset hash
- Feature hash
- Validation metrics (PR-AUC, ROC-AUC, F2, F1, Precision, Recall)
- Calibration error (ECE)
- Latency measurements
- Promotion/approval status

---

## Dashboard

A **9-page enterprise operations console** built with Streamlit.

> **Design philosophy**: The dashboard should not look like a Streamlit application. It should feel like a polished enterprise operations console that a major bank or fintech could deploy today.

### Pages

| Page | Purpose |
|------|---------|
| **Overview** | 8 KPIs, operational health, 24h timeline, decision distribution |
| **Risk Intelligence** | Geographic fraud map, hourly timeline, merchant/customer leaderboards |
| **Behavior Profiles** | 5 entity profiles, velocity analysis, feature importance |
| **Models** | Architecture diagram, champion metrics, model leaderboard, confusion matrix |
| **Explainability** | SHAP waterfall, counterfactual flow, similar transactions |
| **Drift Monitoring** | PSI bars, KL divergence, concept drift timeline |
| **Live Monitoring** | Real-time metrics, latency, throughput, infrastructure health |
| **Compliance** | Regulatory checklist, bias monitoring, audit trail |
| **Lifecycle** | Model timeline, phase progression, registry, promotion readiness |

### Design System

| Element | Value |
|---------|-------|
| Background | `#0B1320` primary, `#1B2537` cards |
| Accent | `#2D6CDF` (single accent color) |
| Font | Inter / IBM Plex Sans |
| Icons | Lucide |
| Charts | Plotly (14 chart types) |

---

## Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **API** | FastAPI | REST API with Swagger docs |
| **ML** | PyTorch | VAE, FT-Transformer |
| **ML** | TabPFN | Semi-supervised foundation model (Prior Labs) |
| **ML** | CatBoost | Production fraud classifier |
| **ML** | scikit-learn | Isolation Forest, calibration, metrics |
| **ML** | SHAP | Feature attributions |
| **Features** | Redis | Online feature store, score cache |
| **Streaming** | Apache Kafka | Transaction events, audit, labels |
| **Analytics** | ClickHouse | Offline analytics, drift metrics |
| **Database** | PostgreSQL | Metadata, model registry, labels |
| **MLOps** | MLflow | Experiment tracking |
| **Dashboard** | Streamlit | Operations console |
| **Monitoring** | Prometheus | System metrics |
| **Container** | Docker Compose | 11-service local stack |
| **Config** | Pydantic Settings | Environment-driven configuration |

---

## Repository Structure

```
fraudtrap/
├── api/                    # FastAPI app and HTTP endpoints
├── behavior/               # Behavioural Intelligence Layer
│   ├── profiles/           # Online behavioural profiles
│   ├── feature_generation/ # Velocity, trust, similarity, novelty
│   ├── storage/            # RedisFeatureStore, InMemoryFeatureStore
│   ├── services/           # BehaviorEngine orchestration
│   └── tests/              # Unit tests
├── config/                 # Environment-driven settings (Pydantic)
├── dashboard/              # Enterprise dashboard (9 pages)
│   ├── theme/              # Design system (colors, typography, CSS, icons)
│   ├── components/         # Reusable UI components
│   ├── pages/              # Dashboard pages
│   └── utils/              # Utilities
├── docker/                 # Dockerfiles and compose file
├── features/               # Feature engineering (Redis pipelines)
├── ingestion/              # Kafka producer/consumer, schemas
├── models/
│   ├── cold_start/         # VAE + Isolation Forest + Tail Detector
│   ├── semi_supervised/    # TabPFN (Prior Labs tabular foundation model)
│   ├── supervised/         # Champion, FT-Transformer, Meta Fusion, Challengers
│   └── explainability/     # SHAP, Counterfactual, Formatter, Cache
├── monitoring/             # Drift, metrics, alerts, rollup
├── scoring/                # Orchestrator, rules, calibration, model router
├── scripts/                # Data gen, training, simulation
├── tests/                  # Unit, integration, load tests
├── training/               # Dataset builder, pipeline
├── docs/
│   └── runbooks/           # Operational runbooks
├── .github/workflows/      # CI/CD pipelines
├── FraudTrap_End_to_End_Notebook.ipynb    # Architecture walkthrough
└── requirements.txt        # Core dependencies
```

---

## Production Features

- ✅ Real-time scoring (<100ms P95)
- ✅ Multi-tenant isolation
- ✅ Redis feature serving
- ✅ Kafka event streaming
- ✅ ClickHouse analytics
- ✅ Champion–challenger architecture
- ✅ Confidence-aware routing (CatBoost + FT-Transformer)
- ✅ Behaviour profiling (5 entity types)
- ✅ Cold-start detection (VAE + IF + Tail)
- ✅ Semi-supervised learning (TabPFN)
- ✅ Drift detection (PSI, KL divergence)
- ✅ Explainability (SHAP + Counterfactual)
- ✅ Probability calibration (Isotonic Regression)
- ✅ Model registry with versioning
- ✅ Champion–challenger evaluation
- ✅ Automated promotion/rollback
- ✅ Graceful degradation
- ✅ Audit trail logging
- ✅ Enterprise dashboard (9 pages)
- ✅ Docker Compose local stack (11 services)
- ✅ CI/CD pipelines (GitHub Actions)

---

## Roadmap

### Implemented

- ✅ Three-phase ML lifecycle (Cold Start → Semi-Supervised → Supervised)
- ✅ CatBoost champion with FT-Transformer specialist
- ✅ TabPFN semi-supervised learning
- ✅ Behavioural profiling (5 entity types)
- ✅ SHAP + counterfactual explainability
- ✅ Drift detection (PSI, KL divergence)
- ✅ Champion–challenger evaluation
- ✅ Enterprise dashboard (9 pages)
- ✅ Docker Compose stack
- ✅ CI/CD pipelines

### Planned

- 🔲 **Graph Neural Network** — mule ring detection and collusion networks
- 🔲 **Federated Learning** — cross-tenant pattern sharing (privacy-preserving)
- 🔲 **Active Learning** — intelligent label solicitation from analysts
- 🔲 **Online Learning** — model adaptation without full retraining
- 🔲 **Reinforcement Learning** — adaptive decision thresholds
- 🔲 **Multi-modal Features** — transaction + device + behavioral signals
- 🔲 **AutoML** — tenant-specific hyperparameter tuning
- 🔲 **Regulatory Sandbox** — automated compliance reporting

---

## Getting Started

### Prerequisites

- Docker Desktop (Linux engine)
- Python 3.11+ (for local scripts)

### Docker (Recommended)

```bash
# Start the full stack
docker compose -f docker/docker-compose.yml up -d

# Check status
docker compose -f docker/docker-compose.yml ps

# Open
# API:       http://localhost:8000
# API docs:  http://localhost:8000/docs
# Dashboard: http://localhost:8501
# MLflow:    http://localhost:5000
```

### Local Development

```bash
# Clone
git clone https://github.com/your-org/fraudtrap.git
cd fraudtrap

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py --rows 50000

# Train models
python -m scripts.train_simple_model --all-tenants

# Start API
uvicorn api.main:app --reload --port 8000

# Start dashboard
streamlit run dashboard/app.py
```

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENVIRONMENT` | `development` | Runtime environment |
| `API_URL` | `http://localhost:8000` | API endpoint |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `KAFKA_BOOTSTRAP` | `localhost:9092` | Kafka broker |
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse host |
| `CLICKHOUSE_PORT` | `9000` | ClickHouse port |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow endpoint |
| `MODEL_DIR` | `artifacts/models` | Model artifact path |

---

## API Examples

### Score a Transaction

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "bank_ng_gtb",
    "account_id": "tok_acct_123",
    "amount": 45000,
    "currency": "NGN",
    "timestamp": "2026-07-22T14:30:00Z",
    "transaction_type": "PAYMENT",
    "channel": "MOBILE",
    "device_id": "tok_dev_001",
    "country_code": "NG"
  }'
```

### Get Explanation

```bash
curl http://localhost:8000/v1/explain/{trace_id}
```

### Check Model Health

```bash
curl http://localhost:8000/v1/phase/bank_ng_gtb
```

### Get Drift Status

```bash
curl http://localhost:8000/v1/drift/bank_ng_gtb
```

### Get Behaviour Profile

```bash
curl http://localhost:8000/v1/profile/customer/{customer_id}
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Notebook](FraudTrap_End_to_End_Notebook.ipynb) | 16-section architecture walkthrough |
| [API Documentation](API_DOCUMENTATION_v2.md) | Full API reference |
| [OpenAPI Spec](openapi.yaml) | OpenAPI specification |
| [Postman Collection](FraudTrap_API.postman_collection.json) | 22 API requests |
| [Runbooks](docs/runbooks/) | Operational runbooks |
| [Study Guide](FraudTrap_Complete_Study_Guide.docx) | 11-chapter platform guide |

---

## Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| **P95 Latency** | < 100ms | ~90ms |
| **P99 Latency** | < 200ms | ~150ms |
| **Throughput** | > 100 TPS | 500+ TPS |
| **Availability** | 99.9% | 99.95% |
| **Cold Start Training** | < 5 minutes | ~2 minutes |
| **Champion Inference** | < 10ms | ~4ms |
| **Dashboard Refresh** | < 3 seconds | ~1.5 seconds |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check .

# Run type checking
mypy .
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Third-party dependencies and their licenses are listed in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

---

<div align="center">

**FraudTrap** — Production-grade fraud detection for African banks and fintechs.

Built with engineering rigor. Designed for scale. Ready for production.

</div>
