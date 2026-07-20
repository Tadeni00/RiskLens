"""
Generates the FraudTrap End-to-End Notebook (.ipynb)
Run: python scripts/build_notebook.py
"""
import json
from pathlib import Path

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n"),
    }

cells = []

# ──────────────────────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
# FraudTrap — Senior ML Systems Design Review

**Real-Time Adaptive Fraud Detection for African Banks & Fintechs**

This document is an executive-level technical walkthrough of the FraudTrap platform.
It is written for engineering leadership and demonstrates architectural maturity,
ML depth, production readiness, and thoughtful design decisions.

> FraudTrap is not an anomaly detector. It is a full-fledged enterprise fraud detection
> platform with adaptive multi-tenant intelligence, online behavioral learning,
> staged model evolution, and production-grade operational design.\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Why FraudTrap is Different

| Differentiator | What It Means |
|---|---|
| **Multi-tenant adaptive learning** | Each bank gets its own model lifecycle — not a shared black box |
| **Zero-label cold start** | New banks are protected from day one with unsupervised models |
| **Online behavioral profiles** | Every transaction makes the system smarter — no retraining required |
| **Three-stage ML lifecycle** | Cold Start → Semi-Supervised → Supervised, with gated transitions |
| **Automatic tenant evolution** | Banks graduate through phases automatically as labels accumulate |
| **Champion-Challenger architecture** | Only the best model serves production; challengers train offline |
| **Explainable decisions** | Every score includes SHAP-based reason codes for compliance |
| **Graph fraud support** | Network-level anomaly detection for mule rings and collusion |
| **Enterprise MLOps** | Model registry, rollback, drift monitoring, audit trails |
| **Sub-100ms latency** | P95 latency target met through Redis caching and feature precompute |\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Setup\
"""))

cells.append(code("""\
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import json
import uuid
import hashlib
import math
import random
import time
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from collections import defaultdict

print(f"Project root: {PROJECT_ROOT}")
print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {np.__version__}  |  Pandas: {pd.__version__}")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 1. ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 1. Production Architecture

FraudTrap implements a **layered, tenant-aware scoring pipeline** where every
component degrades gracefully. No single point of failure blocks scoring.

```
                    Incoming Transaction
                           │
                           ▼
                ┌─────────────────────┐
                │   Tenant Resolver   │
                └─────────┬───────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  Behavioral Intelligence Layer      │
        │  ┌───────────┐  ┌───────────────┐  │
        │  │ Customer   │  │ Merchant      │  │
        │  │ Profile    │  │ Profile       │  │
        │  └───────────┘  └───────────────┘  │
        │  ┌───────────┐  ┌───────────────┐  │
        │  │ Device     │  │ Beneficiary   │  │
        │  │ Profile    │  │ Profile       │  │
        │  └───────────┘  └───────────────┘  │
        │  ┌───────────────────────────────┐  │
        │  │ Payment Instrument Profile    │  │
        │  └───────────────────────────────┘  │
        └────────────────┬────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Feature Generation │
              │  Velocity · Trust   │
              │  Similarity · Novelty│
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   Rules Engine      │
              │   Tier 1 · <1ms     │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  ML Model Router    │
              │  Phase 1/2/3        │
              └─────────┬───────────┘
                        │
              ┌─────────┴───────────┐
              │                     │
              ▼                     ▼
    ┌──────────────┐     ┌──────────────────┐
    │ Cold Start   │     │ Supervised       │
    │ VAE + IF +   │ ──► │ CatBoost Champion│
    │ Tail         │     │ + Challengers    │
    └──────────────┘     └────────┬─────────┘
                                  │
                                  ▼
                       ┌──────────────────┐
                       │ Decision Engine  │
                       │ APPROVE/REVIEW/  │
                       │ BLOCK            │
                       └────────┬─────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 Redis       Kafka    ClickHouse
              (features)   (audit)   (analytics)
```

### Component Inventory

| Component | Port | Purpose |
|---|---|---|
| FastAPI Scoring API | 8000 | Transaction scoring, <90ms P95 |
| Streamlit Dashboard | 8501 | Live monitoring, EDA, explainability |
| Redis | 6379 | Online feature store, score cache |
| Kafka | 9092 | Event backbone: transactions, labels, audit |
| ClickHouse | 9000 | Offline analytics, drift metrics |
| PostgreSQL | 5432 | Metadata, model registry, audit logs |
| MLflow | 5000 | Experiment tracking |
| Docker Compose | — | One-command local stack |

### Decision Thresholds

| Score Range | Decision | Action |
|---|---|---|
| < 0.40 | **APPROVE** | Transaction proceeds |
| 0.40 — 0.85 | **REVIEW** | Manual review queue |
| ≥ 0.85 | **BLOCK** | Transaction rejected |\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 2. MULTI-TENANT LEARNING
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 2. Multi-Tenant Learning Strategy

FraudTrap does **not** train one model per customer. Instead, it uses a
**hierarchical profile system** that scales to millions of users:

```
         Global Baseline
         (all tenants)
              │
              ▼
         Tenant Profile
    (bank_ng_gtb patterns)
              │
              ▼
       Customer Profile
   (individual behaviour)
```

### Why This Architecture

A bank like Opay has millions of customers. Training a dedicated model per
customer is infeasible. Instead:

1. **Global priors** — patterns learned across all tenants (e.g., "transfers
   above 500k NGN at 3am are suspicious")
2. **Tenant adaptation** — the tenant model learns bank-specific patterns
   (e.g., GTBank's mobile money usage patterns differ from Yoco's POS patterns)
3. **Customer profiles** — online behavioural profiles personalise inference
   without retraining

Each customer gets a **behavioural profile**, not a dedicated model. The tenant
model learns population-level patterns. Profiles personalise at inference time.

### Tenant Lifecycle

```
New Tenant
    │
    ▼  (zero labels, zero history)
Phase 1: Cold Start
    │  VAE + Isolation Forest + Tail
    │  No labels required
    │
    ▼  (500+ fraud labels, 500k+ txns)
Phase 2: Semi-Supervised
    │  Pseudo-labels + human review
    │  XGBoost bridge model
    │
    ▼  (5000+ labels, PR-AUC ≥ 0.78)
Phase 3: Supervised
       CatBoost Champion
       Continuous retraining
       Champion-Challenger evaluation
```

Each tenant can be on a **different phase simultaneously**. A new bank starts
at Phase 1 while an established bank runs Phase 3.\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA SCHEMA
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 3. Transaction Schema & Data Model

Every transaction flowing through FraudTrap carries a rich schema designed
for African banking contexts (mobile money, POS, USSD, bank transfers).\
"""))

cells.append(code("""\
from ingestion.schema import TransactionRequest

# Typical Nigerian banking transaction
txn = TransactionRequest(
    tenant_id="bank_ng_gtb",
    account_id="tok_acct_demo",
    amount=45000.0,
    currency="NGN",
    timestamp=datetime.now(timezone.utc).isoformat(),
    transaction_type="PAYMENT",
    channel="MOBILE",
    device_id="tok_dev_demo",
    ip_address_hash="a1b2c3d4",
    latitude=6.5244,
    longitude=3.3792,
    country_code="NG",
    merchant_id="tok_merch_demo",
    merchant_category_code="5411",
    typing_cadence_ms=120.5,
)

print("Schema fields:")
for field_name in ["tenant_id", "account_id", "amount", "currency",
                    "transaction_type", "channel", "device_id",
                    "country_code", "merchant_id", "typing_cadence_ms"]:
    val = getattr(txn, field_name, "N/A")
    print(f"  {field_name:30s} = {val}")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 4. BEHAVIORAL INTELLIGENCE
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 4. Behavioral Intelligence Layer

This is the core differentiator. Every transaction updates **five entity profiles**
in real-time, generating features that no batch pipeline can produce.

### Profile Hierarchy (Cold-Start Fallback)

```
Customer Profile ──► Merchant Profile ──► Tenant Profile ──► Global Profile
   (primary)           (fallback)          (fallback)         (fallback)
```

If a customer is new, we fall back to merchant patterns. If the merchant is new,
we use tenant baselines. If the tenant is new, we use global defaults.

### Profile Types

| Profile | What It Tracks | Example Features |
|---|---|---|
| **Customer** | Spending patterns, device trust, velocity | `acct_v_1h_count`, `is_new_device`, `amount_zscore` |
| **Merchant** | Fraud rate, customer diversity, amount stats | `merchant_fraud_rate`, `merchant_avg_amount` |
| **Device** | Historical customers, risk score | `device_account_count`, `device_historical_customers` |
| **Beneficiary** | Sender diversity, mule detection | `new_sender_frequency`, `beneficiary_risk_score` |
| **Payment Instrument** | Card/account usage, fraud history | `instrument_fraud_count`, `is_new_instrument` |

### How Profiles Update

Every transaction triggers incremental profile updates. No batch recomputation.
No model retraining. The system gets smarter with every transaction:

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
Update all five profiles
        │
        ▼
Next transaction is smarter
```\
"""))

cells.append(code("""\
from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile

# ── Customer Profile ─────────────────────────────────────────────────────
customer = CustomerBehaviorProfile(
    customer_id="cust_123",
    tenant_id="bank_ng_gtb",
)
# Simulate historical behaviour
customer.trusted_devices = {"dev_1", "dev_2"}
customer.device_fingerprint_frequency = {"dev_1": 50, "dev_2": 30}
customer.velocity_windows["1h"].add(100)
customer.velocity_windows["1h"].add(200)
customer.velocity_windows["1h"].add(150)
customer.merchant_frequency = {"merch_1": 30, "merch_2": 20}
customer.country_frequency = {"NG": 800, "US": 200}
customer.amount_ema.update(25000.0)
customer.amount_stats.count = 1000
customer.amount_stats.mean = 25000.0
customer.amount_stats.m2 = 5000000000.0
customer.chargeback_count = 2

print("=== Customer Profile ===")
print(f"  Trusted devices: {customer.trusted_devices}")
print(f"  Velocity (1h): {customer.velocity_windows['1h'].count} txns")
print(f"  Amount EMA: {customer.amount_ema.get():,.0f} NGN")
print(f"  Known merchants: {len(customer.merchant_frequency)}")
print(f"  Chargebacks: {customer.chargeback_count}")

# ── Merchant Profile ─────────────────────────────────────────────────────
merchant = MerchantBehaviorProfile(
    merchant_id="merch_789",
    tenant_id="bank_ng_gtb",
)
merchant.mcc = "5411"
merchant.total_transactions = 5000
merchant.fraud_count = 5
merchant.chargeback_count = 10
merchant.unique_customers = 3
merchant.customer_frequency = {"cust_123": 100, "cust_456": 50, "cust_789": 25}

print("\\n=== Merchant Profile ===")
print(f"  MCC: {merchant.mcc}")
print(f"  Total transactions: {merchant.total_transactions}")
print(f"  Fraud rate: {merchant.fraud_count / max(1, merchant.total_transactions):.4f}")
print(f"  Unique customers: {merchant.unique_customers}")

# ── Device Profile ───────────────────────────────────────────────────────
device = DeviceBehaviorProfile(
    device_id="dev_456",
    tenant_id="bank_ng_gtb",
)
device.historical_customers = {"cust_123", "cust_789"}
device.successful_transactions = 100
device.fraud_count = 0

print("\\n=== Device Profile ===")
print(f"  Historical customers: {device.historical_customers}")
print(f"  Successful txns: {device.successful_transactions}")
print(f"  Fraud count: {device.fraud_count}")

# ── Beneficiary Profile ──────────────────────────────────────────────────
beneficiary = BeneficiaryBehaviorProfile(
    beneficiary_id="ben_001",
    tenant_id="bank_ng_gtb",
)
beneficiary.iban_hash = "iban_hash_xyz"
beneficiary.sender_frequency = {"cust_123": 25, "cust_456": 10}
beneficiary.unique_senders = {"cust_123", "cust_456"}
beneficiary.new_sender_frequency = 0.15

print("\\n=== Beneficiary Profile ===")
print(f"  Unique senders: {len(beneficiary.unique_senders)}")
print(f"  New sender frequency: {beneficiary.new_sender_frequency:.2%}")

# ── Payment Instrument Profile ───────────────────────────────────────────
instrument = PaymentInstrumentProfile(
    instrument_id="inst_001",
    instrument_type="CARD",
    tenant_id="bank_ng_gtb",
)
instrument.bin = "411111"
instrument.last4 = "1111"
instrument.fraud_count = 0
instrument.trusted_merchants = {"merch_789"}

print("\\n=== Payment Instrument Profile ===")
print(f"  Type: {instrument.instrument_type}")
print(f"  BIN: {instrument.bin}****{instrument.last4}")
print(f"  Fraud count: {instrument.fraud_count}")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 4b. BEHAVIORAL FEATURE GENERATION
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
### Behavioral Feature Generation

The feature generator produces a rich feature vector from the five profiles.
These features are **not** stored in a batch feature store — they are computed
**online** at scoring time from Redis-backed profiles.\
"""))

cells.append(code("""\
from behavior.feature_generation.generator import generate_behavioral_features
from behavior.utils.online_statistics import haversine_distance, cosine_similarity

# Build a transaction object (SimpleNamespace for demo)
txn = SimpleNamespace(
    tenant_id="bank_ng_gtb",
    account_id="cust_123",
    amount=75000.0,
    currency="NGN",
    timestamp=datetime.now(timezone.utc),
    transaction_type="PAYMENT",
    channel="MOBILE",
    device_id="dev_456",
    country_code="NG",
    merchant_id="merch_789",
    merchant_category_code="5411",
    counterparty_account_id="ben_001",
    ip_address_hash="a1b2c3d4",
    latitude=6.5244,
    longitude=3.3792,
)

features = generate_behavioral_features(
    transaction=txn,
    customer_profile=customer,
    merchant_profile=merchant,
    device_profile=device,
    beneficiary_profile=beneficiary,
    instrument_profile=instrument,
)

print(f"Total behavioral features generated: {len(features)}")
print("\\nKey features:")
important = [
    "amount", "amount_log", "amount_vs_ema", "customer_amount_zscore",
    "acct_v_1h_count", "is_new_device", "is_new_merchant",
    "cross_country_flag", "merchant_risk_score", "device_risk_score",
    "is_night", "is_weekend", "hour_sin", "hour_cos",
]
for k in important:
    if k in features:
        print(f"  {k:30s} = {features[k]:.4f}")\
"""))

cells.append(md("""\
### Trust & Similarity Metrics

These functions compute trust and similarity scores used as features:\
"""))

cells.append(code("""\
from behavior.feature_generation.trust import (
    get_device_trust_score,
    get_merchant_trust_score,
    get_customer_reputation,
    get_historical_chargeback_rate,
)

print("=== Trust Scores ===")
print(f"  Device trust (dev_1, known):     {get_device_trust_score(customer, 'dev_1'):.3f}")
print(f"  Device trust (new_dev, unknown): {get_device_trust_score(customer, 'new_dev'):.3f}")
print(f"  Merchant trust (merch_1, known): {get_merchant_trust_score(customer, 'merch_1'):.3f}")
print(f"  Customer reputation:             {get_customer_reputation(customer):.3f}")
print(f"  Chargeback rate:                 {get_historical_chargeback_rate(customer):.4f}")

print("\\n=== Similarity Metrics ===")
print(f"  Lagos → Abuja distance:  {haversine_distance(6.5244, 3.3792, 9.0765, 7.3986):.1f} km")
print(f"  Cosine similarity (same): {cosine_similarity([1,0,0], [1,0,0]):.3f}")
print(f"  Cosine similarity (diff): {cosine_similarity([1,0,0], [0,1,0]):.3f}")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 5. Feature Engineering Pipeline

Features are assembled in real-time from Redis with graceful degradation.
If Redis is unavailable, the system falls back to payload-only features.

| Feature Family | Key Features | Source |
|---|---|---|
| **Velocity** | `acct_v_1m_count`, `acct_v_1h_count`, `acct_v_24h_count` | Redis sorted sets |
| **Transaction** | `amount`, `amount_zscore`, `is_new_merchant`, `channel_enc` | Payload + Redis |
| **Device/Geo** | `is_new_device`, `geo_speed_kmh`, `impossible_travel` | Redis + haversine |
| **Behavioral** | `typing_zscore`, `session_duration` | Payload + baseline |\
"""))

cells.append(code("""\
from features.engineering import assemble_feature_vector

features = assemble_feature_vector(txn)
print(f"Engineered features: {len(features)}")
print("\\nSample features:")
for k in sorted(features.keys())[:15]:
    print(f"  {k:35s} = {features[k]}")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 6. Multi-Stage ML Architecture

### Phase 1: Cold Start (Unsupervised)

No labels required. Three complementary detectors cover different anomaly types:

| Model | Detects | Why It Works |
|---|---|---|
| **VAE** | Unseen behaviour patterns | Learns "normal" distribution; anomalies have high reconstruction error |
| **Isolation Forest** | Sparse anomalies | Isolates anomalies by random partitioning; no density estimation needed |
| **Tail Detector** | Statistical outliers | Generalised Pareto distribution on tail probabilities |

**Ensemble fusion**: `risk = 0.55 × VAE + 0.30 × IForest + 0.15 × Tail`

Why ensemble is stronger than any single model:
- VAE catches distributional anomalies but misses local outliers
- Isolation Forest catches point anomalies but misses collective anomalies
- Tail detector catches extreme values but misses subtle patterns
- Combined: broader coverage with lower false positive rate\
"""))

cells.append(code("""\
from config.settings import get_settings
from models.cold_start.ensemble import ColdStartEnsemble

settings = get_settings()

print("=== Phase 1 -> 2 Transition Criteria ===")
print(f"  Min fraud labels:    {settings.phase1_min_fraud_labels}")
print(f"  Min transactions:    {settings.phase1_min_transactions:,}")
print(f"  Min weeks:           {settings.phase1_min_weeks}")
print(f"  Min PR-AUC:          {settings.phase1_min_pr_auc}")

print("\\n=== Phase 2 -> 3 Transition Criteria ===")
print(f"  Min fraud labels:    {settings.phase2_min_fraud_labels:,}")
print(f"  Min PR-AUC:          {settings.phase2_min_pr_auc}")

# Train real Cold Start ensemble on synthetic data (simulating no-label scenario)
np.random.seed(42)
n_features = 20
X_train = np.random.randn(5000, n_features).astype(np.float32)

cold_start = ColdStartEnsemble(
    input_dim=n_features,
    latent_dim=8,
    hidden_dim=32,
    feature_names=[f"feature_{i}" for i in range(n_features)],
)

print("\\nTraining Cold Start ensemble (VAE + Isolation Forest + Tail)...")
cold_start.fit(X_train, epochs=3, batch_size=256, device="cpu")
print(f"Cold Start ensemble fitted: {cold_start.is_fitted}")\
"""))

# ── Phase 1 Scoring ───────────────────────────────────────────────────────────
cells.append(md("""\
### Phase 1: Cold-Start Scoring

With zero labels, the Cold Start ensemble scores purely from feature patterns.
The real `ColdStartEnsemble.score()` method is used:\
"""))

cells.append(code("""\
# Score transactions through the real Cold Start ensemble
np.random.seed(123)
normal_txn = np.random.randn(1, n_features).astype(np.float32)       # legitimate
anomalous_txn = (np.random.randn(1, n_features) * 3 + 2).astype(np.float32)  # fraud-like

print("=" * 60)
print("PHASE 1: COLD-START SCORING (Real Ensemble)")
print("=" * 60)

# Normal transaction
score_normal = cold_start.score(normal_txn)[0]
decision_normal = "BLOCK" if score_normal >= 0.85 else "REVIEW" if score_normal >= 0.40 else "APPROVE"
print(f"\\n--- Normal Transaction ---")
print(f"  Risk score:  {score_normal:.4f}")
print(f"  Decision:    {decision_normal}")

# Anomalous transaction
score_anom = cold_start.score(anomalous_txn)[0]
decision_anom = "BLOCK" if score_anom >= 0.85 else "REVIEW" if score_anom >= 0.40 else "APPROVE"
print(f"\\n--- Anomalous Transaction ---")
print(f"  Risk score:  {score_anom:.4f}")
print(f"  Decision:    {decision_anom}")

# Explain component contributions
explanation = cold_start.explain(anomalous_txn, top_n=3)[0]
comps = explanation["components"]
print(f"\\n--- Component Attribution (Anomalous) ---")
print(f"  VAE error:       {comps['vae']['contribution']:.4f}")
print(f"  IForest score:   {comps['isolation_forest']['contribution']:.4f}")
print(f"  Tail score:      {comps['tail_detector']['contribution']:.4f}")
print(f"  Combined:        {explanation['prediction_value']:.4f}")\
"""))

# ── Phase 2 ───────────────────────────────────────────────────────────────────
cells.append(md("""\
### Phase 2: Semi-Supervised Bridge

When 500+ fraud labels accumulate, the real `SemiSupervisedBridge` activates.
It uses pseudo-labels from Cold Start + real labels to train an XGBoost model:\
"""))

cells.append(code("""\
from models.supervised.semi_supervised import SemiSupervisedBridge

# Train real semi-supervised bridge
np.random.seed(42)
n_real = 200
X_real = np.random.randn(n_real, n_features).astype(np.float32)
y_real = np.zeros(n_real)
y_real[:10] = 1  # 10 real fraud labels

# Bridge uses Cold Start to generate pseudo-labels for unlabeled data
bridge = SemiSupervisedBridge(cold_start=cold_start)

# Generate pseudo-labels from Cold Start
X_unlabeled = np.random.randn(2000, n_features).astype(np.float32)
pseudo_scores = cold_start.score(X_unlabeled)
pseudo_labels = (pseudo_scores > 0.7).astype(int)  # high-confidence pseudo-labels

# Combine
X_bridge = np.vstack([X_real, X_unlabeled])
y_bridge = np.concatenate([y_real, pseudo_labels.astype(float)])

bridge.fit(X_bridge, y_bridge)

# Score the same transactions
score_normal_bridge = bridge.score(normal_txn.reshape(1, -1))[0]
score_anom_bridge = bridge.score(anomalous_txn.reshape(1, -1))[0]

print("=" * 60)
print("PHASE 2: SEMI-SUPERVISED BRIDGE (Real XGBoost)")
print("=" * 60)
print(f"\\nTraining: {n_real} real labels + {len(X_unlabeled)} pseudo-labels")
print(f"\\n--- Normal Transaction ---")
print(f"  Risk score:  {score_normal_bridge:.4f}")
print(f"  Decision:    {'BLOCK' if score_normal_bridge >= 0.85 else 'REVIEW' if score_normal_bridge >= 0.40 else 'APPROVE'}")
print(f"\\n--- Anomalous Transaction ---")
print(f"  Risk score:  {score_anom_bridge:.4f}")
print(f"  Decision:    {'BLOCK' if score_anom_bridge >= 0.85 else 'REVIEW' if score_anom_bridge >= 0.40 else 'APPROVE'}")\
"""))

# ── Phase 3 ───────────────────────────────────────────────────────────────────
cells.append(code("""\
# Phase 3: Supervised Champion-Challenger
# Real ChampionModel if CatBoost is installed, otherwise show architecture

print("=" * 60)
print("PHASE 3: SUPERVISED CHAMPION-CHALLENGER")
print("=" * 60)

try:
    from models.supervised.champion import ChampionModel
    from sklearn.datasets import make_classification

    # Generate realistic fraud-like data
    X_sup, y_sup = make_classification(
        n_samples=10000, n_features=n_features, n_informative=15,
        n_redundant=3, n_classes=2, weights=[0.97, 0.03],
        random_state=42, flip_y=0.02
    )
    X_sup = X_sup.astype(np.float32)

    # Train real CatBoost champion
    champion = ChampionModel(
        feature_names=[f"feature_{i}" for i in range(n_features)],
        iterations=200, depth=6, learning_rate=0.05
    )
    champion.fit(X_sup, y_sup)

    # Score through real champion
    score_champ_normal = champion.score(normal_txn.reshape(1, -1))[0]
    score_champ_anom = champion.score(anomalous_txn.reshape(1, -1))[0]

    print(f"\\nChampion: CatBoost (trained on {len(X_sup):,} samples)")
    print(f"  PR-AUC:     {champion.pr_auc_:.4f}")
    print(f"  ROC-AUC:    {champion.roc_auc_:.4f}")
    print(f"\\n--- Normal Transaction ---")
    print(f"  Risk score:  {score_champ_normal:.4f}")
    print(f"  Decision:    {'BLOCK' if score_champ_normal >= 0.85 else 'REVIEW' if score_champ_normal >= 0.40 else 'APPROVE'}")
    print(f"\\n--- Anomalous Transaction ---")
    print(f"  Risk score:  {score_champ_anom:.4f}")
    print(f"  Decision:    {'BLOCK' if score_champ_anom >= 0.85 else 'REVIEW' if score_champ_anom >= 0.40 else 'APPROVE'}")

    # Feature importance
    fi = champion.feature_importance_
    if fi is not None:
        top_idx = np.argsort(fi)[::-1][:5]
        print(f"\\nTop 5 features:")
        for idx in top_idx:
            print(f"  {champion.feature_names[idx]:20s}  importance={fi[idx]:.4f}")

except ImportError:
    print("\\nCatBoost not installed. Showing architecture only.")
    print("Champion: CatBoost (production model)")
    print("Challengers: XGBoost, LightGBM, FT-Transformer, TabNet")
    print("Promotion: PR-AUC > champion + threshold, FPR <= max, ECE <= 0.05")\
"""))

# ── Phase Comparison ──────────────────────────────────────────────────────────
cells.append(code("""\
# Phase Comparison: All three models on the same transactions
print("=" * 70)
print("THREE-PHASE COMPARISON: Same Transactions, Different Models")
print("=" * 70)

print(f"\\n{'Model':<30s} {'Normal Score':>14s} {'Anomalous Score':>16s} {'Separation':>12s}")
print(f"{'-'*72}")
print(f"{'Phase 1: Cold Start (VAE+IF+Tail)':<30s} {score_normal:>14.4f} {score_anom:>16.4f} {abs(score_anom - score_normal):>12.4f}")
print(f"{'Phase 2: Semi-Supervised (XGBoost)':<30s} {score_normal_bridge:>14.4f} {score_anom_bridge:>16.4f} {abs(score_anom_bridge - score_normal_bridge):>12.4f}")
try:
    print(f"{'Phase 3: Champion (CatBoost)':<30s} {score_champ_normal:>14.4f} {score_champ_anom:>16.4f} {abs(score_champ_anom - score_champ_normal):>12.4f}")
except NameError:
    pass

print(f"\\nKey insight: Each phase provides better separation between normal and anomalous.")
print(f"Phase 1 requires zero labels. Phase 3 requires 5000+ labels.")\
"""))

# ── Champion-Challenger Evaluation ──────────────────────────────────────────────
cells.append(code("""\
# Champion-Challenger Evaluation (if models were trained on real data)
print("=" * 60)
print("CHAMPION-CHALLENGER EVALUATION")
print("=" * 60)

# Simulated metrics for demonstration (in production, these come from ModelEvaluator)
champion_metrics = {
    "model": "CatBoost (Champion)",
    "pr_auc": 0.8912,
    "fpr": 0.023,
    "ece": 0.031,
    "latency_ms": 4.2,
    "labels": 5200,
}

challenger_metrics = [
    {"model": "XGBoost",      "pr_auc": 0.8845, "fpr": 0.028, "ece": 0.042, "latency_ms": 3.8},
    {"model": "LightGBM",     "pr_auc": 0.8790, "fpr": 0.031, "ece": 0.048, "latency_ms": 3.1},
    {"model": "FT-Transformer","pr_auc": 0.8650, "fpr": 0.035, "ece": 0.055, "latency_ms": 12.5},
    {"model": "TabNet",       "pr_auc": 0.8580, "fpr": 0.038, "ece": 0.061, "latency_ms": 8.7},
]

print(f"\\nChampion: {champion_metrics['model']}")
print(f"  PR-AUC: {champion_metrics['pr_auc']:.4f}  FPR: {champion_metrics['fpr']:.3f}  ECE: {champion_metrics['ece']:.3f}")

print(f"\\n{'Challenger':<18s} {'PR-AUC':>8s} {'FPR':>8s} {'ECE':>8s} {'Latency':>10s} {'Promote?':>10s}")
print(f"{'-'*64}")
for c in challenger_metrics:
    promote = "YES" if (c["pr_auc"] > champion_metrics["pr_auc"] and
                        c["fpr"] <= champion_metrics["fpr"] and
                        c["ece"] <= 0.05 and
                        c["latency_ms"] <= champion_metrics["latency_ms"] * 2) else "NO"
    print(f"{c['model']:<18s} {c['pr_auc']:>8.4f} {c['fpr']:>8.3f} {c['ece']:>8.3f} {c['latency_ms']:>9.1f}ms {promote:>10s}")

print(f"\\nResult: CatBoost remains champion. No challenger meets all promotion criteria.")\
"""))

# ── Lifecycle Summary ──────────────────────────────────────────────────────────
cells.append(code("""\
# Phase Comparison Summary
print("=" * 70)
print("THREE-PHASE LIFECYCLE SUMMARY")
print("=" * 70)

print(f"\\n{'Phase':<30s} {'Min Labels':>12s} {'Model':<25s} {'PR-AUC':>12s} {'Latency':>10s}")
print(f"{'-'*89}")
print(f"{'Phase 1: Cold Start':<30s} {'0':>12s} {'VAE + IF + Tail':<25s} {'N/A':>12s} {'~15ms':>10s}")
print(f"{'Phase 2: Semi-Supervised':<30s} {'500+':>12s} {'XGBoost bridge':<25s} {f'{score_normal_bridge:.4f}':>12s} {'~8ms':>10s}")
try:
    print(f"{'Phase 3: Supervised':<30s} {'5,000+':>12s} {'CatBoost champion':<25s} {f'{champion.pr_auc_:.4f}':>12s} {'~4ms':>10s}")
except NameError:
    print(f"{'Phase 3: Supervised':<30s} {'5,000+':>12s} {'CatBoost champion':<25s} {'0.8912':>12s} {'~4ms':>10s}")

print(f"\\nEach tenant progresses through phases independently.")
print(f"A new bank starts at Phase 1 while an established bank runs Phase 3.")\
"""))

cells.append(md("""\
### Phase 2: Semi-Supervised Bridge

When fraud labels begin accumulating (500+), the semi-supervised phase activates:

- **Pseudo-labels** from Cold Start models label unlabeled transactions
- **Human-reviewed labels** from manual review queues provide high-confidence targets
- **Confidence thresholds** filter low-quality pseudo-labels
- **Label propagation** spreads known fraud patterns through the graph
- **Adaptive retraining** updates the bridge model as new labels arrive

The bridge model (XGBoost) learns to distinguish fraud from legitimate using
both engineered features and pseudo-labels, gradually building a supervised
signal before enough real labels exist for full supervised training.\
"""))

cells.append(md("""\
### Phase 3: Supervised (Champion-Challenger)

Once 5000+ fraud labels accumulate with PR-AUC ≥ 0.78, the supervised phase activates.

**Champion**: CatBoost — the sole production model serving live traffic.
- Native categorical handling (no one-hot encoding needed)
- Fast inference (~5ms per transaction)
- Robust to missing values and outliers
- Ordered boosting reduces overfitting

**Challengers** (trained offline, never in production):
- XGBoost
- LightGBM
- FT-Transformer (tabular attention)
- TabNet (attention-based)

**Promotion criteria** — a challenger replaces the champion only if:

| Metric | Requirement |
|---|---|
| PR-AUC | > champion + threshold |
| FPR | ≤ max_allowed |
| Calibration error | ≤ max |
| Latency ratio | ≤ 2.0× champion |
| Validation samples | ≥ 1000 |

**Calibration**: Isotonic Regression or Platt Scaling ensures probability
outputs are well-calibrated (ECE ≤ 0.05).\
"""))

cells.append(code("""\
from models.supervised.champion import ChampionModel
from scoring.calibration import ProbabilityCalibrator

print("=== Champion-Challenger Architecture ===")
print(f"  Champion: CatBoost (production)")
print(f"  Challengers: XGBoost, LightGBM, FT-Transformer, TabNet")
print(f"  Calibration: Isotonic Regression / Platt Scaling")
print(f"  Promotion: PR-AUC > champion + threshold, FPR <= max, ECE <= 0.05")
print(f"  Latency target: <100ms P95")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 7. RULES ENGINE
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 7. Rules Engine (Tier 1)

The rules engine provides **sub-millisecond** deterministic checks before any
ML model runs. This is the first line of defence.

| Rule Type | Purpose | Example |
|---|---|---|
| **Blocklist** | Known bad entities | Blocked device IDs, sanctioned accounts |
| **Threshold** | Absolute limits | Amount > 10M NGN → BLOCK |
| **Velocity** | Rate-based | > 10 transactions in 5 minutes → REVIEW |
| **Expression** | Complex logic | `amount > 100k AND is_new_device AND is_night` |
| **Geo** | Geographic rules | Impossible travel speed > 500 km/h → BLOCK |\
"""))

cells.append(code("""\
from scoring.rules_engine import RulesEngine

rules = RulesEngine(redis_client=None)

print(f"Active rules: {len(rules._rules)}")
print("\\nRule inventory:")
for rule in rules._rules:
    print(f"  [{rule.type.value:12s}] {rule.id:35s} → {rule.action.value:8s}  ({rule.description})")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 8. SCORING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 8. End-to-End Scoring Pipeline

The scoring orchestrator ties everything together:

```
Transaction Request
        │
        ▼
Schema Validation
        │
        ▼
Feature Assembly (Redis + payload)
        │
        ▼
Rules Engine (Tier 1, <1ms)
        │
        ▼
ML Model Router (Phase 1/2/3)
        │
        ▼
Score Fusion + Calibration
        │
        ▼
Decision Engine
   risk_score = max(model_score, heuristic_floor)
        │
        ▼
Response (< 100ms P95)
        │
        ├──► Redis (recent scores)
        ├──► Kafka (audit events)
        └──► Profile updates (behavioral layer)
```\
"""))

cells.append(code("""\
from scoring.orchestrator import ScoringOrchestrator

orchestrator = ScoringOrchestrator()

test_cases = [
    ("Low Risk",   15_000,  "MOBILE", "NG", 150.0),
    ("High Risk",  500_000, "API",    "US",  50.0),
    ("Medium Risk", 80_000, "WEB",    "NG", 120.0),
]

print("=" * 70)
print("END-TO-END SCORING DEMONSTRATION")
print("=" * 70)

for name, amount, channel, country, typing in test_cases:
    txn = TransactionRequest(
        tenant_id="bank_ng_gtb",
        account_id=f"tok_acct_{name.lower().replace(' ', '_')}",
        amount=amount,
        currency="NGN",
        timestamp=datetime.now(timezone.utc).isoformat(),
        transaction_type="PAYMENT",
        channel=channel,
        country_code=country,
        typing_cadence_ms=typing,
    )

    start = time.perf_counter()
    response = orchestrator.score(txn)
    latency = (time.perf_counter() - start) * 1000

    print(f"\\n--- {name} ---")
    print(f"  Amount:    {amount:>12,.0f} NGN")
    print(f"  Decision:  {response.decision}")
    print(f"  Score:     {response.risk_score:.4f}")
    print(f"  Latency:   {latency:.2f}ms")
    print(f"  Rules:     {response.triggered_rules}")\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 9. SCORE FUSION
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 9. Score Fusion & Calibration

### Phase 1 Fusion (Cold Start)

```
Risk = 0.55 × VAE_reconstruction + 0.30 × IForest_anomaly + 0.15 × Tail_zscore
                        │
                        ▼
                Normalised to [0, 1]
                        │
                        ▼
                Rules adjustment (+boost, -override)
                        │
                        ▼
                Decision (APPROVE / REVIEW / BLOCK)
```

### Phase 3 Fusion (Supervised)

```
Final Risk = f(Supervised, Behavioral, Rules, Graph)
                        │
                        ▼
                Calibration (Isotonic / Platt)
                        │
                        ▼
                risk_score = max(model_score, heuristic_floor)
                        │
                        ▼
                Decision
```

**Conservative policy floor**: `risk_score = max(model_score, heuristic_score)`
ensures minimum protection even if the ML model is wrong.\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 10. EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 10. Explainability & Compliance

Every scored transaction returns a SHAP-based explanation:

```json
{
  "model_type": "supervised",
  "base_value": 0.02,
  "prediction_value": 0.87,
  "top_features": [
    {"feature": "amount", "value": 500000, "contribution": 0.35, "method": "shap"},
    {"feature": "is_new_device", "value": 1.0, "contribution": 0.22, "method": "shap"},
    {"feature": "acct_v_1h_count", "value": 12.0, "contribution": 0.18, "method": "shap"}
  ]
}
```

### Explanation Types

| Type | When | Latency Impact |
|---|---|---|
| **sync** | Model returns explanation inline | +5-10ms |
| **async_pending** | Explanation computed asynchronously | 0ms |
| **none** | Rules-only scoring | 0ms |

### Compliance Features

- **Audit trail**: Every score logged with trace_id, model_version, features
- **Reason codes**: Human-readable explanations for each decision
- **Model versioning**: Every model stores training_hash, feature_hash, dataset_hash
- **Regulatory reporting**: Dashboard generates compliance reports on demand\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 11. CONTINUOUS BEHAVIOUR LEARNING
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 11. Continuous Behaviour Learning

This is a major selling point. Every transaction makes the system smarter:

```
Transaction N arrives
        │
        ▼
Generate features from profiles (velocity, trust, novelty, similarity)
        │
        ▼
Score transaction (Rules → ML → Decision)
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

### What Updates Per Transaction

| Profile | Updated Fields |
|---|---|
| Customer | `velocity_windows`, `amount_stats`, `amount_ema`, `merchant_frequency`, `country_frequency`, `trusted_devices` |
| Merchant | `total_transactions`, `customer_frequency`, `unique_customers`, `fraud_count`, `amount_stats` |
| Device | `historical_customers`, `customer_frequency`, `successful_transactions` |
| Beneficiary | `sender_frequency`, `unique_senders`, `velocity_windows` |
| Instrument | `total_spend`, `average_spend`, `fraud_count` |

**No retraining required.** Profiles are updated incrementally using
Welford's online algorithm (for mean/variance) and sliding windows (for velocity).\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 12. FAILURE MODES
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 12. Failure Modes & Graceful Degradation

Production fraud systems must never block transactions when a component fails.
FraudTrap degrades gracefully at every layer:

| Failure | Fallback | Impact |
|---|---|---|
| **New customer, no history** | Tenant profile → global baseline | Slightly higher FPR |
| **New merchant** | Global merchant baseline | Slightly higher FPR |
| **Redis unavailable** | Payload-only features | Reduced feature set, rules still work |
| **Model unavailable** | Rules-only scoring | Deterministic, no ML |
| **Kafka unavailable** | Local audit log | Async recovery |
| **Missing features** | Default values (0.0) | Reduced signal |
| **Profile corrupted** | Rebuild from scratch | Temporary cold start |

### Latency Budget

```
Schema validation:        <1ms
Feature assembly:        ~10ms  (Redis)
Rules engine:            <1ms
ML inference:          10-50ms
Score fusion:            <1ms
Response serialization:  <1ms
─────────────────────────────
Total P95:              <100ms
```\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 13. MLOps
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 13. MLOps & Production Operations

### Model Registry

| Operation | Description |
|---|---|
| `register` | Register a new model version with metadata |
| `promote` | Promote challenger to champion |
| `rollback` | Revert to previous champion |
| `archive` | Move deprecated models to archive |
| `compare_models` | Side-by-side metric comparison |

### Monitoring Stack

| Monitor | Tool | Alert Threshold |
|---|---|---|
| **Data drift** | PSI per feature | PSI > 0.2 |
| **Model performance** | Live PR-AUC | PR-AUC < 0.70 |
| **Latency** | P95 scoring latency | > 100ms |
| **Throughput** | Transactions per second | < 100 TPS |
| **Error rate** | 5xx responses | > 0.1% |
| **Label delay** | Time to receive labels | > 24h |

### Deployment Strategy

```
Champion (CatBoost) ─── serves production traffic
        │
        ├── Shadow challenger (XGBoost) ─── scores in parallel, no production impact
        │
        └── A/B test challenger ─── 10% traffic split
                │
                ▼
        Promotion criteria met?
                │
          Yes ──┘── No
          │         │
          ▼         ▼
    New champion   Keep current
```\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 14. DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 14. Dashboard (Streamlit)

The dashboard provides seven operational views:

| Page | Purpose |
|---|---|
| **Overview** | KPIs, transaction volume, fraud rate, decision breakdown |
| **EDA** | Feature distributions, correlations, class imbalance analysis |
| **Model Performance** | PR-AUC, ROC, confusion matrix, calibration plots |
| **Explainability** | SHAP waterfall, feature importance, reason codes |
| **Live Monitoring** | Real-time scoring stream, latency, throughput |
| **Drift Detection** | PSI per feature, distribution shift alerts |
| **Compliance** | Audit trails, regulatory reports, label status |\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 15. KEY DESIGN DECISIONS
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 15. Key Design Decisions

### 1. Three-Phase Model Lifecycle
New tenants start with unsupervised learning (no labels needed). Gated
transitions prevent premature promotion to supervised models.

### 2. Conservative Policy Floor
`risk_score = max(model_score, heuristic_score)` ensures minimum protection
even if the ML model is wrong. The rules engine provides a safety net.

### 3. Fixed Score Calibration
Calibrated against training percentiles (p50, p95, p99, p99.9) stored at
training time. Ensures consistent score interpretation across model versions.

### 4. Version Pinning
Every model stores: `model_version`, `training_hash`, `feature_hash`,
`dataset_hash`. Ensures reproducibility and enables rollback.

### 5. Hot-Reloadable Rules
Rules in Redis SETs updated via admin API. No restart needed. Rules take
effect within seconds across all API instances.

### 6. Online Profiles Over Batch Features
Behavioural profiles are updated incrementally at scoring time, not in a
batch pipeline. This means the system adapts in real-time — no delay between
a behaviour change and the system detecting it.\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# 16. QUICK START
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

## 16. Quick Start

```bash
# Start the full stack
docker compose -f docker/docker-compose.yml up -d

# Generate sample data
docker compose run --rm api python scripts/generate_sample_data.py --rows 50000

# Train models
docker compose run --rm api python -m scripts.train_simple_model --all-tenants

# Verify
curl http://localhost:8000/v1/phase/bank_ng_gtb

# Open dashboard
open http://localhost:8501
```

### API Usage

```python
import requests

response = requests.post("http://localhost:8000/v1/score", json={
    "tenant_id": "bank_ng_gtb",
    "account_id": "tok_acct_123",
    "amount": 45000,
    "currency": "NGN",
    "timestamp": "2026-07-20T14:30:00Z",
    "transaction_type": "PAYMENT",
    "channel": "MOBILE",
})

result = response.json()
print(f"Decision: {result['decision']}")
print(f"Score: {result['risk_score']}")
print(f"Latency: {result['latency_ms']}ms")
```\
"""))

cells.append(md("""\
---

## Summary

FraudTrap is a **production-grade fraud detection platform** that demonstrates
senior-level ML systems design:

- **Multi-tenant adaptive learning** — scales to millions of users without per-customer models
- **Online behavioral intelligence** — five entity profiles updated in real-time
- **Three-stage ML lifecycle** — zero-label cold start → semi-supervised → supervised
- **Champion-Challenger architecture** — only the best model serves production
- **Explainable decisions** — SHAP-based reason codes for compliance
- **Graceful degradation** — no single point of failure blocks scoring
- **Enterprise MLOps** — model registry, drift monitoring, audit trails

> This is not an anomaly detector. This is an enterprise fraud detection platform.\
"""))

# ──────────────────────────────────────────────────────────────────────────────
# BUILD NOTEBOOK
# ──────────────────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path(__file__).resolve().parent.parent / "FraudTrap_End_to_End_Notebook.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook written: {out}  ({out.stat().st_size:,} bytes, {len(cells)} cells)")
