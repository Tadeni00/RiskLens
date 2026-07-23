# FraudTrap — Technical Documentation

## Enterprise Fraud Detection Platform

**Version:** 2.1.0
**Last Updated:** July 2026
**Classification:** Internal Engineering Documentation

---

## Table of Contents

1. [What FraudTrap Is](#1-what-fraudtrap-is)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Multi-Tenant Design](#4-multi-tenant-design)
5. [Behavioral Intelligence Layer](#5-behavioral-intelligence-layer)
6. [Feature Engineering](#6-feature-engineering)
7. [Rules Engine](#7-rules-engine)
8. [The Three-Phase ML Lifecycle](#8-the-three-phase-ml-lifecycle)
9. [Phase 1: Cold Start](#9-phase-1-cold-start)
10. [Phase 2: Semi-Supervised Learning](#10-phase-2-semi-supervised-learning)
11. [Phase 3: Champion-Challenger Supervised Learning](#11-phase-3-champion-challenger-supervised-learning)
12. [Dynamic Model Routing](#12-dynamic-model-routing)
13. [The Scoring Orchestrator](#13-the-scoring-orchestrator)
14. [Score Fusion and Decision Making](#14-score-fusion-and-decision-making)
15. [Explainability](#15-explainability)
16. [Online Learning and Profile Updates](#16-online-learning-and-profile-updates)
17. [Probability Calibration](#17-probability-calibration)
18. [Production Infrastructure](#18-production-infrastructure)
19. [Failure Modes and Graceful Degradation](#19-failure-modes-and-graceful-degradation)
20. [MLOps Pipeline](#20-mlops-pipeline)
21. [Security and Tenant Isolation](#21-security-and-tenant-isolation)
22. [API Reference](#22-api-reference)
23. [Deployment](#23-deployment)
24. [Design Decisions and Rationale](#24-design-decisions-and-rationale)

---

## 1. What FraudTrap Is

FraudTrap is a real-time fraud detection platform built for banks and fintech companies. It scores incoming financial transactions and decides whether each one should be approved, reviewed by a human, or blocked entirely.

The platform is designed to work across multiple banks simultaneously. Each bank (called a "tenant") gets its own learning pipeline, its own behavioral profiles, and its own model lifecycle — all running on shared infrastructure. A single FraudTrap deployment can protect Bank A with a mature supervised model while simultaneously protecting a brand-new fintech with zero fraud labels, using only unsupervised anomaly detection.

FraudTrap is not a single machine learning model. It is a complete system that includes behavioral profiling, real-time feature engineering, deterministic rules, multiple stages of machine learning, probability calibration, explainability, and full MLOps infrastructure. The goal is to detect fraud accurately, explain why each decision was made, adapt to new patterns in real-time, and operate reliably in production with strict latency requirements.

---

## 2. The Problem It Solves

Fraud detection has a fundamental cold-start problem. When a new bank joins your platform, you have zero fraud labels. You do not know which transactions are legitimate and which are fraudulent. Without labels, you cannot train a supervised model. Most ML fraud systems require thousands of labeled fraud cases before they become useful, which means a new bank is unprotected for weeks or months while labels accumulate from chargebacks and manual reviews.

FraudTrap solves this with a three-stage machine learning lifecycle. On day one, with zero labels, the system uses unsupervised anomaly detection (a Variational Autoencoder, Isolation Forest, and statistical tail detector) to flag unusual transaction patterns. As labels accumulate from chargebacks and human reviews, the system automatically transitions to semi-supervised learning, using pseudo-labels to expand the training set. Once enough confirmed labels exist, it deploys a full supervised CatBoost model with a Champion-Challenger architecture for continuous improvement.

This means every tenant is protected from their first transaction, and the system automatically evolves its detection strategy as more information becomes available.

Beyond cold start, FraudTrap addresses several other critical problems:

- **Scale**: A bank with millions of customers cannot have a dedicated model per customer. FraudTrap uses hierarchical profiling where one tenant model learns bank-wide patterns and individual customer profiles personalize inference.
- **Latency**: Fraud decisions must happen in under 90 milliseconds. FraudTrap achieves this through Redis-cached features, efficient model inference, and parallel processing.
- **Adaptability**: Fraud patterns change constantly. FraudTrap updates behavioral profiles on every transaction, so the system adapts in real-time without waiting for batch retraining.
- **Explainability**: Regulators and customers demand reasons for fraud decisions. FraudTrap provides SHAP explanations, rule contributions, and human-readable reason codes for every score.
- **Multi-tenancy**: Running multiple banks on one platform requires strict data isolation, per-tenant model lifecycle, and independent feature stores. FraudTrap handles all of this through tenant-scoped Redis keys, separate model directories, and isolated audit trails.

---

## 3. High-Level Architecture

FraudTrap follows a layered architecture where each layer adds intelligence to the scoring decision. Here is the complete request flow when a transaction arrives:

**Layer 1 — Ingestion**: A transaction arrives via the FastAPI scoring endpoint. The system validates the payload, extracts the tenant ID, and begins processing.

**Layer 2 — Tenant Resolution**: The system looks up the tenant's current ML phase from Redis. Each tenant can be in a different phase (Cold Start, Semi-Supervised, or Supervised) depending on how many fraud labels they have accumulated.

**Layer 3 — Behavioral Intelligence**: The system loads behavioral profiles for the customer, merchant, device, beneficiary, and payment instrument associated with this transaction. These profiles contain historical patterns like average spending, known devices, typical channels, and trust scores. If a profile does not exist, the system falls back to the tenant baseline, and if that does not exist, to the global baseline.

**Layer 4 — Feature Generation**: The system computes approximately 40 real-time features from the profiles and Redis. These include velocity features (how many transactions in the last minute, hour, day), trust scores (how trusted is this device, merchant), novelty flags (is this a new device, new country, new merchant), similarity scores (how similar is this transaction to the customer's history), and heuristic features (is it nighttime, is the amount unusually large).

**Layer 5 — Rules Engine**: Before any machine learning model runs, a deterministic rules engine evaluates the transaction in under 1 millisecond. It checks blocklists (is this account or device blocked?), velocity rules (too many transactions too fast?), geographic rules (impossible travel speed?), and threshold rules (amount exceeds limit?). If any rule triggers a hard block, the transaction is immediately rejected without running the ML model.

**Layer 6 — ML Model Router**: If the rules engine does not block the transaction, the system routes to the appropriate ML model based on the tenant's phase. Phase 1 tenants get the cold-start ensemble. Phase 2 tenants get the NetPFN prototype-based model. Phase 3 tenants get the CatBoost champion with confidence-aware routing to FT-Transformer specialist for low-confidence cases.

**Layer 7 — Decision Engine**: The ML model's raw score is combined with the heuristic floor (a minimum risk score based on basic signals), adjusted by any rule risk boosts, calibrated for probability accuracy, and mapped to a decision (APPROVE, REVIEW, or BLOCK).

**Layer 8 — Audit and Profile Update**: The decision is logged to Redis and Kafka for audit purposes. The behavioral profiles are updated incrementally with the new transaction's data. The next transaction for this customer will be scored against fresher history.

The entire flow from ingestion to decision typically completes in 30 to 50 milliseconds, well within the 90 millisecond SLA.

---

## 4. Multi-Tenant Design

FraudTrap does not train one model per customer. That would be unmanageable at scale. Instead, it uses a hierarchical learning strategy where intelligence flows from global to tenant to customer.

**Global Level**: The platform learns patterns across all tenants. For example, it might learn that transactions over 500,000 NGN are 12 times more likely to be fraud, or that API channels have higher fraud rates than mobile channels. These global priors serve as the baseline when no tenant-specific or customer-specific data exists.

**Tenant Level**: Each bank has its own behavioral baseline. Bank A might have a 3.2% fraud rate on the API channel, while Bank B has 0.8%. The tenant model learns these bank-specific patterns. The tenant profile captures aggregate statistics like average transaction amounts, typical channels, fraud rate history, and geographic patterns.

**Customer Level**: Each customer has an individual behavioral profile. Customer A averages 2 transactions per day, always from Lagos, always via mobile. Customer B averages 15 transactions per day, mixing POS and web, split between Abuja and Lagos. These profiles personalize inference — when Customer A suddenly transacts from London via API at 3am, the novelty signal fires without needing a model per customer.

Here is a concrete example using a fintech like Opay:

Opay has millions of customers. FraudTrap does not create 5 million models. Instead, one tenant model learns Opay-specific fraud patterns. Each customer has a behavioral profile that tracks their individual patterns. When a transaction arrives, the system checks the customer profile first. If the customer is new (no profile), it falls back to the merchant profile. If the merchant is new, it falls back to the tenant baseline. If the tenant is new, it falls back to the global baseline.

This means every transaction gets scored, even with zero customer history, because there is always a fallback level of intelligence.

Multi-tenant isolation is enforced at every layer. Redis keys are namespaced by a SHA-256 hash of the tenant ID. Each tenant has its own model files, profile storage, and audit trail. A configuration change or model retraining for Bank A has zero impact on Bank B.

---

## 5. Behavioral Intelligence Layer

The behavioral intelligence layer is the core competitive advantage of FraudTrap. It transforms raw transaction metadata into rich, context-aware features in real-time. Without this layer, the system would only see the transaction in isolation — an amount, a channel, a country. With this layer, the system sees the transaction in context: is this amount unusual for this customer? Is this device known? Has this customer transacted from this country before?

### The Five Profile Types

**Customer Profile**: Tracks individual customer behavior over time. Stores total transaction count, total amount spent, average and standard deviation of transaction amounts, typical channels used (mobile, POS, web, API), typical countries transacted in, known device IDs, frequent merchants, the last transaction timestamp, and historical chargeback count. The customer profile is the most important profile because it provides the baseline for detecting anomalies in individual behavior.

**Merchant Profile**: Tracks merchant-level patterns. Stores total transaction count, historical fraud rate, average transaction amount, merchant category code, merchant country, and a computed risk score. Merchant profiles help detect compromised merchants and unusual purchasing patterns. If a merchant suddenly sees a spike in high-value transactions from new devices, the merchant risk score increases.

**Device Profile**: Tracks device-level patterns. Stores how many accounts have used this device, when the device was first seen, when it was last seen, which countries the device has been used in, whether the device is shared across more than 3 accounts, and a trust score. A device that suddenly appears with a new account in a new country is suspicious. A device that has been used by 15 different accounts is very suspicious.

**Beneficiary Profile**: Tracks counterparty patterns. Stores how many transactions have been sent to this beneficiary, the total amount sent, and the average amount. If a customer who normally sends 5,000 NGN to a handful of beneficiaries suddenly sends 500,000 NGN to a new beneficiary, the beneficiary profile flags this as unusual.

**Payment Instrument Profile**: Tracks payment method patterns. Stores transaction count, total amount, and average amount per payment instrument. This helps detect stolen card patterns where a card suddenly transacts at unusual merchants or for unusual amounts.

### Profile Hierarchy and Cold-Start Fallback

When a profile does not exist, FraudTrap falls back up the hierarchy:

1. Check if the customer profile exists. If yes, use it.
2. If not, check if the merchant profile exists. If yes, use it.
3. If not, check if the tenant baseline exists. If yes, use it.
4. If not, use the global baseline.

This guarantees that every transaction gets scored. A brand-new customer at a new merchant in a new country with no history anywhere will still receive a score based on global patterns. The score will be more conservative (higher uncertainty), but the system will not crash or return an error.

### Profile Update Mechanism

Profiles update incrementally on every scored transaction. The customer profile's transaction count increments by one. The total amount increases. The average and standard deviation are recomputed using an online algorithm (Welford's method) that does not require storing all historical transactions. The set of known devices, channels, countries, and merchants grows as new entities are observed. The last transaction timestamp updates to the current time.

This incremental update means profiles are always fresh. When a customer's spending pattern shifts — for example, they travel to a new country — the next transaction is scored against the updated profile, not last week's snapshot.

---

## 6. Feature Engineering

FraudTrap computes approximately 40 features in real-time from Redis, organized into five families. The key design principle is that some features can always be computed (they depend only on the transaction payload) while others require Redis and default to zero when Redis is unavailable.

### Velocity Features

Velocity features measure how frequently an entity (account, device, IP) has transacted recently. They are computed using Redis sorted sets, which allow O(log N) range queries over time windows.

The system tracks five time windows: 1 minute, 5 minutes, 1 hour, 24 hours, and 7 days. For each window and each entity type, it computes the transaction count, total amount, and average amount. For example, `acct_v_1m_count` is the number of transactions this account has made in the last 1 minute. `acct_v_24h_total_amt` is the total amount this account has transacted in the last 24 hours.

Velocity features are powerful fraud signals. A customer who normally transacts twice per day suddenly making 15 transactions in 1 minute is a strong indicator of account compromise or automated fraud.

### Transaction Features

Transaction features are computed directly from the transaction payload and do not require Redis. They include the raw amount, the log of the amount (to compress the range), the amount's z-score relative to the customer's historical mean (how unusual is this amount), the ratio of this amount to the customer's mean amount, cyclical encodings of the hour (sin and cos transformations to preserve the circular nature of time), the day of week, whether it is a weekend, whether it is nighttime, whether the amount is a round number, and encoded versions of the channel and transaction type.

These features are always available regardless of Redis state. They form the minimum viable feature set for scoring.

### Device and Geo Features

Device and geo features combine device information with geographic signals. The system checks whether the device is new to this customer (not in the customer's known device set), how many accounts have used this device (shared device detection), the great-circle distance between the current and previous transaction locations (using the haversine formula), the implied travel speed between consecutive transactions, whether the travel speed exceeds 900 kilometers per hour (impossible travel), and whether the transaction crosses country borders.

Impossible travel is one of the strongest fraud signals. If a customer transacts in Lagos at 2pm and then in London at 3pm, that is physically impossible by commercial flight. This suggests the account is compromised.

### Behavioural Biometrics Features

Behavioural biometrics features capture how the user interacts with the device. They include typing cadence (how fast the user types), session duration, and field visit count. The system computes a z-score for typing cadence relative to the customer's baseline — if the user is typing significantly faster or slower than usual, it may indicate a different person using the account.

### Trust and Novelty Features

Trust features assign reputation scores to entities. The device trust score reflects how trustworthy a device is based on its history. The merchant trust score reflects the merchant's fraud rate. The customer risk score combines multiple signals into a single risk indicator.

Novelty features detect new entities. The new device flag, new IP flag, new country flag, and new merchant flag all indicate when an entity appears for the first time in a customer's history. New entities are not automatically fraudulent, but they increase the risk score because fraudsters often use new devices, new IPs, and new merchants.

### Always-Computable vs. Redis-Dependent Features

The system distinguishes between features that can always be computed and features that require Redis:

**Always-computable** (no Redis needed): amount, amount_log, hour_sin, hour_cos, day_of_week, is_weekend, is_night, is_round_amount, channel_enc, txn_type_enc. These 10-15 features are sufficient for basic scoring.

**Redis-dependent** (default to 0 when unavailable): all velocity features, all trust scores, all novelty flags, all geo features, all similarity features. These 25-30 features significantly improve accuracy but are not strictly required.

When Redis is unavailable, the system still scores transactions using the always-computable features. The score will be less accurate, but the system will not fail. This graceful degradation is critical for production reliability.

---

## 7. Rules Engine

Before any machine learning model runs, a deterministic rules engine evaluates the transaction in under 1 millisecond. The rules engine is the first line of defense and handles cases that do not require machine learning.

### Rule Types

**Blocklist Rules**: Check whether an account, device, IP address, merchant, or country is in a Redis blocklist. Blocklists are maintained by compliance teams and updated in real-time via an admin API. When a blocklist rule triggers, the transaction is immediately blocked. Blocklist entries have TTL (time-to-live) and full audit trails.

**Velocity Rules**: Check whether an entity has exceeded a transaction count threshold in a time window. For example, "block if more than 10 transactions in 1 minute" or "boost risk if more than 50 transactions in 1 hour." Velocity rules map time windows to pre-computed velocity features.

**Geographic Rules**: Check for impossible travel, cross-border transactions, and transactions from sanctioned countries. Sanctioned countries (North Korea, Iran, Cuba, Syria, Russia) are hardcoded and always checked. Impossible travel detects when the implied speed between consecutive transaction locations exceeds 900 kilometers per hour.

**Threshold Rules**: Compare feature values against limits using operators like greater-than, less-than, equals, and not-equals. For example, "boost risk if amount_vs_mean_ratio > 10" or "block if amount > 500000 AND channel == API."

**Expression Rules**: Evaluate safe AST-based expressions that combine multiple conditions. The expression evaluator uses a whitelist of allowed AST nodes to prevent code injection. Expressions can reference feature names directly, like "amount > 100000 AND is_night == 1 AND is_new_device == 1."

### Rule Actions

When a rule triggers, it can take one of two actions:

**Hard Block**: The transaction is immediately rejected with a risk score of 1.0. The ML model is not consulted. This is used for blocklist matches, sanctioned country transactions, and impossible travel.

**Soft Boost**: The transaction's risk score is increased by a configurable amount (capped at 0.30 total across all triggered rules). Each rule has a per-rule maximum boost. This is used for velocity spikes, new account high-value transactions, and other suspicious but not definitive signals.

### Rule Output

The rules engine produces a RuleResult containing whether any rules were triggered, which rule IDs fired, whether a hard block occurred, the total risk boost from soft rules, and a per-rule breakdown of contributions. This output is used by the scoring orchestrator to make the final decision.

### Hot-Reloadable Rules

Rules are stored in Redis sets and YAML configuration files. Compliance teams can add or remove blocklist entries via an admin API without restarting the system. The rules engine checks file hashes on each evaluation and reloads automatically when changes are detected. This means a sanctions list update takes effect within seconds, not minutes.

---

## 8. The Three-Phase ML Lifecycle

The core innovation of FraudTrap is that it does not require labels to start protecting customers. It uses a three-phase machine learning lifecycle where each phase adds more sophistication as more data becomes available.

**Phase 1 — Cold Start**: Used when a tenant has fewer than 500 fraud labels. Relies entirely on unsupervised anomaly detection. No labeled fraud data is needed.

**Phase 2 — Semi-Supervised**: Used when a tenant has between 500 and 5,000 fraud labels. Uses NetPFN (Neural Prototypical Few-shot Network) for prototype-based few-shot learning with pseudo-labels from the cold-start ensemble.

**Phase 3 — Supervised**: Used when a tenant has more than 5,000 fraud labels. Uses confidence-aware routing with CatBoost champion (100% of transactions) and FT-Transformer specialist (low-confidence cases, ~10-15%). Meta Fusion combines outputs for final score.

Each tenant is independently routed to the appropriate phase. A single FraudTrap deployment can run Phase 1 for a new fintech, Phase 2 for a growing bank, and Phase 3 for a mature bank simultaneously.

The transitions between phases are gated by specific criteria. Phase 1 to Phase 2 requires at least 500 confirmed fraud labels, at least 500,000 transactions, at least 8 weeks of data, and a minimum PR-AUC of 0.65. Phase 2 to Phase 3 requires at least 5,000 confirmed fraud labels and a minimum PR-AUC of 0.78. These gates prevent premature promotion — a tenant is not moved to a more complex model until the simpler model has been validated.

---

## 9. Phase 1: Cold Start

Phase 1 uses three complementary anomaly detectors, each catching different types of fraud patterns. They are combined with fixed weights into a single anomaly score.

### Variational Autoencoder (VAE) — Weight: 0.55

The VAE is a neural network that learns to compress and reconstruct normal transaction patterns. During training, it sees only legitimate transactions and learns to reconstruct them accurately. When a fraudulent transaction arrives, the VAE cannot reconstruct it well, producing a high reconstruction error. This error becomes the anomaly score.

The VAE architecture uses an encoder that maps input features to a latent space (mean and log-variance), and a decoder that reconstructs the input from the latent representation. The loss function combines reconstruction accuracy (MSE) with a KL divergence term that regularizes the latent space. The beta parameter controls the trade-off — higher beta encourages a more structured latent space but may reduce reconstruction quality.

After training, the VAE's anomaly threshold is calibrated at the 98th percentile of training reconstruction errors. Transactions with reconstruction errors above this threshold are considered anomalous. An Extreme Value Theory (EVT) tail model is also fitted to the distribution of reconstruction errors to better estimate the probability of extreme anomalies.

**What the VAE catches well**: Novel fraud patterns that deviate from the learned normal distribution. For example, a transaction with an unusual combination of features (high amount, new device, impossible travel, nighttime) that does not match any normal pattern in the training data.

**What the VAE misses**: Sparse anomalies that are individually normal but collectively suspicious. A single transaction with amount slightly above average is not anomalous, but 50 such transactions in an hour is.

### Isolation Forest — Weight: 0.30

The Isolation Forest is an ensemble of decision trees that isolate anomalies by randomly partitioning the feature space. Anomalies are easier to isolate (require fewer splits) than normal points. The anomaly score is based on the average path length across all trees — shorter paths indicate anomalies.

The implementation uses 200 trees with 2% contamination assumption. It does not require labels and works well on high-dimensional data.

**What the Isolation Forest catches well**: Sparse anomalies in feature space. A transaction that is far from the density of normal transactions in any feature dimension will be isolated quickly.

**What the Isolation Forest misses**: Clustered anomalies. If many fraudulent transactions are similar to each other (same fraud ring), they form a cluster that is not easily isolated.

### Empirical Tail Detector — Weight: 0.15

The tail detector is inspired by ECOD (Empirical Cumulative Distribution Function-based Outlier Detection) and COPOD (Copula-Based Outlier Detection). It uses robust z-scores based on median and interquartile range (IQR) to detect statistical outliers. The computation is O(n*d) — linear in the number of samples and features — making it scalable.

For each feature, the tail detector computes how many IQRs away from the median the current value is. Values far in the tail of the distribution receive high scores. This catches extreme values that the other detectors might miss.

**What the tail detector catches well**: Extreme values in individual features. A transaction amount that is 10 standard deviations above the mean is caught even if the combination of features is not unusual.

**What the tail detector misses**: Subtle anomalies where individual features are normal but the combination is suspicious.

### Why the Ensemble Is Stronger Than Any Single Model

Each detector has blind spots. The VAE may miss sparse anomalies because it learns an "average" normal. The Isolation Forest struggles with clustered anomalies. The tail detector only catches extreme values. By combining all three, the ensemble covers each detector's weaknesses with another detector's strengths. The weighted combination produces a more robust anomaly score than any single method.

---

## 10. Phase 2: Semi-Supervised Learning

When a tenant accumulates enough fraud labels (at least 500 confirmed cases from chargebacks and manual reviews), Phase 2 introduces NetPFN (Neural Prototypical Few-shot Network) for prototype-based few-shot learning while preserving the cold-start foundation.

### Pseudo-Labeling

The core challenge of Phase 2 is that 500 labels is not enough for a robust supervised model. FraudTrap addresses this with NetPFN (Neural Prototypical Few-shot Network), which uses prototype-based learning to discriminate between fraud and legitimate transactions in a learned embedding space. High-confidence predictions from the cold-start ensemble become pseudo-labels for prototype learning.

The system maintains two thresholds that evolve as the label count grows:

**High-confidence fraud threshold**: Starts at 0.99 (very conservative — only flag as fraud if the model is almost certain) and gradually decreases to 0.93 as more labels confirm the model's accuracy.

**Low-confidence legit threshold**: Starts at 0.05 (very conservative — only flag as legitimate if the model is almost certain) and gradually increases to 0.15.

Transactions scoring above the fraud threshold or below the legit threshold receive pseudo-labels. Transactions in the uncertainty zone (between the thresholds) are routed to the human review queue for manual labeling. This ensures the training set grows with both high-confidence predictions and confirmed human labels.

### Label Propagation

FraudTrap also uses graph-based label propagation. When a confirmed fraud case is identified, the system performs a breadth-first search (BFS) through the transaction graph — following links between accounts, devices, and counterparties. Each hop decays the label strength by a configurable factor (default 0.5). This means if Account A is confirmed fraud, and Account B shares a device with Account A, and Account C sends money to Account B, then Accounts B and C receive attenuated fraud labels.

Label propagation helps spread limited labels across the graph, expanding the effective training set without requiring manual review of every connected transaction.

### Adaptive Training

NetPFN learns fraud and legitimate prototypes in a learned embedding space. The model uses:

### Distance-Based Classification

NetPFN classifies transactions by computing distances to fraud and legitimate prototypes in the embedding space:

```
embedding = encoder(transaction_features)
distance_to_fraud = ||embedding - fraud_prototype||
distance_to_legit = ||embedding - legit_prototype||
risk_score = softmax(distance_to_fraud, distance_to_legit)
```

Prototypes are updated incrementally as new labeled data arrives. The model transitions smoothly from unsupervised to supervised learning without sudden jumps in behavior.

---

## 11. Phase 3: Champion-Challenger Supervised Learning

Phase 3 deploys confidence-aware routing where CatBoost serves as the production champion (100% of transactions) and FT-Transformer acts as a specialist for low-confidence cases (~10-15% of transactions). Meta Fusion combines their outputs for the final score.

### Why CatBoost as Champion

CatBoost (Categorical Boosting) is chosen as the production model for several reasons:

**Native categorical handling**: CatBoost handles categorical features natively without target encoding, which prevents target leakage. Other gradient boosting implementations require manual encoding that can introduce subtle biases.

**Built-in class imbalance handling**: The `auto_class_weights="Balanced"` parameter automatically adjusts weights based on the class distribution. This is critical for fraud detection where fraud cases are typically 1-5% of transactions.

**Fast inference**: CatBoost's native format (.cbm) provides fast prediction times, typically 1-5 milliseconds per transaction.

**Robust to overfitting**: Ordered boosting in CatBoost prevents overfitting by ensuring each tree is trained on data that was not used to compute the gradients.

**GPU support**: CatBoost can leverage GPU acceleration for training when available.

### Why FT-Transformer is a Specialist, Not a Challenger

FT-Transformer is invoked selectively for low-confidence cases (~10-15% of transactions) where CatBoost's confidence is below a threshold. The rationale:

**Specialist role**: FT-Transformer captures non-linear interactions that tree models miss. By invoking it only for uncertain cases, we get the best of both worlds: CatBoost's speed for easy cases and FT-Transformer's accuracy for hard cases.

**Risk mitigation**: An unvalidated model in production could miss new fraud patterns or increase false positives. By using FT-Transformer as a specialist (not a full challenger), we limit its impact to uncertain cases.

**Offline challengers**: LightGBM and XGBoost are trained offline and evaluated continuously. They exist to detect when a different algorithm would be superior, but are never used in production inference.

### The Specialist and Offline Challengers

**FT-Transformer Specialist**: A tabular transformer that tokenizes features and applies self-attention. Uses 64-dimensional tokens, 4 attention heads, 2 transformer layers, and a classification head. Invoked for low-confidence cases (~10-15% of transactions) where CatBoost confidence is below threshold. Falls back to GradientBoosting if PyTorch is unavailable.

**Meta Fusion Layer**: Logistic regression combining CatBoost and FT-Transformer outputs with confidence-weighted gating:

```
if catboost_confidence < threshold:
    final_score = meta_fusion(catboost_score, ft_transformer_score)
else:
    final_score = catboost_score
```

**Offline Challengers** (never in production):
- **LightGBM**: Light Gradient Boosting Machine with 500 estimators, max depth 6, 63 leaves. Uses `is_unbalance=True` for class imbalance handling.
- **XGBoost**: Extreme Gradient Boosting with 500 estimators, max depth 6, learning rate 0.05. Uses `scale_pos_weight` for class imbalance and `eval_metric=aucpr` for optimization.

### Promotion Criteria

A challenger can only be promoted to champion if it meets all of the following criteria:

**PR-AUC improvement**: The challenger's PR-AUC must exceed the champion's PR-AUC by at least 0.01. PR-AUC (Precision-Recall Area Under Curve) is the primary metric because it is more informative than ROC-AUC for imbalanced datasets where fraud is rare.

**False Positive Rate**: The challenger's FPR must be at most 0.01 (1%). Higher FPR means more legitimate transactions are incorrectly flagged, which increases customer friction and review queue volume.

**Calibration error**: The challenger's calibration error must be at most 0.05. Calibration measures how well the predicted probabilities match actual fraud rates. A model that predicts 0.7 risk should be correct approximately 70% of the time. Poorly calibrated probabilities make threshold-based decisions unreliable.

**Latency**: The challenger's inference latency must be no more than 2x the champion's latency. Production systems cannot tolerate a challenger that doubles scoring time.

**Validation sample size**: The challenger must be evaluated on at least 1,000 validation samples to ensure statistical significance.

### Promotion Workflow

When a challenger is trained, the evaluation framework computes all metrics on a held-out test set. If all criteria are met, a promotion request is generated. The request includes the challenger's metrics, the champion's metrics, the improvement deltas, and whether each criterion was met.

If auto-approve is enabled (default), the promotion is executed automatically. If manual approval is required, the request is queued for human review. The reviewer sees the side-by-side comparison and can approve or reject.

On approval, the current champion is demoted to challenger status, and the challenger is promoted to champion. The promotion is logged with full audit trail including the reviewer, timestamp, metrics comparison, and reason.

### Rollback

If the new champion degrades performance after promotion, the system can roll back to the previous champion. The previous champion is preserved in the model registry as an archived model. Rollback restores the previous champion as active, demotes the new champion to archived, and logs the rollback with reason.

---

## 12. Dynamic Model Routing

The model router is responsible for directing each tenant's transactions to the appropriate ML model. The routing decision is based solely on the tenant's label maturity — how many confirmed fraud labels they have.

The routing logic is straightforward:

1. Look up the tenant's current phase from Redis.
2. If the tenant has fewer than 500 fraud labels, route to Phase 1 (Cold Start).
3. If the tenant has between 500 and 5,000 fraud labels, route to Phase 2 (Semi-Supervised NetPFN).
4. If the tenant has more than 5,000 fraud labels, route to Phase 3 (Confidence-Aware Routing: CatBoost champion + FT-Transformer specialist).

The phase is stored in Redis and updated automatically when the label count crosses a threshold. The routing check adds less than 1 millisecond to the scoring path.

Each tenant's phase is independent. Bank A might be in Phase 3 while Bank B is in Phase 1. The system handles this seamlessly because each tenant has its own model files, its own feature store namespace, and its own profile storage.

The routing also handles edge cases. If the model for the tenant's phase is unavailable (failed to load, corrupted, etc.), the router falls back to the previous phase's model. If Phase 3 model is unavailable, try Phase 2. If Phase 2 is unavailable, try Phase 1. If all models are unavailable, fall back to heuristic scoring (a rules-only mode that uses basic signals like amount, channel, and geo to estimate risk).

---

## 13. The Scoring Orchestrator

The scoring orchestrator is the central component that wires all other components into a single real-time decision. It is the entry point called by the FastAPI scoring endpoint.

### Step-by-Step Flow

**Step 1 — Feature Assembly (5-15ms)**: The orchestrator calls `assemble_feature_vector` with the transaction and the Redis connection. This function computes all features from the five families (velocity, transaction, device/geo, behavioural, trust/novelty). If Redis is unavailable, it falls back to always-computable features only.

**Step 2 — Feature Validation**: The orchestrator checks every feature value for NaN and Inf. Invalid values are replaced with 0.0 to prevent model crashes. A warning is logged for monitoring.

**Step 3 — Schema Validation**: The orchestrator validates that the feature vector is compatible with the model's expected schema. If features are missing, they default to 0.0 with a warning. If the schema has changed since the model was trained, a compatibility warning is logged.

**Step 4 — Rules Engine (<1ms)**: The orchestrator passes the transaction and features to the rules engine. If any rule triggers a hard block, the orchestrator immediately returns a BLOCK decision with risk score 1.0. The ML model is not consulted. This saves computation and ensures deterministic enforcement of compliance rules.

**Step 5 — ML Scoring (10-30ms)**: If the rules engine does not block, the orchestrator routes to the appropriate ML model based on the tenant's phase. For Phase 3 tenants, confidence-aware routing invokes FT-Transformer specialist for low-confidence cases. The model receives the feature vector and returns a risk score.

**Step 6 — Policy Floor**: The orchestrator applies the policy floor: `risk_score = max(model_score, heuristic_score)`. The heuristic score is a simple formula based on basic signals (amount, new device, impossible travel, velocity). This ensures a minimum risk level even if the model outputs a very low score.

**Step 7 — Rules Boost**: If soft rules triggered (not hard blocks), their risk boost is added to the score, capped at 1.0.

**Step 8 — Decision Mapping**: The score is mapped to a decision: below 0.40 is APPROVE, between 0.40 and 0.85 is REVIEW, above 0.85 is BLOCK.

**Step 9 — Audit**: The decision is written to Redis (recent scores list, limited to 5,000 entries) and emitted to Kafka (audit topic, scored transaction topic).

**Step 10 — Async Tasks**: The orchestrator submits async tasks for GNN scoring (Tier 4, if available) and profile updates. These do not block the response.

### Timing Budget

The entire flow has a 90 millisecond SLA. The budget is allocated as: feature assembly (15ms), rules engine (1ms), ML scoring (50ms), decision and audit (5ms), buffer (19ms). In practice, the system typically completes in 30-50 milliseconds.

### Model Warmup

On startup, the orchestrator runs dummy inference on all loaded models. This warms up JIT compilation, caches, and internal buffers. Without warmup, the first request after a restart could take 200+ milliseconds due to cold-start overhead. With warmup, the first request meets the same latency as subsequent requests.

### Hot-Reload

The orchestrator uses a filesystem watchdog to detect model file changes. When a new model is saved to disk, the watchdog triggers a reload. The reload uses double-buffered atomic swap: the new model is loaded into a staging area, validated, and then atomically swapped with the active model. The old model continues serving traffic until the swap is complete, ensuring zero downtime.

---

## 14. Score Fusion and Decision Making

The final risk score is a carefully calibrated combination of multiple signals. Here is the exact formula for each phase.

### Phase 1 Fusion

```
risk_score = 0.55 × VAE_reconstruction_error
           + 0.30 × IsolationForest_anomaly_score
           + 0.15 × TailDetector_zscore
```

The raw score is normalized to [0, 1] using calibration points stored during training (percentiles of training reconstruction errors). Rules adjustments are then applied: hard blocks set the score to 1.0, soft boosts add to the score.

### Phase 2 Fusion

```
embedding = NetPFN_encoder(transaction_features)
distance_to_fraud = ||embedding - fraud_prototype||
distance_to_legit = ||embedding - legit_prototype||
risk_score = softmax(distance_to_fraud, distance_to_legit)
```

NetPFN classifies transactions by computing distances to fraud and legitimate prototypes in the embedding space. Prototypes are updated incrementally as new labeled data arrives.

### Phase 3 Fusion

```
if CatBoost_confidence < threshold:
    risk_score = meta_fusion(CatBoost_score, FT_Transformer_score)
else:
    risk_score = CatBoost_score
```

The CatBoost score is calibrated using Isotonic Regression. For low-confidence cases, Meta Fusion combines CatBoost and FT-Transformer outputs using a logistic regression meta-learner. The policy floor is still applied: `risk_score = max(risk_score, heuristic_score)`.

### The Policy Floor

The policy floor is a critical safety mechanism. Even if the ML model outputs 0.05 (very low risk), the heuristic floor ensures a minimum risk score based on basic signals:

```
heuristic_score = 0.08  # base
+ min(0.30, amount_zscore × 0.04)     # large amount
+ min(0.22, amount / 2_500_000)        # absolute amount
+ 0.16 if is_new_device                 # new device
+ 0.12 if is_new_merchant               # new merchant
+ 0.22 if impossible_travel             # impossible travel
+ min(0.18, velocity_1m × 0.015)       # velocity spike
+ 0.05 if is_night                      # nighttime
+ 0.05 if channel in [API, USSD]        # high-risk channel
```

The model can raise the score above the floor, but never below it. This ensures that even a poorly calibrated model cannot completely suppress risk signals.

### Decision Thresholds

| Score Range | Decision | Meaning |
|-------------|----------|---------|
| < 0.40 | APPROVE | Transaction proceeds normally |
| 0.40 — 0.85 | REVIEW | Flagged for human review queue |
| ≥ 0.85 | BLOCK | Transaction rejected |

The REVIEW band is the most important. It catches transactions that are suspicious but not definitively fraud. Human reviewers examine these cases, make a decision, and the labels flow back into the training pipeline. This creates a virtuous cycle: more reviews produce more labels, which improve the model, which reduces the review volume.

---

## 15. Explainability

Every fraud decision must be explainable. Regulators require it for compliance. Customer service needs it for dispute resolution. ML engineers need it for debugging. FraudTrap provides explainability at multiple levels.

### Phase 1 Explanations

The cold-start ensemble provides per-component explanations. For each transaction, it reports the VAE reconstruction error, the Isolation Forest path length, and the tail detector z-score. This tells the reviewer which anomaly detector flagged the transaction and by how much.

### Phase 2 and 3 Explanations

For supervised models, FraudTrap uses SHAP (SHapley Additive exPlanations). SHAP computes the contribution of each feature to the prediction. For example, a transaction might have: impossible_travel contributing +0.31, acct_v_1m_count contributing +0.22, amount_zscore contributing +0.18, is_new_device contributing +0.11. The base value (average prediction) is 0.12, and the final prediction is 0.87.

SHAP values are theoretically grounded in game theory and provide locally accurate explanations. The contributions sum to the difference between the base value and the prediction.

### Rule Explanations

When rules trigger, the rules engine provides per-rule contributions. Each rule reports its ID, description, the feature values that triggered it, and the risk boost it applied. This gives reviewers a clear picture of which compliance rules were violated.

### Reason Codes

For human reviewers, explanations are translated into reason codes:

- IMPOSSIBLE_TRAVEL: Transaction location implies impossible travel speed
- VELOCITY_SPIKE: Unusual burst of transactions in short window
- AMOUNT_ANOMALY: Transaction amount significantly deviates from customer baseline
- NEW_DEVICE: Transaction from unrecognized device
- HIGH_RISK_CHANNEL: Transaction via high-risk channel (API, USSD)
- NEW_MERCHANT: Transaction with unfamiliar merchant

These reason codes are displayed in the review queue and included in customer communications.

---

## 16. Online Learning and Profile Updates

Every transaction makes FraudTrap smarter. This is not batch retraining — it is real-time profile evolution.

When a transaction is scored, the system updates the relevant profiles:

**Customer profile**: Transaction count increments. Total amount increases. Average and standard deviation are recomputed using Welford's online algorithm. The set of known channels, countries, merchants, and devices grows as new entities are observed. The last transaction timestamp updates.

**Merchant profile**: Transaction count increments. Fraud rate is updated (if a label arrives). Average amount is recomputed.

**Device profile**: Account count may increment (if a new account uses this device). Last seen timestamp updates. Country set grows.

**Beneficiary profile**: Transaction count and total amount update.

The update is incremental — it does not require reading the entire profile from Redis, computing statistics, and writing back. Instead, it reads the current aggregates, applies the new transaction, and writes the updated aggregates. This keeps profile updates fast (1-2 milliseconds) and Redis load low.

The consequence is that the next transaction for this customer is scored against fresher history. If a customer's spending pattern shifts — they travel to a new country, change their typical channel, or start transacting at unusual hours — the profile captures this shift immediately, and the next transaction reflects the updated baseline.

This online learning loop is a significant competitive advantage over systems that retrain models weekly or monthly. Fraud patterns evolve daily, and FraudTrap adapts at the same speed.

---

## 17. Probability Calibration

Raw model outputs are not true probabilities. A CatBoost model might output 0.7 for a transaction, but that does not mean there is a 70% chance the transaction is fraud. The raw output is a relative ranking — higher values mean higher risk, but the absolute values are not calibrated probabilities.

FraudTrap calibrates probabilities using two methods:

### Isotonic Regression

Isotonic regression fits a non-parametric, monotonically increasing function to the relationship between raw model outputs and actual fraud rates. It bins the raw predictions, computes the actual fraud rate in each bin, and fits a piecewise constant function that maps raw scores to calibrated probabilities.

Isotonic regression is flexible and can capture complex calibration curves. It is the default method for CatBoost champion calibration.

### Platt Scaling

Platt scaling fits a logistic regression model to the raw predictions. It learns parameters A and B such that `calibrated = sigmoid(A × raw + B)`. Platt scaling is parametric (assumes a sigmoid shape) but is faster at inference time and more stable with small datasets.

Platt scaling is available as an alternative calibration method for Phase 2 NetPFN.

### Calibration Evaluation

FraudTrap evaluates calibration quality using:

**Expected Calibration Error (ECE)**: The average absolute difference between predicted probabilities and actual fraud rates across probability bins. Lower is better. A perfectly calibrated model has ECE = 0.

**Brier Score**: The mean squared difference between predicted probabilities and actual outcomes. Lower is better. It measures both calibration and discrimination.

**Reliability Diagram**: A plot of predicted probabilities (x-axis) vs. actual fraud rates (y-axis). A perfectly calibrated model follows the diagonal line.

---

## 18. Production Infrastructure

### Redis

Redis serves as the online feature store. It stores:
- Velocity counters (sorted sets with time-windowed transaction counts)
- Behavioral profiles (hash maps with customer, merchant, device, beneficiary, payment instrument data)
- Feature values (cached for repeated access)
- Blocklists (sets with TTL)
- Recent scores (lists capped at 5,000 entries)
- Tenant phase state (which ML phase each tenant is in)

Redis keys are namespaced by tenant: `ft:{tenant_hash}:{entity_type}:{entity_id}:{feature_name}`. This ensures complete isolation between tenants.

The TTL for most keys is 86,400 seconds (24 hours). Velocity counters for the 7-day window have a TTL of 604,800 seconds.

### Kafka

Kafka is the event backbone. It carries four topics:
- `fraudtrap.transactions`: Raw incoming transactions
- `fraudtrap.scored`: Scored transactions with decisions
- `fraudtrap.labels`: Fraud labels from chargebacks and reviews
- `fraudtrap.audit`: Complete audit trail of all scoring decisions

Kafka provides durability, replay capability, and decoupling between components. If the audit consumer is temporarily unavailable, events are buffered in Kafka and processed when it recovers.

### ClickHouse

ClickHouse is the offline analytics store. It stores historical transaction data, drift metrics, model performance rollups, and audit data. It powers the monitoring dashboards and enables ad-hoc analysis of fraud patterns.

### MLflow

MLflow tracks experiments, models, and metrics. Every training run logs:
- Hyperparameters
- Training and validation metrics (PR-AUC, ROC-AUC, F2, FPR, calibration error)
- Model artifacts
- Feature importance
- Dataset hash and statistics

The MLflow model registry stores trained models with versioning, stage transitions (staging/production/archived), and metadata.

### PostgreSQL

PostgreSQL stores metadata that requires relational queries: tenant configurations, model registry entries, promotion request history, and audit logs.

---

## 19. Failure Modes and Graceful Degradation

Production systems fail. FraudTrap is designed to degrade gracefully, never catastrophically.

### Redis Unavailable

When Redis is unavailable, all Redis-dependent features default to zero. Velocity features are zero (no velocity spike detected). Trust scores default to 0.5 (neutral). Novelty flags default to 0 (conservative: assume known entities). The ML model still runs on the always-computable features (amount, channel, time, etc.).

Impact: Lower accuracy, but scoring continues. Rules that depend on Redis blocklists may also be unavailable, so blocklist rules skip gracefully.

### ML Model Unavailable

When the ML model fails to load or crashes during scoring, the system falls back to heuristic scoring. The heuristic score is computed from basic signals (amount z-score, new device, impossible travel, velocity). Rules still enforce hard blocks.

Impact: No ML-based fraud detection, but known fraud patterns (blocklists, velocity rules, geo rules) still work.

### Kafka Unavailable

When Kafka is unavailable, audit logging is skipped. The scoring response is still returned to the client. Events are not lost — they can be reconstructed from Redis recent scores if needed.

Impact: Audit trail gap, but no impact on scoring.

### Model Scoring Fails

If the ML model throws an exception during scoring (corrupted input, internal error), the system catches the exception, logs it, and falls back to the heuristic score. The transaction is still scored, just with lower accuracy.

### New Customer, No History

A new customer has no customer profile. The system falls back to the merchant profile. If the merchant is also new, it falls back to the tenant baseline. If the tenant is new, it falls back to the global baseline. The transaction is scored against whatever level of history exists.

### New Merchant, No History

A new merchant has no merchant profile. The system uses the global merchant baseline, which reflects average fraud rates across all merchants. The merchant risk score defaults to 0.5 (neutral). Other signals (customer profile, velocity, geo) still contribute to the score.

---

## 20. MLOps Pipeline

### Training Pipeline

The training pipeline runs on a weekly cron schedule (Monday 02:00 UTC). It:

1. Pulls labeled data with a 70-day chargeback buffer (chargebacks can take weeks to arrive).
2. Engineers features over a 180-day training window.
3. Trains the CatBoost champion with early stopping and validation split.
4. Trains offline challengers (LightGBM, XGBoost) and FT-Transformer specialist.
5. Evaluates all models on a held-out test set.
6. Registers all models in MLflow and the model registry.
7. Runs promotion evaluation against the current champion.
8. If a challenger beats the champion on all criteria, recommends promotion.

### Drift Monitoring

The system monitors several drift signals:

**Population Stability Index (PSI)**: Measures how much the feature distribution has shifted between training and serving. PSI > 0.20 triggers an alert and may trigger retraining.

**Performance Drop**: If the F1 score drops by more than 0.05 absolute compared to the training baseline, an alert is triggered and a shadow evaluation is queued.

**Calibration Drift**: If the calibration error exceeds 0.10, the probabilities are recalibrated using recent labeled data.

### Retraining Triggers

Retraining can be triggered by:
- Weekly cron schedule (Monday 02:00 UTC)
- PSI drift alert
- Performance drop alert
- New feature added or feature removed
- Hyperparameter change
- Manual trigger via admin API

### Feature Versioning

Every model stores a feature hash computed from the feature names used during training. If the feature set changes (new feature added, old feature removed), the feature hash changes, and the model is automatically flagged as potentially incompatible with the current feature pipeline. This prevents silently scoring with mismatched features.

### Model Registry

The model registry maintains the lifecycle of every model:

- **Registered**: New model trained and logged
- **Challenger**: Under evaluation, not in production
- **Champion**: Serving production traffic
- **Archived**: Previously champion, now retired
- **Failed**: Did not pass evaluation criteria

The registry supports rollback: if a new champion degrades performance, the previous champion can be restored from the archive.

---

## 21. Security and Tenant Isolation

### Data Isolation

Redis keys are namespaced by a SHA-256 hash of the tenant ID: `ft:{tenant_hash}:{entity_type}:{entity_id}:{feature_name}`. This prevents cross-tenant data leakage even if key patterns overlap.

Model files are stored in per-tenant directories: `models/{tenant_id}/phase1/`, `models/{tenant_id}/phase2/`, `models/{tenant_id}/phase3/`. Each tenant's models are completely separate.

Behavioral profiles are tenant-scoped. A customer profile for Bank A cannot be accessed by Bank B.

### Audit Trail

Every scoring decision is logged with:
- Transaction ID and tenant ID
- Risk score and decision
- Model phase and version
- Triggered rules
- Trace ID for end-to-end tracking
- Timestamp

Audit events are written to Redis (recent scores, limited to 5,000) and Kafka (audit topic, unlimited). This provides both real-time access (for dashboards) and long-term storage (for compliance).

### Blocklist Management

Blocklists are maintained via an admin API with authentication. Changes are logged with the admin identity and timestamp. Blocklist entries have configurable TTL — temporary blocks expire automatically. Permanent blocks require explicit removal.

---

## 22. API Reference

### POST /v1/score

Score a transaction.

**Request**:
```json
{
  "tenant_id": "bank_ng_gtb",
  "account_id": "tok_acct_12345",
  "amount": 45000,
  "currency": "NGN",
  "transaction_type": "PAYMENT",
  "channel": "MOBILE",
  "country_code": "NG",
  "device_id": "tok_dev_67890",
  "timestamp": "2026-07-20T15:30:00Z"
}
```

**Response**:
```json
{
  "transaction_id": "txn-uuid",
  "tenant_id": "bank_ng_gtb",
  "risk_score": 0.246,
  "decision": "APPROVE",
  "model_phase": "SUPERVISED",
  "model_version": "v1_cb_a1b2c3d4",
  "latency_ms": 32.5,
  "triggered_rules": [],
  "trace_id": "trace-uuid",
  "scored_at": "2026-07-20T15:30:00Z"
}
```

### GET /v1/recent

Retrieve recent scored transactions.

**Query Parameters**:
- `limit`: Number of results (default 100, max 5000)

### GET /v1/phase/{tenant_id}

Get a tenant's current ML phase and model version.

### GET /v1/explain/{trace_id}

Get the explanation for a previously scored transaction.

### GET /v1/health

Health check endpoint.

---

## 23. Deployment

### Docker Compose

The recommended local development setup uses Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up -d api dashboard live_simulator
```

This starts the FastAPI scoring API, Streamlit dashboard, and live transaction simulator.

### Production Deployment

For production, the components should be deployed separately:

- **API**: Kubernetes deployment with horizontal pod autoscaling based on request latency
- **Redis**: Redis Cluster for high availability
- **Kafka**: Confluent Cloud or self-managed cluster
- **ClickHouse**: ClickHouse Cloud or self-managed cluster
- **PostgreSQL**: Managed service (RDS, Cloud SQL)
- **MLflow**: Self-hosted or Databricks

### Scaling Considerations

- **API**: Stateless, scales horizontally. Each pod handles ~1000 requests/second.
- **Redis**: Redis Cluster with sharding. Each shard handles ~100,000 operations/second.
- **Kafka**: Partitions scale with throughput. Each topic should have enough partitions for consumer parallelism.
- **Model loading**: Models are loaded on API startup. For large models, consider model caching with LRU eviction.

---

## 24. Design Decisions and Rationale

### Why unsupervised first, not supervised?

Most fraud detection systems require labeled data before they can function. This creates a chicken-and-egg problem: you need fraud detection to identify fraud, but you need fraud labels to train the detector. FraudTrap breaks this cycle with unsupervised anomaly detection that works from day one.

### Why hierarchical profiling, not per-customer models?

A bank with 5 million customers cannot have 5 million models. The training, serving, and maintenance costs would be prohibitive. Hierarchical profiling provides personalization through profiles while maintaining a manageable number of models (one per tenant per phase).

### Why online profile updates, not batch retraining?

Fraud patterns evolve daily. A customer who normally transacts in Lagos might travel to London tomorrow. Batch retraining (weekly or monthly) means the model sees yesterday's patterns today. Online profile updates mean the system adapts in real-time.

### Why a policy floor?

ML models can be wrong. A model might output 0.05 for a transaction with impossible travel and a velocity spike. The policy floor ensures that basic risk signals always contribute to the score, preventing the model from completely overriding rule-based safety mechanisms.

### Why Champion-Challenger instead of a single model?

A single model deployed and forgotten will degrade over time as fraud patterns evolve. The Champion-Challenger architecture ensures continuous evaluation. Even if no challenger beats the champion today, the evaluation process catches degradation and provides a mechanism for improvement.

### Why FT-Transformer as a specialist, not a full challenger?

FT-Transformer captures non-linear interactions that tree models miss, but it is slower than CatBoost. By invoking it only for low-confidence cases (~10-15% of transactions), we get the best of both worlds: CatBoost's speed for easy cases and FT-Transformer's accuracy for hard cases. This keeps latency within SLA while improving accuracy on uncertain transactions.

### Why NetPFN for Phase 2 instead of XGBoost?

With only 500-5,000 labels, traditional supervised models like XGBoost overfit. NetPFN (Neural Prototypical Few-shot Network) uses prototype-based learning that generalizes better with limited data. It learns fraud and legitimate prototypes in a learned embedding space, classifying new transactions by distance to prototypes. This approach is more robust than pseudo-labeling with XGBoost when labels are scarce.

### Why Isotonic Regression for calibration?

Isotonic regression is non-parametric and can capture complex calibration curves. Platt scaling assumes a sigmoid shape, which may not match the actual relationship between model outputs and fraud probabilities. Isotonic regression produces more reliable calibrated probabilities, which is critical for threshold-based decisions.

### Why Redis for features, not a database?

Redis provides sub-millisecond latency for feature lookups. A relational database would add 5-10 milliseconds per query, which is unacceptable for a 90ms SLA. Redis also supports sorted sets (for velocity counters) and sets (for novelty checks) natively, making feature computation efficient.

### Why hot-reloadable rules?

Sanctions lists change frequently. A bank might block a new set of accounts daily. Requiring a system restart for each change would be operationally unacceptable. Hot-reloadable rules take effect within seconds.

### Why graceful degradation instead of fail-fast?

A fraud detection system that crashes when a dependency is unavailable leaves customers completely unprotected. Graceful degradation ensures that even with partial failures, the system continues to catch the most obvious fraud patterns (blocklists, velocity rules, impossible travel) while accepting reduced accuracy on subtler patterns.

---

*Document generated from FraudTrap source code. Architecture reflects the production system with behavioral intelligence, multi-stage ML lifecycle, and Champion-Challenger supervised learning.*
