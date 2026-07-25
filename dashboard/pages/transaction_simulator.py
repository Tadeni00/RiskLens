"""
FraudTrap Dashboard — Transaction Simulator
Enterprise console for demonstrating the complete FraudTrap inference pipeline.
Designed for stakeholder demos, executive presentations, and compliance reviews.
"""

import json
import time
import hashlib
import random
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import streamlit as st

from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


# ── Design System Helpers ─────────────────────────────────────────────────────


def _card_html(title: str, content: str, icon: str = None, badge: str = None, badge_color: str = None) -> str:
    """Build an enterprise card as raw HTML (no rendering)."""
    icon_html = ""
    if icon:
        icon_html = Icons.html(icon, 16, Colors.ACCENT) + "&nbsp;"

    badge_html = ""
    if badge:
        bc = badge_color or Colors.ACCENT
        badge_html = f'<span style="display:inline-flex;align-items:center;padding:3px 10px;border-radius:6px;font-size:{Typography.TEXT_XS};font-weight:{Typography.WEIGHT_SEMIBOLD};background:{Colors.rgba(bc, 0.15)};color:{bc};border:1px solid {Colors.rgba(bc, 0.25)}">{badge}</span>'

    return f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:12px;padding:20px;margin-bottom:16px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
        <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{icon_html}{title}</span>
        </div>
        {badge_html}
    </div>
    {content}
</div>"""


def _iframe(html: str, height: int = 800):
    """Render HTML inside an iframe via st.components.v1.html (avoids Streamlit column HTML bugs)."""
    wrapped = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body{{margin:0;padding:16px;background:{Colors.BG_PRIMARY};font-family:'Inter','IBM Plex Sans',system-ui,sans-serif;}}
*{{box-sizing:border-box;}}
pre{{background:{Colors.BG_SECONDARY};padding:12px;border-radius:8px;font-size:0.6875rem;color:{Colors.TEXT_SECONDARY};overflow-x:auto;font-family:'IBM Plex Mono','Fira Code','Consolas',monospace;margin:0;white-space:pre-wrap;}}
</style></head><body>{html}</body></html>"""
    st.components.v1.html(wrapped, height=height, scrolling=True)


def _card(title: str, content: str, icon: str = None, badge: str = None, badge_color: str = None):
    """Render an enterprise card."""
    st.markdown(_card_html(title, content, icon, badge, badge_color), unsafe_allow_html=True)


def _kv_row(label: str, value: str, value_color: str = None):
    """Render a key-value row."""
    vc = value_color or Colors.TEXT_PRIMARY
    return f"""
<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
    <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED}">{label}</span>
    <span style="font-size:{Typography.TEXT_SM};color:{vc};font-weight:{Typography.WEIGHT_MEDIUM}">{value}</span>
</div>"""


def _progress_bar(value: float, max_val: float = 1.0, color: str = Colors.ACCENT, height: int = 6):
    """Render a progress bar."""
    pct = min(value / max_val, 1.0) * 100 if max_val > 0 else 0
    return f"""
<div style="height:{height}px;background:{Colors.BG_SECONDARY};border-radius:{height//2}px;overflow:hidden;margin:4px 0">
    <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:{height//2}px;transition:width 0.3s ease"></div>
</div>"""


def _status_dot(color: str, pulse: bool = False) -> str:
    pulse_cls = " status-pulse" if pulse else ""
    return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};box-shadow:0 0 6px {Colors.rgba(color, 0.4)}{pulse_cls}"></span>'


def _decision_color(decision: str) -> str:
    return {"APPROVE": Colors.SUCCESS, "REVIEW": Colors.WARNING, "BLOCK": Colors.CRITICAL}.get(decision, Colors.TEXT_MUTED)


def _risk_color(score: float) -> str:
    if score < 0.40:
        return Colors.SUCCESS
    elif score < 0.85:
        return Colors.WARNING
    return Colors.CRITICAL


# ── Demo Scenarios ────────────────────────────────────────────────────────────

SCENARIOS = {
    "Normal Customer": {
        "description": "Regular customer, known device, normal merchant, low amount",
        "expected": "APPROVE",
        "amount": 15_000,
        "currency": "NGN",
        "transaction_type": "PAYMENT",
        "channel": "MOBILE",
        "country_code": "NG",
        "account_id": "acct_normal_001",
        "merchant_id": "merch_lagos_042",
        "beneficiary_id": "ben_family_001",
        "ip_address": "105.111.22.33",
        "device_fingerprint": "dev_known_abc123",
        "typing_cadence_ms": 150.0,
        "is_new_device": False,
        "is_new_ip": False,
        "customer_trust": 0.94,
        "velocity_1h": 2,
        "merchant_risk": 0.08,
        "beneficiary_novelty": 0.12,
        "geo_distance_km": 15,
        "impossible_travel": False,
    },
    "New Device": {
        "description": "Known customer, unknown device, slight risk increase",
        "expected": "REVIEW",
        "amount": 45_000,
        "currency": "NGN",
        "transaction_type": "TRANSFER",
        "channel": "WEB",
        "country_code": "NG",
        "account_id": "acct_normal_001",
        "merchant_id": "merch_online_118",
        "beneficiary_id": "ben_friend_007",
        "ip_address": "41.204.88.12",
        "device_fingerprint": "dev_new_xyz789",
        "typing_cadence_ms": 200.0,
        "is_new_device": True,
        "is_new_ip": True,
        "customer_trust": 0.88,
        "velocity_1h": 3,
        "merchant_risk": 0.15,
        "beneficiary_novelty": 0.30,
        "geo_distance_km": 320,
        "impossible_travel": False,
    },
    "Velocity Attack": {
        "description": "Rapid transfers, high velocity, behavior engine detects anomaly",
        "expected": "BLOCK",
        "amount": 25_000,
        "currency": "NGN",
        "transaction_type": "TRANSFER",
        "channel": "API",
        "country_code": "NG",
        "account_id": "acct_velocity_003",
        "merchant_id": "merch_unknown_999",
        "beneficiary_id": "ben_mule_012",
        "ip_address": "197.210.55.88",
        "device_fingerprint": "dev_velocity_001",
        "typing_cadence_ms": 12.0,
        "is_new_device": False,
        "is_new_ip": False,
        "customer_trust": 0.65,
        "velocity_1h": 18,
        "merchant_risk": 0.42,
        "beneficiary_novelty": 0.55,
        "geo_distance_km": 5,
        "impossible_travel": False,
    },
    "Account Takeover": {
        "description": "Known customer, new device, impossible travel, large transfer",
        "expected": "BLOCK",
        "amount": 500_000,
        "currency": "NGN",
        "transaction_type": "TRANSFER",
        "channel": "API",
        "country_code": "US",
        "account_id": "acct_premium_007",
        "merchant_id": "merch_exchange_033",
        "beneficiary_id": "ben_unknown_888",
        "ip_address": "198.51.100.42",
        "device_fingerprint": "dev_suspicious_001",
        "typing_cadence_ms": 50.0,
        "is_new_device": True,
        "is_new_ip": True,
        "customer_trust": 0.45,
        "velocity_1h": 6,
        "merchant_risk": 0.68,
        "beneficiary_novelty": 0.92,
        "geo_distance_km": 8_200,
        "impossible_travel": True,
    },
    "Mule Account": {
        "description": "Beneficiary seen across many accounts, behavioral intelligence triggers",
        "expected": "BLOCK",
        "amount": 75_000,
        "currency": "NGN",
        "transaction_type": "TRANSFER",
        "channel": "WEB",
        "country_code": "NG",
        "account_id": "acct_mule_019",
        "merchant_id": "merch_cashout_005",
        "beneficiary_id": "ben_mule_ring_001",
        "ip_address": "105.111.44.77",
        "device_fingerprint": "dev_mule_001",
        "typing_cadence_ms": 180.0,
        "is_new_device": False,
        "is_new_ip": False,
        "customer_trust": 0.52,
        "velocity_1h": 5,
        "merchant_risk": 0.75,
        "beneficiary_novelty": 0.88,
        "geo_distance_km": 25,
        "impossible_travel": False,
    },
    "Card Testing": {
        "description": "Many tiny transactions, velocity alert expected",
        "expected": "BLOCK",
        "amount": 500,
        "currency": "NGN",
        "transaction_type": "PAYMENT",
        "channel": "API",
        "country_code": "US",
        "account_id": "acct_cardtest_001",
        "merchant_id": "merch_digital_099",
        "beneficiary_id": "ben_unknown_001",
        "ip_address": "203.0.113.55",
        "device_fingerprint": "dev_botnet_001",
        "typing_cadence_ms": 2.0,
        "is_new_device": True,
        "is_new_ip": True,
        "customer_trust": 0.30,
        "velocity_1h": 47,
        "merchant_risk": 0.82,
        "beneficiary_novelty": 0.95,
        "geo_distance_km": 12_000,
        "impossible_travel": True,
    },
    "High Value Transfer": {
        "description": "Very large amount, CatBoost confidence decreases, FT-Transformer consulted",
        "expected": "REVIEW",
        "amount": 2_000_000,
        "currency": "NGN",
        "transaction_type": "TRANSFER",
        "channel": "WEB",
        "country_code": "NG",
        "account_id": "acct_corp_042",
        "merchant_id": "merch_corp_001",
        "beneficiary_id": "ben_vendor_022",
        "ip_address": "41.204.100.12",
        "device_fingerprint": "dev_corp_known",
        "typing_cadence_ms": 300.0,
        "is_new_device": False,
        "is_new_ip": False,
        "customer_trust": 0.82,
        "velocity_1h": 1,
        "merchant_risk": 0.22,
        "beneficiary_novelty": 0.35,
        "geo_distance_km": 8,
        "impossible_travel": False,
    },
    "Cold Start Tenant": {
        "description": "No labels, Cold Start Ensemble selected (Phase 1)",
        "expected": "APPROVE",
        "amount": 20_000,
        "currency": "NGN",
        "transaction_type": "PAYMENT",
        "channel": "MOBILE",
        "country_code": "NG",
        "account_id": "acct_new_tenant_001",
        "merchant_id": "merch_new_001",
        "beneficiary_id": "ben_new_001",
        "ip_address": "105.111.33.11",
        "device_fingerprint": "dev_new_tenant_001",
        "typing_cadence_ms": 120.0,
        "is_new_device": False,
        "is_new_ip": False,
        "customer_trust": 0.85,
        "velocity_1h": 2,
        "merchant_risk": 0.10,
        "beneficiary_novelty": 0.20,
        "geo_distance_km": 10,
        "impossible_travel": False,
        "_tenant_phase": 1,
        "_tenant_labels": 0,
    },
    "Adaptive Tenant": {
        "description": "Few labels, TabPFN selected (Phase 2)",
        "expected": "APPROVE",
        "amount": 35_000,
        "currency": "NGN",
        "transaction_type": "TRANSFER",
        "channel": "WEB",
        "country_code": "NG",
        "account_id": "acct_adaptive_001",
        "merchant_id": "merch_adaptive_012",
        "beneficiary_id": "ben_adaptive_005",
        "ip_address": "41.204.55.66",
        "device_fingerprint": "dev_adaptive_known",
        "typing_cadence_ms": 160.0,
        "is_new_device": False,
        "is_new_ip": False,
        "customer_trust": 0.78,
        "velocity_1h": 4,
        "merchant_risk": 0.18,
        "beneficiary_novelty": 0.28,
        "geo_distance_km": 50,
        "impossible_travel": False,
        "_tenant_phase": 2,
        "_tenant_labels": 350,
    },
    "Mature Tenant": {
        "description": "Large labeled dataset, CatBoost Champion (Phase 3)",
        "expected": "APPROVE",
        "amount": 25_000,
        "currency": "NGN",
        "transaction_type": "PAYMENT",
        "channel": "MOBILE",
        "country_code": "NG",
        "account_id": "acct_mature_001",
        "merchant_id": "merch_mature_042",
        "beneficiary_id": "ben_mature_001",
        "ip_address": "105.111.22.44",
        "device_fingerprint": "dev_mature_known",
        "typing_cadence_ms": 140.0,
        "is_new_device": False,
        "is_new_ip": False,
        "customer_trust": 0.96,
        "velocity_1h": 1,
        "merchant_risk": 0.05,
        "beneficiary_novelty": 0.08,
        "geo_distance_km": 12,
        "impossible_travel": False,
        "_tenant_phase": 3,
        "_tenant_labels": 48_000,
    },
}


# ── Mock Simulation Engine ────────────────────────────────────────────────────


@dataclass
class PipelineStep:
    name: str
    status: str = "pending"  # pending, running, complete
    latency_ms: float = 0.0
    detail: str = ""


@dataclass
class SimulationResult:
    # Transaction
    transaction: dict = field(default_factory=dict)

    # Pipeline steps
    steps: list = field(default_factory=list)

    # Model routing
    tenant_labels: int = 0
    selected_phase: int = 3
    selected_model: str = "CatBoost"
    routing_reason: str = ""
    confidence: float = 0.0
    consultation_triggered: bool = False
    ft_transformer_prob: Optional[float] = None
    meta_fusion_weights: Optional[dict] = None

    # Prediction
    fraud_probability: float = 0.0
    risk_score: float = 0.0
    decision: str = "APPROVE"
    final_probability: float = 0.0

    # Behavior
    behavior: dict = field(default_factory=dict)

    # Cold Start (Phase 1)
    cold_start: Optional[dict] = None

    # Adaptive (Phase 2)
    adaptive: Optional[dict] = None

    # Explainability
    shap_features: list = field(default_factory=list)
    counterfactual: str = ""
    natural_language: str = ""

    # Timeline
    timeline: list = field(default_factory=list)

    # Infrastructure
    infra: dict = field(default_factory=dict)

    # Latency
    total_latency_ms: float = 0.0


def _hash_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def simulate_transaction(params: dict, scenario_name: str) -> SimulationResult:
    """Run a deterministic mock simulation of the FraudTrap pipeline."""
    seed = _hash_seed(json.dumps(params, sort_keys=True, default=str))
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    result = SimulationResult()

    # ── Transaction ───────────────────────────────────────────────────────
    txn_id = f"txn_{hashlib.md5(str(params).encode()).hexdigest()[:12]}"
    result.transaction = {
        "transaction_id": txn_id,
        "tenant_id": params.get("tenant_id", "bank_ng_gtb"),
        "amount": params["amount"],
        "currency": params.get("currency", "NGN"),
        "transaction_type": params.get("transaction_type", "PAYMENT"),
        "channel": params.get("channel", "MOBILE"),
        "country_code": params.get("country_code", "NG"),
        "account_id": params.get("account_id", "acct_demo"),
        "merchant_id": params.get("merchant_id", "merch_demo"),
        "beneficiary_id": params.get("beneficiary_id", "ben_demo"),
        "ip_address": params.get("ip_address", "105.111.22.33"),
        "device_fingerprint": params.get("device_fingerprint", "dev_demo"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ── Tenant Phase ──────────────────────────────────────────────────────
    tenant_phase = params.get("_tenant_phase", 3)
    tenant_labels = params.get("_tenant_labels", 48_000)
    result.tenant_labels = tenant_labels
    result.selected_phase = tenant_phase

    # ── Pipeline Steps ────────────────────────────────────────────────────
    t = 0
    steps = [
        ("Transaction Received", 0),
        ("Feature Engineering", 3),
        ("Behavior Engine", 7),
        ("Rules Engine", 10),
        ("Model Router", 15),
        ("ML Prediction", 26),
        ("Calibration", 30),
        ("Explainability", 40),
        ("Decision", 50),
    ]
    # Add small random jitter to latencies for realism
    t = 0
    for name, base_lat in steps:
        jitter = rng.uniform(-1, 2)
        lat = max(1, base_lat + jitter)
        result.steps.append(PipelineStep(
            name=name,
            status="complete",
            latency_ms=round(lat - t, 1),
            detail="",
        ))
        t = lat

    # ── Behavior ──────────────────────────────────────────────────────────
    customer_trust = params.get("customer_trust", 0.85)
    is_new_device = params.get("is_new_device", False)
    velocity = params.get("velocity_1h", 2)
    merchant_risk = params.get("merchant_risk", 0.10)
    beneficiary_novelty = params.get("beneficiary_novelty", 0.20)
    geo_distance = params.get("geo_distance_km", 10)
    impossible_travel = params.get("impossible_travel", False)

    result.behavior = {
        "customer_trust": customer_trust,
        "device_seen": not is_new_device,
        "velocity_1h": velocity,
        "merchant_risk": merchant_risk,
        "beneficiary_novelty": beneficiary_novelty,
        "geo_distance_km": geo_distance,
        "impossible_travel": impossible_travel,
    }

    # ── Model Routing ─────────────────────────────────────────────────────
    if tenant_phase == 1:
        result.selected_model = "Cold Start Ensemble"
        result.routing_reason = "No labels available. Using unsupervised anomaly detection."
    elif tenant_phase == 2:
        result.selected_model = "TabPFN"
        result.routing_reason = f"Tenant has {tenant_labels} labels — insufficient for supervised learning."
    else:
        result.selected_model = "CatBoost Champion"
        result.routing_reason = f"Tenant has {tenant_labels:,} labels. Full supervised model available."

    # ── Fraud Probability Computation ─────────────────────────────────────
    # Build a deterministic score from features
    base = 0.05
    base += max(0, (params["amount"] - 50_000) / 2_000_000) * 0.15
    base += (1.0 - customer_trust) * 0.20
    base += min(velocity / 50, 0.25)
    base += merchant_risk * 0.15
    base += beneficiary_novelty * 0.15
    if is_new_device:
        base += 0.08
    if impossible_travel:
        base += 0.15
    base += min(geo_distance / 15_000, 0.10)
    if params.get("channel") == "API":
        base += 0.03

    # Clamp
    fraud_prob = max(0.01, min(0.98, base))

    # Phase 1 uses anomaly scores (compressed)
    if tenant_phase == 1:
        # Cold Start scores are anomaly scores, not probabilities
        anomaly_score = fraud_prob * 0.65  # compressed scale
        vae_score = anomaly_score * rng.uniform(0.85, 1.1)
        iforest_score = anomaly_score * rng.uniform(0.90, 1.05)
        tail_score = anomaly_score * rng.uniform(0.70, 1.2)
        ensemble_score = (vae_score * 0.4 + iforest_score * 0.35 + tail_score * 0.25)
        result.cold_start = {
            "vae_score": round(max(0, min(1, vae_score)), 4),
            "iforest_score": round(max(0, min(1, iforest_score)), 4),
            "tail_score": round(max(0, min(1, tail_score)), 4),
            "ensemble_score": round(max(0, min(1, ensemble_score)), 4),
        }
        fraud_prob = ensemble_score

    # Phase 2 uses TabPFN
    if tenant_phase == 2:
        tabpfn_prob = fraud_prob * rng.uniform(0.92, 1.08)
        tabpfn_prob = max(0.01, min(0.98, tabpfn_prob))
        entropy = -tabpfn_prob * np.log2(max(tabpfn_prob, 1e-10)) - (1 - tabpfn_prob) * np.log2(max(1 - tabpfn_prob, 1e-10))
        confidence = 1.0 - min(entropy / 1.0, 1.0)
        pseudo_label = tabpfn_prob >= 0.5
        result.adaptive = {
            "tabpfn_probability": round(float(tabpfn_prob), 4),
            "prediction_entropy": round(float(entropy), 4),
            "confidence": round(float(confidence), 4),
            "pseudo_label": "Fraud" if pseudo_label else "Legitimate",
            "active_learning": confidence < 0.6,
            "analyst_queue": confidence < 0.6,
        }
        fraud_prob = tabpfn_prob
        result.confidence = round(float(confidence), 4)

    # Phase 3 uses CatBoost
    if tenant_phase == 3:
        catboost_prob = fraud_prob * rng.uniform(0.95, 1.05)
        catboost_prob = max(0.01, min(0.98, catboost_prob))
        confidence = abs(catboost_prob - 0.5) * 2  # distance from decision boundary

        # Consult FT-Transformer if confidence is low
        consultation = confidence < 0.5
        if consultation:
            ft_prob = fraud_prob * rng.uniform(0.88, 1.12)
            ft_prob = max(0.01, min(0.98, ft_prob))
            w_cat = 0.65 + rng.uniform(-0.05, 0.05)
            w_ft = 1.0 - w_cat
            final_prob = w_cat * catboost_prob + w_ft * ft_prob
            result.consultation_triggered = True
            result.ft_transformer_prob = round(float(ft_prob), 4)
            result.meta_fusion_weights = {"catboost": round(float(w_cat), 3), "ft_transformer": round(float(w_ft), 3)}
        else:
            final_prob = catboost_prob

        fraud_prob = max(0.01, min(0.98, final_prob))
        result.confidence = round(float(confidence), 4)

    fraud_prob = max(0.01, min(0.98, fraud_prob))
    result.fraud_probability = round(float(fraud_prob), 4)
    result.risk_score = round(float(fraud_prob), 4)

    # ── Decision ──────────────────────────────────────────────────────────
    if fraud_prob < 0.40:
        result.decision = "APPROVE"
    elif fraud_prob < 0.85:
        result.decision = "REVIEW"
    else:
        result.decision = "BLOCK"

    result.final_probability = result.risk_score

    # ── Explainability ────────────────────────────────────────────────────
    features = []
    if params["amount"] > 100_000:
        features.append(("Large transaction amount", params["amount"] / 2_000_000 * 0.3))
    if is_new_device:
        features.append(("New device fingerprint", 0.15))
    if impossible_travel:
        features.append(("Impossible travel detected", 0.22))
    if velocity > 10:
        features.append(("High velocity (1h)", min(velocity / 50, 0.20)))
    if merchant_risk > 0.5:
        features.append(("High-risk merchant", merchant_risk * 0.18))
    if beneficiary_novelty > 0.7:
        features.append(("Novel beneficiary", beneficiary_novelty * 0.15))
    if customer_trust < 0.6:
        features.append(("Low customer trust", (1 - customer_trust) * 0.12))
    if geo_distance > 1000:
        features.append(("Long geo distance", min(geo_distance / 15_000, 0.10)))
    if params.get("channel") == "API":
        features.append(("API channel", 0.03))

    if not features:
        features.append(("Normal transaction pattern", 0.02))

    features.sort(key=lambda x: abs(x[1]), reverse=True)
    result.shap_features = features[:5]

    # Counterfactual
    cf_parts = []
    if params["amount"] > 50_000:
        cf_parts.append(f"reducing amount to ₦{int(params['amount'] * 0.2):,}")
    if is_new_device:
        cf_parts.append("using a previously trusted device")
    if impossible_travel:
        cf_parts.append("transacting from a familiar location")
    if velocity > 10:
        cf_parts.append("reducing transaction frequency")

    if cf_parts:
        new_prob = max(0.05, fraud_prob * 0.3)
        result.counterfactual = (
            f"Reducing {' and '.join(cf_parts[:2])} would reduce "
            f"predicted fraud probability from {fraud_prob:.2f} to {new_prob:.2f}."
        )
    else:
        result.counterfactual = "Transaction already has low fraud indicators. No counterfactual needed."

    # Natural language
    nl_parts = [f"Decision: {result.decision}."]
    if result.decision == "BLOCK":
        nl_parts.append("High-risk transaction blocked.")
    elif result.decision == "REVIEW":
        nl_parts.append("Moderate risk — sent to manual review queue.")
    else:
        nl_parts.append("Low risk — approved automatically.")

    top_contribs = [f[0] for f in features[:3]]
    if top_contribs:
        nl_parts.append(f"Top contributors: {', '.join(top_contribs)}.")
    result.natural_language = " ".join(nl_parts)

    # ── Timeline (dynamic latencies) ──────────────────────────────────────
    t_features = round(rng.uniform(2, 5))
    t_behavior = t_features + round(rng.uniform(3, 6))
    t_rules = t_behavior + round(rng.uniform(1, 3))
    t_routing = t_rules + round(rng.uniform(2, 5))

    if tenant_phase == 1:
        t_model = t_routing + round(rng.uniform(6, 12))
        model_label = "Cold Start Ensemble"
    elif tenant_phase == 2:
        t_model = t_routing + round(rng.uniform(5, 10))
        model_label = "TabPFN"
    else:
        t_model = t_routing + round(rng.uniform(3, 7))
        model_label = "CatBoost"

    t_calib = t_model + round(rng.uniform(1, 3))

    result.timeline = [
        (0, "Transaction received"),
        (t_features, "Feature generation"),
        (t_behavior, "Behavior Engine"),
        (t_rules, "Rules Engine"),
        (t_routing, "Model Routing"),
        (t_model, model_label),
    ]

    if result.consultation_triggered:
        t_ft = t_calib + round(rng.uniform(4, 8))
        t_fusion = t_ft + round(rng.uniform(2, 4))
        result.timeline.append((t_ft, "FT-Transformer consulted"))
        result.timeline.append((t_fusion, "Meta Fusion"))
        t_explain = t_fusion + round(rng.uniform(3, 6))
    else:
        t_explain = t_calib + round(rng.uniform(3, 6))

    result.timeline.append((t_explain, "Explanation"))
    t_decision = t_explain + round(rng.uniform(2, 4))
    result.timeline.append((t_decision, "Decision returned"))
    result.total_latency_ms = result.timeline[-1][0]

    # ── Infrastructure ────────────────────────────────────────────────────
    result.infra = {
        "redis": {"status": "Healthy", "latency": "<1ms", "cache_hit": "92%"},
        "kafka": {"status": "Healthy", "offset": "4,821,033"},
        "clickhouse": {"status": "Healthy", "query_latency": "3ms"},
        "mlflow": {"status": "Healthy", "active_run": "champion_v2.3.1"},
        "model_registry": {"status": "Healthy", "models": "3 active"},
    }

    return result


# ── Visualization Components ──────────────────────────────────────────────────


def _render_pipeline_column(result: SimulationResult, anim_step: int = -1):
    """Render the animated pipeline visualization."""
    steps_html = ""
    for i, step in enumerate(result.steps):
        if anim_step >= 0 and i > anim_step:
            state = "pending"
            bg = Colors.BG_SECONDARY
            border = Colors.BORDER_DEFAULT
            icon_color = Colors.TEXT_MUTED
            text_color = Colors.TEXT_MUTED
        elif anim_step >= 0 and i == anim_step:
            state = "active"
            bg = Colors.ACCENT_BG
            border = Colors.ACCENT
            icon_color = Colors.ACCENT
            text_color = Colors.TEXT_PRIMARY
        else:
            state = "complete"
            bg = Colors.SUCCESS_BG
            border = Colors.SUCCESS
            icon_color = Colors.SUCCESS
            text_color = Colors.TEXT_PRIMARY

        icons_map = {
            "Transaction Received": "ZAP",
            "Feature Engineering": "DATABASE",
            "Behavior Engine": "ACTIVITY",
            "Rules Engine": "SHIELD",
            "Model Router": "CPU",
            "ML Prediction": "BRAIN",
            "Calibration": "SLIDERS",
            "Explainability": "EYE",
            "Decision": "SHIELD_CHECK",
        }
        icon = icons_map.get(step.name, "ZAP")

        pulse = "animation: pulse 1.5s ease-in-out infinite;" if state == "active" else ""

        steps_html += f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;background:{bg};border:1px solid {border};border-radius:8px;transition:all 0.3s ease;{pulse}">
    {_status_dot(icon_color, state == 'active')}
    <div style="flex:1">
        <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{text_color}">{step.name}</div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">{step.latency_ms:.0f}ms</div>
    </div>
    {Icons.html(icon, 16, icon_color)}
</div>"""
        if i < len(result.steps) - 1:
            arrow_color = Colors.SUCCESS if state == "complete" else Colors.TEXT_MUTED
            steps_html += f'<div style="text-align:center;padding:2px 0;color:{arrow_color};font-size:16px">↓</div>'

    st.markdown(
        f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:12px;padding:20px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
        {Icons.html('ZAP', 18, Colors.ACCENT)}
        <span style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Pipeline Execution</span>
    </div>
    {steps_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _render_model_routing_card(result: SimulationResult):
    """Render the model routing explanation card."""
    phase_colors = {1: Colors.PHASE_1, 2: Colors.PHASE_2, 3: Colors.PHASE_3}
    phase_names = {1: "Cold Start", 2: "Adaptive Learning", 3: "Supervised"}
    phase_color = phase_colors.get(result.selected_phase, Colors.ACCENT)

    content = f"""
{_kv_row("Tenant Labels", f"{result.tenant_labels:,}")}
{_kv_row("Selected Layer", f'<span style="color:{phase_color};font-weight:{Typography.WEIGHT_SEMIBOLD}">{phase_names.get(result.selected_phase, "Unknown")}</span>')}
{_kv_row("Reason", result.routing_reason)}
{_kv_row("Current Model", f'<span style="color:{phase_color}">{result.selected_model}</span>')}
"""
    if result.consultation_triggered:
        content += f"""
<div style="margin-top:12px;padding:10px 14px;background:{Colors.rgba(Colors.WARNING, 0.08)};border:1px solid {Colors.rgba(Colors.WARNING, 0.2)};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.WARNING};font-weight:{Typography.WEIGHT_SEMIBOLD};margin-bottom:6px">
        {Icons.html('ALERT_TRIANGLE', 14, Colors.WARNING)} Confidence Below Threshold
    </div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY}">
        CatBoost confidence {result.confidence:.2f} &lt; 0.50 — FT-Transformer consulted
    </div>
</div>"""
        if result.meta_fusion_weights:
            w = result.meta_fusion_weights
            content += f"""
<div style="margin-top:8px;padding:10px 14px;background:{Colors.BG_SECONDARY};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};margin-bottom:6px">Meta Fusion Weights</div>
    <div style="display:flex;gap:12px">
        <div style="flex:1">
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">CatBoost</div>
            <div style="font-size:{Typography.TEXT_MD};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_BOLD}">{w['catboost']:.0%}</div>
            {_progress_bar(w['catboost'], 1.0, Colors.PHASE_3)}
        </div>
        <div style="flex:1">
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">FT-Transformer</div>
            <div style="font-size:{Typography.TEXT_MD};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_BOLD}">{w['ft_transformer']:.0%}</div>
            {_progress_bar(w['ft_transformer'], 1.0, Colors.CHART_5)}
        </div>
    </div>
</div>"""

    _card("Model Routing", content, icon="CPU", badge=phase_names.get(result.selected_phase, ""), badge_color=phase_color)


def _render_behavior_card(result: SimulationResult):
    """Render the behavior intelligence card."""
    b = result.behavior
    trust_color = Colors.SUCCESS if b["customer_trust"] > 0.8 else Colors.WARNING if b["customer_trust"] > 0.5 else Colors.CRITICAL
    velocity_color = Colors.SUCCESS if b["velocity_1h"] < 5 else Colors.WARNING if b["velocity_1h"] < 15 else Colors.CRITICAL
    merchant_color = Colors.SUCCESS if b["merchant_risk"] < 0.3 else Colors.WARNING if b["merchant_risk"] < 0.6 else Colors.CRITICAL
    novelty_color = Colors.SUCCESS if b["beneficiary_novelty"] < 0.3 else Colors.WARNING if b["beneficiary_novelty"] < 0.7 else Colors.CRITICAL
    geo_color = Colors.CRITICAL if b["impossible_travel"] else (Colors.WARNING if b["geo_distance_km"] > 500 else Colors.SUCCESS)

    content = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Customer Trust</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{trust_color}">{b['customer_trust']:.0%}</div>
        {_progress_bar(b['customer_trust'], 1.0, trust_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Device Seen</div>
        <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{"Yes" if b['device_seen'] else "No"}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Velocity (1h)</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{velocity_color}">{b['velocity_1h']}</div>
        {_progress_bar(b['velocity_1h'], 50, velocity_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Merchant Risk</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{merchant_color}">{b['merchant_risk']:.2f}</div>
        {_progress_bar(b['merchant_risk'], 1.0, merchant_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Beneficiary Novelty</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{novelty_color}">{b['beneficiary_novelty']:.0%}</div>
        {_progress_bar(b['beneficiary_novelty'], 1.0, novelty_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Geo Distance</div>
        <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{geo_color}">{b['geo_distance_km']:,} km</div>
    </div>
</div>
<div style="margin-top:12px;padding:10px 14px;background:{Colors.rgba(Colors.CRITICAL if b['impossible_travel'] else Colors.BG_SECONDARY, 0.1 if b['impossible_travel'] else 1.0)};border:1px solid {Colors.rgba(Colors.CRITICAL, 0.3) if b['impossible_travel'] else Colors.BORDER_SUBTLE};border-radius:8px">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html('ALERT_TRIANGLE' if b['impossible_travel'] else 'CHECK_CIRCLE', 14, Colors.CRITICAL if b['impossible_travel'] else Colors.SUCCESS)}
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.CRITICAL if b['impossible_travel'] else Colors.SUCCESS};font-weight:{Typography.WEIGHT_SEMIBOLD}">
            {"Impossible Travel Detected" if b['impossible_travel'] else "Travel Pattern Normal"}
        </span>
    </div>
</div>"""

    _card("Behavior Intelligence", content, icon="ACTIVITY")


def _render_cold_start_card(result: SimulationResult):
    """Render Phase 1 cold start visualization."""
    if result.cold_start is None:
        return
    cs = result.cold_start

    scores = [
        ("VAE Reconstruction", cs["vae_score"], "Variational Autoencoder anomaly score"),
        ("Isolation Forest", cs["iforest_score"], "Isolation Forest anomaly score"),
        ("EVT Tail", cs["tail_score"], "Extreme Value Theory tail probability"),
    ]

    gauges_html = ""
    for label, score, desc in scores:
        color = Colors.SUCCESS if score < 0.3 else Colors.WARNING if score < 0.6 else Colors.CRITICAL
        # SVG gauge
        angle = score * 180
        gauges_html += f"""
<div style="text-align:center">
    <svg width="100" height="60" viewBox="0 0 100 60">
        <path d="M10 55 A40 40 0 0 1 90 55" fill="none" stroke="{Colors.BORDER_DEFAULT}" stroke-width="8" stroke-linecap="round"/>
        <path d="M10 55 A40 40 0 0 1 90 55" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"
              stroke-dasharray="{score * 125.6:.1f} 125.6"/>
        <text x="50" y="50" text-anchor="middle" fill="{Colors.TEXT_PRIMARY}" font-size="16" font-weight="700" font-family="sans-serif">{score:.2f}</text>
    </svg>
    <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};margin-top:4px">{label}</div>
    <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">{desc}</div>
</div>"""

    content = f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px">
    {gauges_html}
</div>
<div style="padding:12px 16px;background:{Colors.BG_SECONDARY};border-radius:8px;text-align:center">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER}">Ensemble Score</div>
    <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(cs['ensemble_score'])}">{cs['ensemble_score']:.4f}</div>
</div>"""

    _card("Cold Start Ensemble (Phase 1)", content, icon="SHIELD", badge="Phase 1", badge_color=Colors.PHASE_1)


def _render_adaptive_card(result: SimulationResult):
    """Render Phase 2 adaptive learning visualization."""
    if result.adaptive is None:
        return
    ad = result.adaptive

    content = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px">
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">TabPFN Probability</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(ad['tabpfn_probability'])}">{ad['tabpfn_probability']:.4f}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Confidence</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{ad['confidence']:.1%}</div>
        {_progress_bar(ad['confidence'], 1.0, Colors.ACCENT)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Prediction Entropy</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{ad['prediction_entropy']:.4f}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Pseudo-label</div>
        <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.CRITICAL if ad['pseudo_label'] == 'Fraud' else Colors.SUCCESS}">{ad['pseudo_label']}</div>
    </div>
</div>"""

    if ad["active_learning"]:
        content += f"""
<div style="padding:10px 14px;background:{Colors.rgba(Colors.WARNING, 0.08)};border:1px solid {Colors.rgba(Colors.WARNING, 0.2)};border-radius:8px;margin-top:8px">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html('ALERT_TRIANGLE', 14, Colors.WARNING)}
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.WARNING};font-weight:{Typography.WEIGHT_SEMIBOLD}>Active Learning</span>
    </div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};margin-top:4px">
        Prediction confidence is below pseudo-label threshold. Added to analyst review queue.
    </div>
</div>"""

    _card("Adaptive Learning (Phase 2)", content, icon="LAYERS", badge="Phase 2", badge_color=Colors.PHASE_2)


def _render_supervised_card(result: SimulationResult):
    """Render Phase 3 supervised visualization."""
    if result.selected_phase != 3:
        return

    prob = result.fraud_probability
    content = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px">
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">CatBoost Probability</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(prob)}">{prob:.4f}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Confidence</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{result.confidence:.1%}</div>
        {_progress_bar(result.confidence, 1.0, Colors.ACCENT)}
    </div>
</div>"""

    if result.consultation_triggered:
        content += f"""
<div style="padding:12px 14px;background:{Colors.rgba(Colors.WARNING, 0.08)};border:1px solid {Colors.rgba(Colors.WARNING, 0.2)};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.WARNING};font-weight:{Typography.WEIGHT_SEMIBOLD};margin-bottom:8px">
        {Icons.html('ALERT_TRIANGLE', 14, Colors.WARNING)} Low Confidence — FT-Transformer Consulted
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;text-align:center">
        <div>
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">CatBoost</div>
            <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.PHASE_3}">{prob:.4f}</div>
        </div>
        <div>
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">FT-Transformer</div>
            <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.CHART_5}">{result.ft_transformer_prob:.4f}</div>
        </div>
        <div>
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">Meta Fusion</div>
            <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(result.final_probability)}">{result.final_probability:.4f}</div>
        </div>
    </div>
</div>"""

    _card("Supervised Model (Phase 3)", content, icon="BRAIN", badge="Phase 3", badge_color=Colors.PHASE_3)


def _render_explainability_card(result: SimulationResult):
    """Render explainability section."""
    # SHAP waterfall
    shap_html = ""
    for feat, weight in result.shap_features:
        bar_w = abs(weight) * 100
        bar_color = Colors.CRITICAL if weight > 0.1 else Colors.WARNING if weight > 0.05 else Colors.ACCENT
        direction = "→ +" if weight > 0 else "→ -"
        shap_html += f"""
<div style="display:flex;align-items:center;gap:8px;padding:6px 0">
    <div style="width:140px;font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};text-align:right">{feat}</div>
    <div style="flex:1;height:18px;background:{Colors.BG_SECONDARY};border-radius:4px;overflow:hidden;position:relative">
        <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.3s"></div>
    </div>
    <div style="width:60px;font-size:{Typography.TEXT_SM};color:{bar_color};font-weight:{Typography.WEIGHT_SEMIBOLD}">{direction} {weight:.3f}</div>
</div>"""

    content = f"""
<div style="margin-bottom:16px">
    <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY};margin-bottom:10px">Top SHAP Features</div>
    {shap_html}
</div>
<div style="padding:12px 14px;background:{Colors.BG_SECONDARY};border-radius:8px;margin-bottom:12px">
    <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY};margin-bottom:6px">Counterfactual</div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};line-height:1.5">{result.counterfactual}</div>
</div>
<div style="padding:12px 14px;background:{Colors.rgba(Colors.ACCENT, 0.06)};border:1px solid {Colors.rgba(Colors.ACCENT, 0.15)};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.ACCENT_LIGHT};margin-bottom:6px">Natural Language Explanation</div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};line-height:1.5">{result.natural_language}</div>
</div>"""

    _card("Explainability", content, icon="EYE")


def _render_timeline_card(result: SimulationResult):
    """Render execution timeline."""
    items = ""
    for i, (ms, label) in enumerate(result.timeline):
        is_last = i == len(result.timeline) - 1
        dot_color = Colors.SUCCESS if is_last else Colors.ACCENT
        line_html = "" if is_last else f'<div style="width:2px;height:20px;background:{Colors.BORDER_DEFAULT};margin-left:4px"></div>'
        items += f"""<div style="display:flex;gap:10px;margin-bottom:2px">
<div style="display:flex;flex-direction:column;align-items:center"><div style="width:10px;height:10px;border-radius:50%;background:{dot_color};flex-shrink:0"></div>{line_html}</div>
<div style="padding-bottom:10px"><span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};font-family:{Typography.FONT_MONO}">{ms}ms</span> <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_PRIMARY}">{label}</span></div>
</div>"""

    content = f"""<div style="margin-bottom:10px;font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED}">Total: <span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_BOLD}">{result.total_latency_ms}ms</span></div>{items}"""

    _card("Execution Timeline", content, icon="CLOCK")


def _timeline_html(result: SimulationResult) -> str:
    """Build timeline card as raw HTML."""
    items = ""
    for i, (ms, label) in enumerate(result.timeline):
        is_last = i == len(result.timeline) - 1
        dot_color = Colors.SUCCESS if is_last else Colors.ACCENT
        line_html = "" if is_last else f'<div style="width:2px;height:20px;background:{Colors.BORDER_DEFAULT};margin-left:4px"></div>'
        items += f"""<div style="display:flex;gap:10px;margin-bottom:2px">
<div style="display:flex;flex-direction:column;align-items:center"><div style="width:10px;height:10px;border-radius:50%;background:{dot_color};flex-shrink:0"></div>{line_html}</div>
<div style="padding-bottom:10px"><span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};font-family:{Typography.FONT_MONO}">{ms}ms</span> <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_PRIMARY}">{label}</span></div>
</div>"""
    content = f"""<div style="margin-bottom:10px;font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED}">Total: <span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_BOLD}">{result.total_latency_ms}ms</span></div>{items}"""
    return _card_html("Execution Timeline", content, icon="CLOCK")


def _build_pipeline_html(result: SimulationResult, anim_step: int = -1) -> str:
    """Build pipeline + timeline as a single HTML block for use inside columns."""
    steps_html = ""
    for i, step in enumerate(result.steps):
        if anim_step >= 0 and i > anim_step:
            state = "pending"
            bg = Colors.BG_SECONDARY
            border = Colors.BORDER_DEFAULT
            icon_color = Colors.TEXT_MUTED
            text_color = Colors.TEXT_MUTED
        elif anim_step >= 0 and i == anim_step:
            state = "active"
            bg = Colors.ACCENT_BG
            border = Colors.ACCENT
            icon_color = Colors.ACCENT
            text_color = Colors.TEXT_PRIMARY
        else:
            state = "complete"
            bg = Colors.SUCCESS_BG
            border = Colors.SUCCESS
            icon_color = Colors.SUCCESS
            text_color = Colors.TEXT_PRIMARY

        icons_map = {
            "Transaction Received": "ZAP",
            "Feature Engineering": "DATABASE",
            "Behavior Engine": "ACTIVITY",
            "Rules Engine": "SHIELD",
            "Model Router": "CPU",
            "ML Prediction": "BRAIN",
            "Calibration": "SLIDERS",
            "Explainability": "EYE",
            "Decision": "SHIELD_CHECK",
        }
        icon = icons_map.get(step.name, "ZAP")
        pulse = "animation: pulse 1.5s ease-in-out infinite;" if state == "active" else ""

        steps_html += f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;background:{bg};border:1px solid {border};border-radius:8px;transition:all 0.3s ease;{pulse}">
    {_status_dot(icon_color, state == 'active')}
    <div style="flex:1">
        <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{text_color}">{step.name}</div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">{step.latency_ms:.0f}ms</div>
    </div>
    {Icons.html(icon, 16, icon_color)}
</div>"""
        if i < len(result.steps) - 1:
            arrow_color = Colors.SUCCESS if state == "complete" else Colors.TEXT_MUTED
            steps_html += f'<div style="text-align:center;padding:2px 0;color:{arrow_color};font-size:16px">↓</div>'

    pipeline_card = f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:12px;padding:20px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
        {Icons.html('ZAP', 18, Colors.ACCENT)}
        <span style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Pipeline Execution</span>
    </div>
    {steps_html}
</div>"""

    return pipeline_card + f'<div style="height:16px"></div>' + _timeline_html(result)


def _build_routing_html(result: SimulationResult) -> str:
    """Build model routing card as raw HTML for use inside columns."""
    phase_colors = {1: Colors.PHASE_1, 2: Colors.PHASE_2, 3: Colors.PHASE_3}
    phase_names = {1: "Cold Start", 2: "Adaptive Learning", 3: "Supervised"}
    phase_color = phase_colors.get(result.selected_phase, Colors.ACCENT)

    content = f"""
{_kv_row("Tenant Labels", f"{result.tenant_labels:,}")}
{_kv_row("Selected Layer", f'<span style="color:{phase_color};font-weight:{Typography.WEIGHT_SEMIBOLD}">{phase_names.get(result.selected_phase, "Unknown")}</span>')}
{_kv_row("Reason", result.routing_reason)}
{_kv_row("Current Model", f'<span style="color:{phase_color}">{result.selected_model}</span>')}
"""
    if result.consultation_triggered:
        content += f"""
<div style="margin-top:12px;padding:10px 14px;background:{Colors.rgba(Colors.WARNING, 0.08)};border:1px solid {Colors.rgba(Colors.WARNING, 0.2)};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.WARNING};font-weight:{Typography.WEIGHT_SEMIBOLD};margin-bottom:6px">
        {Icons.html('ALERT_TRIANGLE', 14, Colors.WARNING)} Confidence Below Threshold
    </div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY}">
        CatBoost confidence {result.confidence:.2f} &lt; 0.50 — FT-Transformer consulted
    </div>
</div>"""
        if result.meta_fusion_weights:
            w = result.meta_fusion_weights
            content += f"""
<div style="margin-top:8px;padding:10px 14px;background:{Colors.BG_SECONDARY};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};margin-bottom:6px">Meta Fusion Weights</div>
    <div style="display:flex;gap:12px">
        <div style="flex:1">
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">CatBoost</div>
            <div style="font-size:{Typography.TEXT_MD};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_BOLD}">{w['catboost']:.0%}</div>
            {_progress_bar(w['catboost'], 1.0, Colors.PHASE_3)}
        </div>
        <div style="flex:1">
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">FT-Transformer</div>
            <div style="font-size:{Typography.TEXT_MD};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_BOLD}">{w['ft_transformer']:.0%}</div>
            {_progress_bar(w['ft_transformer'], 1.0, Colors.CHART_5)}
        </div>
    </div>
</div>"""

    return _card_html("Model Routing", content, icon="CPU", badge=phase_names.get(result.selected_phase, ""), badge_color=phase_color)


def _render_infra_card(result: SimulationResult):
    """Render infrastructure status card."""
    rows = ""
    for name, info in result.infra.items():
        status_color = Colors.SUCCESS
        rows += f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
    <div style="display:flex;align-items:center;gap:8px">
        {_status_dot(status_color)}
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{name.title()}</span>
    </div>
    <span class="ft-badge success" style="font-size:{Typography.TEXT_XS}">Healthy</span>
</div>"""

    _card("Infrastructure", rows, icon="SERVER")


# ── HTML Builders for Full-Width Cards ────────────────────────────────────────


def _build_behavior_html(result: SimulationResult) -> str:
    """Build behavior intelligence card as raw HTML."""
    b = result.behavior
    trust_color = Colors.SUCCESS if b["customer_trust"] > 0.8 else Colors.WARNING if b["customer_trust"] > 0.5 else Colors.CRITICAL
    velocity_color = Colors.SUCCESS if b["velocity_1h"] < 5 else Colors.WARNING if b["velocity_1h"] < 15 else Colors.CRITICAL
    merchant_color = Colors.SUCCESS if b["merchant_risk"] < 0.3 else Colors.WARNING if b["merchant_risk"] < 0.6 else Colors.CRITICAL
    novelty_color = Colors.SUCCESS if b["beneficiary_novelty"] < 0.3 else Colors.WARNING if b["beneficiary_novelty"] < 0.7 else Colors.CRITICAL
    geo_color = Colors.CRITICAL if b["impossible_travel"] else (Colors.WARNING if b["geo_distance_km"] > 500 else Colors.SUCCESS)

    content = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Customer Trust</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{trust_color}">{b['customer_trust']:.0%}</div>
        {_progress_bar(b['customer_trust'], 1.0, trust_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Device Seen</div>
        <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{"Yes" if b['device_seen'] else "No"}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Velocity (1h)</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{velocity_color}">{b['velocity_1h']}</div>
        {_progress_bar(b['velocity_1h'], 50, velocity_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Merchant Risk</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{merchant_color}">{b['merchant_risk']:.2f}</div>
        {_progress_bar(b['merchant_risk'], 1.0, merchant_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Beneficiary Novelty</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{novelty_color}">{b['beneficiary_novelty']:.0%}</div>
        {_progress_bar(b['beneficiary_novelty'], 1.0, novelty_color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Geo Distance</div>
        <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{geo_color}">{b['geo_distance_km']:,} km</div>
    </div>
</div>
<div style="margin-top:12px;padding:10px 14px;background:{Colors.rgba(Colors.CRITICAL if b['impossible_travel'] else Colors.BG_SECONDARY, 0.1 if b['impossible_travel'] else 1.0)};border:1px solid {Colors.rgba(Colors.CRITICAL, 0.3) if b['impossible_travel'] else Colors.BORDER_SUBTLE};border-radius:8px">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html('ALERT_TRIANGLE' if b['impossible_travel'] else 'CHECK_CIRCLE', 14, Colors.CRITICAL if b['impossible_travel'] else Colors.SUCCESS)}
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.CRITICAL if b['impossible_travel'] else Colors.SUCCESS};font-weight:{Typography.WEIGHT_SEMIBOLD}">
            {"Impossible Travel Detected" if b['impossible_travel'] else "Travel Pattern Normal"}
        </span>
    </div>
</div>"""

    return _card_html("Behavior Intelligence", content, icon="ACTIVITY")


def _build_explainability_html(result: SimulationResult) -> str:
    """Build explainability card as raw HTML."""
    shap_html = ""
    for feat, weight in result.shap_features:
        bar_w = abs(weight) * 100
        bar_color = Colors.CRITICAL if weight > 0.1 else Colors.WARNING if weight > 0.05 else Colors.ACCENT
        direction = "→ +" if weight > 0 else "→ -"
        shap_html += f"""
<div style="display:flex;align-items:center;gap:8px;padding:6px 0">
    <div style="width:140px;font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};text-align:right">{feat}</div>
    <div style="flex:1;height:18px;background:{Colors.BG_SECONDARY};border-radius:4px;overflow:hidden;position:relative">
        <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.3s"></div>
    </div>
    <div style="width:60px;font-size:{Typography.TEXT_SM};color:{bar_color};font-weight:{Typography.WEIGHT_SEMIBOLD}">{direction} {weight:.3f}</div>
</div>"""

    content = f"""
<div style="margin-bottom:16px">
    <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY};margin-bottom:10px">Top SHAP Features</div>
    {shap_html}
</div>
<div style="padding:12px 14px;background:{Colors.BG_SECONDARY};border-radius:8px;margin-bottom:12px">
    <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY};margin-bottom:6px">Counterfactual</div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};line-height:1.5">{result.counterfactual}</div>
</div>
<div style="padding:12px 14px;background:{Colors.rgba(Colors.ACCENT, 0.06)};border:1px solid {Colors.rgba(Colors.ACCENT, 0.15)};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.ACCENT_LIGHT};margin-bottom:6px">Natural Language Explanation</div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};line-height:1.5">{result.natural_language}</div>
</div>"""

    return _card_html("Explainability", content, icon="EYE")


def _build_cold_start_html(result: SimulationResult) -> str:
    """Build Phase 1 cold start card as raw HTML."""
    if result.cold_start is None:
        return ""
    cs = result.cold_start

    scores = [
        ("VAE Reconstruction", cs["vae_score"], "Variational Autoencoder anomaly score"),
        ("Isolation Forest", cs["iforest_score"], "Isolation Forest anomaly score"),
        ("EVT Tail", cs["tail_score"], "Extreme Value Theory tail probability"),
    ]

    gauges_html = ""
    for label, score, desc in scores:
        color = Colors.SUCCESS if score < 0.3 else Colors.WARNING if score < 0.6 else Colors.CRITICAL
        gauges_html += f"""
<div style="text-align:center">
    <svg width="100" height="60" viewBox="0 0 100 60">
        <path d="M10 55 A40 40 0 0 1 90 55" fill="none" stroke="{Colors.BORDER_DEFAULT}" stroke-width="8" stroke-linecap="round"/>
        <path d="M10 55 A40 40 0 0 1 90 55" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"
              stroke-dasharray="{score * 125.6:.1f} 125.6"/>
        <text x="50" y="50" text-anchor="middle" fill="{Colors.TEXT_PRIMARY}" font-size="16" font-weight="700" font-family="sans-serif">{score:.2f}</text>
    </svg>
    <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};margin-top:4px">{label}</div>
    <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">{desc}</div>
</div>"""

    content = f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px">
    {gauges_html}
</div>
<div style="padding:12px 16px;background:{Colors.BG_SECONDARY};border-radius:8px;text-align:center">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER}">Ensemble Score</div>
    <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(cs['ensemble_score'])}">{cs['ensemble_score']:.4f}</div>
</div>"""

    return _card_html("Cold Start Ensemble (Phase 1)", content, icon="SHIELD", badge="Phase 1", badge_color=Colors.PHASE_1)


def _build_adaptive_html(result: SimulationResult) -> str:
    """Build Phase 2 adaptive learning card as raw HTML."""
    if result.adaptive is None:
        return ""
    ad = result.adaptive

    content = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px">
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">TabPFN Probability</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(ad['tabpfn_probability'])}">{ad['tabpfn_probability']:.4f}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Confidence</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{ad['confidence']:.1%}</div>
        {_progress_bar(ad['confidence'], 1.0, Colors.ACCENT)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Prediction Entropy</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{ad['prediction_entropy']:.4f}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Pseudo-label</div>
        <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.CRITICAL if ad['pseudo_label'] == 'Fraud' else Colors.SUCCESS}">{ad['pseudo_label']}</div>
    </div>
</div>"""

    if ad["active_learning"]:
        content += f"""
<div style="padding:10px 14px;background:{Colors.rgba(Colors.WARNING, 0.08)};border:1px solid {Colors.rgba(Colors.WARNING, 0.2)};border-radius:8px;margin-top:8px">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html('ALERT_TRIANGLE', 14, Colors.WARNING)}
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.WARNING};font-weight:{Typography.WEIGHT_SEMIBOLD}>Active Learning</span>
    </div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};margin-top:4px">
        Prediction confidence is below pseudo-label threshold. Added to analyst review queue.
    </div>
</div>"""

    return _card_html("Adaptive Learning (Phase 2)", content, icon="LAYERS", badge="Phase 2", badge_color=Colors.PHASE_2)


def _build_supervised_html(result: SimulationResult) -> str:
    """Build Phase 3 supervised card as raw HTML."""
    if result.selected_phase != 3:
        return ""

    prob = result.fraud_probability
    content = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px">
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">CatBoost Probability</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(prob)}">{prob:.4f}</div>
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Confidence</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{result.confidence:.1%}</div>
        {_progress_bar(result.confidence, 1.0, Colors.ACCENT)}
    </div>
</div>"""

    if result.consultation_triggered:
        content += f"""
<div style="padding:12px 14px;background:{Colors.rgba(Colors.WARNING, 0.08)};border:1px solid {Colors.rgba(Colors.WARNING, 0.2)};border-radius:8px">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.WARNING};font-weight:{Typography.WEIGHT_SEMIBOLD};margin-bottom:8px">
        {Icons.html('ALERT_TRIANGLE', 14, Colors.WARNING)} Low Confidence — FT-Transformer Consulted
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;text-align:center">
        <div>
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">CatBoost</div>
            <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.PHASE_3}">{prob:.4f}</div>
        </div>
        <div>
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">FT-Transformer</div>
            <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.CHART_5}">{result.ft_transformer_prob:.4f}</div>
        </div>
        <div>
            <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">Meta Fusion</div>
            <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_BOLD};color:{_risk_color(result.final_probability)}">{result.final_probability:.4f}</div>
        </div>
    </div>
</div>"""

    return _card_html("Supervised Model (Phase 3)", content, icon="BRAIN", badge="Phase 3", badge_color=Colors.PHASE_3)


def _build_infra_html(result: SimulationResult) -> str:
    """Build infrastructure status card as raw HTML."""
    rows = ""
    for name, info in result.infra.items():
        status_color = Colors.SUCCESS
        rows += f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
    <div style="display:flex;align-items:center;gap:8px">
        {_status_dot(status_color)}
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{name.title()}</span>
    </div>
    <span class="ft-badge success" style="font-size:{Typography.TEXT_XS}">Healthy</span>
</div>"""

    return _card_html("Infrastructure", rows, icon="SERVER")


# ── Main Render ───────────────────────────────────────────────────────────────


def render(tenant_id: str):
    # ── Page Header ───────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="page-header">
    <h1>{Icons.html('ZAP', 28, Colors.ACCENT)}&nbsp;Transaction Simulator</h1>
    <p>Enterprise console for demonstrating the complete FraudTrap inference pipeline</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Demo Mode Toggle ──────────────────────────────────────────────────
    demo_col, _ = st.columns([1, 3])
    with demo_col:
        demo_mode = st.toggle("Demo Mode", value=True, help="Deterministic outputs for repeatable presentations")

    st.markdown(f'<div style="height:8px"></div>', unsafe_allow_html=True)

    # ── Sidebar Controls (in main area) ───────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    with ctrl_col1:
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
        scenario = SCENARIOS[scenario_name]
        st.caption(scenario["description"])

    with ctrl_col2:
        amount = st.number_input("Transaction Amount (NGN)", value=scenario["amount"], min_value=100, max_value=100_000_000, step=1000)
        txn_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "WITHDRAWAL"], index=["PAYMENT", "TRANSFER", "WITHDRAWAL"].index(scenario["transaction_type"]))
        channel = st.selectbox("Channel", ["MOBILE", "WEB", "API", "USSD"], index=["MOBILE", "WEB", "API", "USSD"].index(scenario["channel"]))

    with ctrl_col3:
        country = st.selectbox("Country", ["NG", "US", "GB", "ZA", "KE", "GH"], index=["NG", "US", "GB", "ZA", "KE", "GH"].index(scenario["country_code"]))
        model_override = st.selectbox("Model Override (Demo)", ["Auto (follow pipeline)", "Force Phase 1", "Force Phase 2", "Force Phase 3"])

    st.markdown(f'<div style="height:4px"></div>', unsafe_allow_html=True)

    # Run button
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)

    # ── Run Simulation ────────────────────────────────────────────────────
    if run_simulation:
        # Build params from controls
        params = dict(scenario)
        params["amount"] = amount
        params["transaction_type"] = txn_type
        params["channel"] = channel
        params["country_code"] = country
        params["tenant_id"] = tenant_id

        # Model override
        if "Force Phase 1" in model_override:
            params["_tenant_phase"] = 1
            params["_tenant_labels"] = 0
        elif "Force Phase 2" in model_override:
            params["_tenant_phase"] = 2
            params["_tenant_labels"] = 350
        elif "Force Phase 3" in model_override:
            params["_tenant_phase"] = 3
            params["_tenant_labels"] = 48_000

        # Run with animation
        result = simulate_transaction(params, scenario_name)

        # Store in session
        st.session_state["sim_result"] = result
        st.session_state["sim_running"] = True
        st.session_state["sim_step"] = 0

    # ── Render Results ────────────────────────────────────────────────────
    if "sim_result" in st.session_state:
        result = st.session_state["sim_result"]

        st.markdown(f'<div style="height:16px"></div>', unsafe_allow_html=True)

        # ── Three-Column Layout ───────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 1, 1])

        # ── Column 1: Transaction ─────────────────────────────────────────
        with col1:
            txn_display = {k: v for k, v in result.transaction.items()}
            txn_json_html = f"""<pre style="background:{Colors.BG_SECONDARY};padding:12px;border-radius:8px;font-size:{Typography.TEXT_XS};color:{Colors.TEXT_SECONDARY};overflow-x:auto;font-family:{Typography.FONT_MONO};margin:0;white-space:pre-wrap">{json.dumps(txn_display, indent=2)}</pre>"""

            feat_content = ""
            for k, v in result.transaction.items():
                if k != "timestamp":
                    feat_content += _kv_row(k.replace("_", " ").title(), str(v))

            b = result.behavior
            beh_content = f"""
{_kv_row("Customer Trust", f"{b['customer_trust']:.0%}", Colors.SUCCESS if b['customer_trust'] > 0.8 else Colors.WARNING)}
{_kv_row("Device Seen", "Yes" if b['device_seen'] else "No", Colors.SUCCESS if b['device_seen'] else Colors.CRITICAL)}
{_kv_row("Velocity (1h)", str(b['velocity_1h']), Colors.SUCCESS if b['velocity_1h'] < 5 else Colors.WARNING)}
{_kv_row("Merchant Risk", f"{b['merchant_risk']:.2f}", Colors.SUCCESS if b['merchant_risk'] < 0.3 else Colors.WARNING)}
{_kv_row("Beneficiary Novelty", f"{b['beneficiary_novelty']:.0%}", Colors.SUCCESS if b['beneficiary_novelty'] < 0.3 else Colors.WARNING)}
{_kv_row("Impossible Travel", "Yes" if b['impossible_travel'] else "No", Colors.CRITICAL if b['impossible_travel'] else Colors.SUCCESS)}"""

            col1_html = (
                _card_html("Transaction Payload", txn_json_html, icon="FILE_TEXT")
                + _card_html("Feature Summary", feat_content, icon="DATABASE")
                + _card_html("Behavior Summary", beh_content, icon="ACTIVITY")
            )
            _iframe(col1_html, height=900)

        # ── Column 2: Animated Pipeline ───────────────────────────────────
        with col2:
            col2_html = _build_pipeline_html(result, anim_step=len(result.steps))
            _iframe(col2_html, height=900)

        # ── Column 3: Results ─────────────────────────────────────────────
        with col3:
            decision_color = _decision_color(result.decision)
            risk_color = _risk_color(result.risk_score)

            kpi_html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
    <div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:12px;padding:16px;text-align:center">
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:6px">Fraud Probability</div>
        <div style="font-size:{Typography.TEXT_3XL};font-weight:{Typography.WEIGHT_BOLD};color:{risk_color}">{result.fraud_probability:.2%}</div>
    </div>
    <div style="background:{Colors.BG_CARD};border:1px solid {decision_color};border-radius:12px;padding:16px;text-align:center">
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:6px">Decision</div>
        <div style="font-size:{Typography.TEXT_3XL};font-weight:{Typography.WEIGHT_BOLD};color:{decision_color}">{result.decision}</div>
    </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">
    <div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Latency</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{result.total_latency_ms} ms</div>
    </div>
    <div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Phase</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{result.selected_phase}</div>
    </div>
    <div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">Confidence</div>
        <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{result.confidence:.1%}</div>
    </div>
</div>"""

            phase_colors = {1: Colors.PHASE_1, 2: Colors.PHASE_2, 3: Colors.PHASE_3}
            prediction_html = _card_html(
                "Prediction Source",
                _kv_row("Model", result.selected_model, phase_colors.get(result.selected_phase, Colors.ACCENT)),
                icon="CPU",
            )

            # Build model routing HTML inline
            routing_html = _build_routing_html(result)

            col3_html = kpi_html + prediction_html + routing_html
            _iframe(col3_html, height=900)

        # ── Full-Width Cards Below ────────────────────────────────────────
        st.markdown(f'<div style="height:8px"></div>', unsafe_allow_html=True)

        full_row1, full_row2 = st.columns(2)
        with full_row1:
            left_html = (
                _build_behavior_html(result)
                + _build_explainability_html(result)
            )
            _iframe(left_html, height=800)
        with full_row2:
            if result.selected_phase == 1:
                right_html = _build_cold_start_html(result)
            elif result.selected_phase == 2:
                right_html = _build_adaptive_html(result)
            else:
                right_html = _build_supervised_html(result)
            right_html += _build_infra_html(result)
            _iframe(right_html, height=800)

        # Clear running state
        st.session_state["sim_running"] = False

    elif not run_simulation:
        # Empty state
        st.markdown(
            f"""
<div style="text-align:center;padding:80px 24px;color:{Colors.TEXT_MUTED}">
    {Icons.html('ZAP', 48, Colors.TEXT_MUTED)}
    <div style="font-size:{Typography.TEXT_XL};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_SECONDARY};margin-top:16px;margin-bottom:8px">Ready to Simulate</div>
    <div style="font-size:{Typography.TEXT_BASE}">Select a scenario and click <strong>Run Simulation</strong> to visualize the complete FraudTrap inference pipeline.</div>
</div>
""",
            unsafe_allow_html=True,
        )
