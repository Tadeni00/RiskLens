"""
FraudTrap — Transaction Schema
Flexible Pydantic schema that accepts any client payload.
Mandatory fields are the minimum required to compute velocity features.
All other fields enrich the feature vector but are optional.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
import uuid

# ── Mandatory core (every client MUST provide these) ─────────────────────────


class TransactionRequest(BaseModel):
    """
    Inbound scoring payload from client bank / fintech.
    Mandatory fields are the absolute minimum for real-time scoring.
    Optional fields improve accuracy — clients are encouraged to send all they have.
    """

    # Identity
    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transaction identifier (idempotency key)",
    )
    tenant_id: str = Field(..., description="Client bank / fintech identifier")
    account_id: str = Field(..., description="Tokenised account identifier")
    session_id: Optional[str] = Field(None, description="Tokenised session identifier")

    # Transaction core
    amount: float = Field(..., gt=0, description="Transaction amount in local currency")
    currency: str = Field(..., min_length=3, max_length=3, description="ISO 4217 currency code")
    timestamp: datetime = Field(..., description="Transaction initiation time (UTC)")
    transaction_type: str = Field(
        ...,
        description="Type: PAYMENT, TRANSFER, WITHDRAWAL, TOP_UP, REFUND, LOAN_DISBURSEMENT",
    )
    channel: str = Field(..., description="Channel: WEB, MOBILE, API, POS, ATM, USSD")

    # Counterparty (optional but high-signal)
    merchant_id: Optional[str] = Field(None, description="Tokenised merchant identifier")
    merchant_category_code: Optional[str] = Field(None, description="MCC code")
    merchant_country: Optional[str] = Field(None, description="ISO 3166-1 alpha-2")
    counterparty_account_id: Optional[str] = Field(
        None, description="Tokenised destination account (for transfers)"
    )

    # Device / network (optional but high-signal for ATO)
    device_id: Optional[str] = Field(None, description="Tokenised device fingerprint")
    device_type: Optional[str] = Field(None, description="MOBILE, DESKTOP, TABLET, POS_TERMINAL")
    ip_address_hash: Optional[str] = Field(None, description="Hashed IP address (privacy-safe)")
    user_agent_hash: Optional[str] = Field(None, description="Hashed user-agent string")

    # Geolocation (optional)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    country_code: Optional[str] = Field(None, description="ISO 3166-1 alpha-2")

    # Behavioural biometrics (optional — from SDK)
    typing_cadence_ms: Optional[float] = Field(
        None, description="Mean inter-keystroke interval in ms"
    )
    session_duration_seconds: Optional[float] = Field(None)
    field_visit_count: Optional[int] = Field(None, description="Number of form fields visited")

    # Client-specific extensions
    extra_fields: Optional[dict[str, Any]] = Field(
        None,
        description="Pass-through dict for client-specific fields — stored but not used in base model",
    )

    @field_validator("currency")
    @classmethod
    def currency_upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("transaction_type", "channel")
    @classmethod
    def to_upper(cls, v: str) -> str:
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_abc123",
                "tenant_id": "bank_ng_gtb",
                "account_id": "tok_acct_xyz789",
                "amount": 45000.00,
                "currency": "NGN",
                "timestamp": "2025-01-15T14:23:11Z",
                "transaction_type": "PAYMENT",
                "channel": "MOBILE",
                "merchant_id": "tok_merch_456",
                "merchant_category_code": "5411",
                "device_id": "tok_dev_abc",
                "country_code": "NG",
            }
        }


# ── Scoring response ──────────────────────────────────────────────────────────


class FeatureContribution(BaseModel):
    """Individual feature contribution to the score."""

    feature: str = Field(description="Feature name")
    value: float = Field(description="Feature value for this transaction")
    contribution: float = Field(description="Contribution to risk score (can be negative)")
    method: str = Field(
        description="Attribution method: shap, rule_weight, vae_recon, iforest_path, tail_zscore, calibration"
    )


class Explanation(BaseModel):
    """
    Unified explanation format for all model types.
    Supports SHAP (supervised), rule weights (rules), component breakdown (cold-start),
    and calibration breakdown (semi-supervised).
    Enhanced with counterfactual and confidence data from the ExplainabilityEngine.
    """

    model_type: str = Field(
        description="Model type: rules, cold_start, adaptive_learning, supervised, champion, explainability"
    )
    base_value: float = Field(description="Model baseline (expected value)")
    prediction_value: float = Field(description="Final risk score")
    top_features: list[FeatureContribution] = Field(
        description="Top N features/components by absolute contribution"
    )
    components: Optional[dict[str, float]] = Field(
        default=None,
        description="Phase-specific component breakdown (e.g., vae/iforest/tail for cold-start)",
    )
    latency_ms: float = Field(description="Time to compute explanation")
    # Extended fields from ExplainabilityEngine
    confidence: Optional[dict[str, Any]] = Field(
        default=None,
        description="Model confidence metadata (expert_used, confidence, ft_invoked)",
    )
    counterfactual: Optional[dict[str, Any]] = Field(
        default=None,
        description="Counterfactual explanation (nearest_neighbor or dice)",
    )
    formatted_report: Optional[dict[str, Any]] = Field(
        default=None, description="Analyst-friendly formatted report"
    )


class ScoringResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    tenant_id: str
    risk_score: float = Field(ge=0.0, le=1.0, description="Fraud probability 0–1")
    decision: str = Field(description="APPROVE | REVIEW | BLOCK")
    model_phase: str = Field(description="UNSUPERVISED | ADAPTIVE_LEARNING | SUPERVISED")
    model_version: str
    latency_ms: float
    explanation: Optional[Explanation] = None
    explanation_type: Optional[str] = Field(default=None, description="sync | async_pending | none")
    triggered_rules: list[str] = Field(default_factory=list)
    trace_id: str = Field(description="Unique trace for audit log lookup")
    scored_at: datetime


# ── Label ingestion ───────────────────────────────────────────────────────────


class LabelPayload(BaseModel):
    """Ground-truth label arriving from chargeback or manual review."""

    transaction_id: str
    tenant_id: str
    label: int = Field(..., ge=0, le=1, description="1=fraud, 0=legitimate")
    label_source: str = Field(description="CHARGEBACK | MANUAL_REVIEW | DISPUTE_RESOLVED")
    chargeback_reason_code: Optional[str] = Field(
        None, description="Visa/MC reason code — used to filter non-fraud chargebacks"
    )
    labelled_at: datetime
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
