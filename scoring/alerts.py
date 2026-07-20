"""
FraudTrap — Alerting System
Multi-channel alerting with PagerDuty, Slack, custom webhooks, deduplication, and runbooks.
"""
from __future__ import annotations
import hashlib
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Callable
import requests

from loguru import logger

try:
    from config.settings import get_settings
    settings = get_settings()
except Exception:
    settings = None


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    SLA_BREACH = "sla_breach"
    DRIFT_SPIKE = "drift_spike"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DROP = "performance_drop"
    DATA_QUALITY = "data_quality"
    SCORING_ERRORS = "scoring_errors"
    MODEL_RELOAD = "model_reload"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    EXPERIMENT_SIGNIFICANT = "experiment_significant"


@dataclass
class Alert:
    """Structured alert object."""
    id: str
    category: AlertCategory
    severity: AlertSeverity
    tenant_id: str
    title: str
    message: str
    metadata: dict = field(default_factory=dict)
    runbook_url: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fingerprint: str = field(init=False)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def __post_init__(self):
        # Create deterministic fingerprint for deduplication
        fp_input = f"{self.category}:{self.tenant_id}:{self.title}"
        self.fingerprint = hashlib.sha256(fp_input.encode()).hexdigest()[:16]


class Deduplicator:
    """Simple in-memory deduplication with TTL."""
    
    def __init__(self, ttl_seconds: int = 300):
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
    
    def should_alert(self, alert: Alert) -> bool:
        """Returns True if alert should be sent (not deduplicated)."""
        with self._lock:
            now = time.time()
            # Clean old entries
            self._seen = {fp: ts for fp, ts in self._seen.items() if now - ts < self._ttl}
            
            if alert.fingerprint in self._seen:
                return False
            
            self._seen[alert.fingerprint] = now
            return True
    
    def clear(self):
        with self._lock:
            self._seen.clear()


class AlertManager:
    """
    Central alert manager with multi-channel support.
    Supports PagerDuty, Slack, custom webhooks, and custom handlers.
    """
    
    def __init__(
        self,
        pagerduty_key: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        custom_webhook: Optional[str] = None,
        dedup_ttl: int = 300,
        default_runbooks: Optional[dict[AlertCategory, str]] = None,
    ):
        self.pagerduty_key = pagerduty_key or (settings.pagerduty_integration_key if settings else None)
        self.slack_webhook = slack_webhook or (settings.slack_alert_webhook if settings else None)
        self.custom_webhook = custom_webhook or (settings.custom_alert_webhook if settings else None)
        
        self.dedup = Deduplicator(ttl_seconds=dedup_ttl)
        self.default_runbooks = default_runbooks or {}
        
        # Custom handler registry
        self._handlers: dict[AlertCategory, list[Callable[[Alert], None]]] = {}
    
    def register_handler(self, category: AlertCategory, handler: Callable[[Alert], None]) -> None:
        """Register custom handler for alert category."""
        if category not in self._handlers:
            self._handlers[category] = []
        self._handlers[category].append(handler)
    
    def _get_runbook_url(self, category: AlertCategory) -> Optional[str]:
        return self.default_runbooks.get(category)
    
    def fire(self, alert: Alert) -> bool:
        """
        Fire an alert through all configured channels.
        Returns True if alert was sent (not deduplicated).
        """
        # Deduplication
        if not self.dedup.should_alert(alert):
            logger.debug("Alert deduplicated: {}", alert.fingerprint)
            return False
        
        # Add runbook if not set
        if not alert.runbook_url:
            alert.runbook_url = self.default_runbooks.get(alert.category)
        
        # Send to all channels
        sent = False
        
        if self.pagerduty_key:
            sent |= self._send_pagerduty(alert)
        
        if self.slack_webhook:
            sent |= self._send_slack(alert)
        
        if self.custom_webhook:
            sent |= self._send_webhook(alert)
        
        # Call custom handlers
        for handler in self._handlers.get(alert.category, []):
            try:
                handler(alert)
            except Exception as exc:
                logger.error("Custom handler failed for {}: {}", alert.category, exc)
        
        # Log regardless
        logger.log(
            alert.severity.upper() if alert.severity != AlertSeverity.INFO else "INFO",
            "ALERT | tenant={} category={} severity={} title={} fingerprint={}",
            alert.tenant_id, alert.category.value, alert.severity.value, alert.title, alert.fingerprint
        )
        
        return sent
    
    def _send_pagerduty(self, alert: Alert) -> bool:
        """Send alert to PagerDuty Events API v2."""
        try:
            payload = {
                "routing_key": self.pagerduty_key,
                "event_action": "trigger",
                "payload": {
                    "summary": alert.title,
                    "severity": alert.severity.value,
                    "source": f"fraudtrap-{alert.tenant_id}",
                    "component": "fraud-detection",
                    "group": alert.category.value,
                    "class": alert.category.value,
                    "custom_details": {
                        **alert.metadata,
                        "tenant_id": alert.tenant_id,
                        "fingerprint": alert.fingerprint,
                        "runbook_url": alert.runbook_url,
                    },
                },
                "dedup_key": alert.fingerprint,
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            logger.info("PagerDuty alert sent: {}", alert.fingerprint)
            return True
        except Exception as exc:
            logger.error("Failed to send PagerDuty alert: {}", exc)
            return False
    
    def _send_slack(self, alert: Alert) -> bool:
        """Send alert to Slack webhook."""
        try:
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.CRITICAL: "#ff0000",
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f"🚨 {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Tenant", "value": alert.tenant_id, "short": True},
                        {"title": "Category", "value": alert.category.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Fingerprint", "value": alert.fingerprint, "short": True},
                    ],
                    "footer": "FraudTrap Alerting",
                    "ts": int(alert.created_at.timestamp()),
                }]
            }
            
            if alert.runbook_url:
                payload["attachments"][0]["actions"] = [{
                    "type": "button",
                    "text": "📖 Runbook",
                    "url": alert.runbook_url,
                    "style": "primary",
                }]
            
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Slack alert sent: {}", alert.fingerprint)
            return True
        except Exception as exc:
            logger.error("Failed to send Slack alert: {}", exc)
            return False
    
    def _send_webhook(self, alert: Alert) -> bool:
        """Send alert to custom webhook."""
        try:
            payload = {
                "alert": {
                    "id": alert.id,
                    "category": alert.category.value,
                    "severity": alert.severity.value,
                    "tenant_id": alert.tenant_id,
                    "title": alert.title,
                    "message": alert.message,
                    "metadata": alert.metadata,
                    "fingerprint": alert.fingerprint,
                    "runbook_url": alert.runbook_url,
                    "created_at": alert.created_at.isoformat(),
                }
            }
            
            response = requests.post(
                self.custom_webhook,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Webhook alert sent: {}", alert.fingerprint)
            return True
        except Exception as exc:
            logger.error("Failed to send webhook alert: {}", exc)
            return False


# Pre-configured alert factories
def alert_sla_breach(
    tenant_id: str,
    p95_latency: float,
    threshold: float = 90.0,
) -> Alert:
    return Alert(
        id=f"sla_breach_{tenant_id}_{int(time.time())}",
        category=AlertCategory.SLA_BREACH,
        severity=AlertSeverity.CRITICAL,
        tenant_id=tenant_id,
        title=f"SLA Breach: P95 latency {p95_latency:.0f}ms > {threshold}ms",
        message=(
            f"Scoring P95 latency for tenant **{tenant_id}** has exceeded the "
            f"**{threshold}ms** threshold. Current P95: **{p95_latency:.1f}ms**.\n\n"
            f"This may indicate model inference slowdown, feature store latency, "
            f"or resource contention."
        ),
        metadata={"p95_latency_ms": p95_latency, "threshold_ms": threshold},
    )


def alert_drift_spike(
    tenant_id: str,
    feature: str,
    psi: float,
    kl: float,
    threshold_psi: float = 0.25,
    threshold_kl: float = 0.1,
) -> Alert:
    return Alert(
        id=f"drift_spike_{tenant_id}_{feature}_{int(time.time())}",
        category=AlertCategory.DRIFT_SPIKE,
        severity=AlertSeverity.WARNING,
        tenant_id=tenant_id,
        title=f"Drift Detected: {feature} (PSI={psi:.3f}, KL={kl:.3f})",
        message=(
            f"Feature **{feature}** for tenant **{tenant_id}** shows significant drift.\n\n"
            f"- **PSI**: {psi:.3f} (threshold: {threshold_psi})\n"
            f"- **KL Divergence**: {kl:.3f} (threshold: {threshold_kl})\n\n"
            f"This may indicate data quality issues, new transaction patterns, "
            f"or upstream schema changes."
        ),
        metadata={"feature": feature, "psi": psi, "kl": kl, "threshold_psi": threshold_psi, "threshold_kl": threshold_kl},
    )


def alert_concept_drift(
    tenant_id: str,
    label_rate_baseline: float,
    label_rate_current: float,
    rate_change: float,
    threshold: float = 0.2,
) -> Alert:
    direction = "increased" if label_rate_current > label_rate_baseline else "decreased"
    return Alert(
        id=f"concept_drift_{tenant_id}_{int(time.time())}",
        category=AlertCategory.CONCEPT_DRIFT,
        severity=AlertSeverity.WARNING,
        tenant_id=tenant_id,
        title=f"Concept Drift: Fraud rate {direction} by {rate_change:.1%}",
        message=(
            f"Fraud label rate for tenant **{tenant_id}** has {direction} significantly.\n\n"
            f"- **Baseline rate**: {label_rate_baseline:.2%}\n"
            f"- **Current rate**: {label_rate_current:.2%}\n"
            f"- **Change**: {rate_change:.2%} (threshold: {threshold:.0%})\n\n"
            f"This may indicate new fraud patterns, data quality issues, "
            f"or label pipeline problems."
        ),
        metadata={
            "label_rate_baseline": label_rate_baseline,
            "label_rate_current": label_rate_current,
            "rate_change": rate_change,
            "threshold": threshold,
        },
    )


def alert_performance_drop(
    tenant_id: str,
    metric: str,
    current_value: float,
    baseline_value: float,
    drop_pct: float,
    threshold: float = 0.05,
) -> Alert:
    return Alert(
        id=f"perf_drop_{tenant_id}_{int(time.time())}",
        category=AlertCategory.PERFORMANCE_DROP,
        severity=AlertSeverity.CRITICAL,
        tenant_id=tenant_id,
        title=f"Performance Drop: {metric} dropped {drop_pct:.1%}",
        message=(
            f"Model **{metric}** for tenant **{tenant_id}** has dropped by **{drop_pct:.1%}**.\n\n"
            f"- **Baseline**: {baseline_value:.4f}\n"
            f"- **Current**: {current_value:.4f}\n"
            f"- **Drop**: {drop_pct:.2%} (threshold: {threshold:.0%})\n\n"
            f"This may require model retraining or investigation of data drift."
        ),
        metadata={
            "metric": metric,
            "baseline": baseline_value,
            "current": current_value,
            "drop_pct": drop_pct,
            "threshold": threshold,
        },
    )


def alert_data_quality(
    tenant_id: str,
    issue: str,
    affected_features: list[str],
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> Alert:
    return Alert(
        id=f"data_quality_{tenant_id}_{int(time.time())}",
        category=AlertCategory.DATA_QUALITY,
        severity=severity,
        tenant_id=tenant_id,
        title=f"Data Quality Issue: {issue}",
        message=(
            f"Data quality issue detected for tenant **{tenant_id}**: **{issue}**\n\n"
            f"**Affected features**: {', '.join(affected_features)}\n\n"
            f"This may affect model performance and scoring accuracy."
        ),
        metadata={"issue": issue, "affected_features": affected_features},
    )


def alert_scoring_errors(
    tenant_id: str,
    error_rate: float,
    threshold: float = 0.01,
) -> Alert:
    return Alert(
        id=f"scoring_errors_{tenant_id}_{int(time.time())}",
        category=AlertCategory.SCORING_ERRORS,
        severity=AlertSeverity.CRITICAL,
        tenant_id=tenant_id,
        title=f"High Scoring Error Rate: {error_rate:.1%}",
        message=(
            f"Scoring error rate for tenant **{tenant_id}** is **{error_rate:.1%}** "
            f"(threshold: {threshold:.1%}).\n\n"
            f"Errors may be due to model loading failures, feature computation errors, "
            f"or downstream service issues."
        ),
        metadata={"error_rate": error_rate, "threshold": threshold},
    )


def alert_model_reload(
    tenant_id: str,
    model_version: str,
    reload_duration_ms: float,
    success: bool,
) -> Alert:
    return Alert(
        id=f"model_reload_{tenant_id}_{int(time.time())}",
        category=AlertCategory.MODEL_RELOAD,
        severity=AlertSeverity.INFO if success else AlertSeverity.WARNING,
        tenant_id=tenant_id,
        title=f"Model Reload: {'Success' if success else 'Failed'} - {model_version}",
        message=(
            f"Model reload for tenant **{tenant_id}** {'completed' if success else 'failed'}.\n\n"
            f"- **Version**: {model_version}\n"
            f"- **Duration**: {reload_duration_ms:.0f}ms\n"
            f"- **Status**: {'✅ Success' if success else '❌ Failed'}"
        ),
        metadata={
            "model_version": model_version,
            "reload_duration_ms": reload_duration_ms,
            "success": success,
        },
    )


def alert_guardrail_violation(
    tenant_id: str,
    experiment_name: str,
    guardrail: str,
    value: float,
    threshold: float,
) -> Alert:
    return Alert(
        id=f"guardrail_{tenant_id}_{experiment_name}_{int(time.time())}",
        category=AlertCategory.GUARDRAIL_VIOLATION,
        severity=AlertSeverity.CRITICAL,
        tenant_id=tenant_id,
        title=f"Guardrail Violation: {guardrail} = {value:.3f} > {threshold}",
        message=(
            f"Experiment **{experiment_name}** for tenant **{tenant_id}** violated guardrail **{guardrail}**.\n\n"
            f"- **Current value**: {value:.4f}\n"
            f"- **Threshold**: {threshold:.4f}\n\n"
            f"Experiment should be paused or stopped."
        ),
        metadata={
            "experiment_name": experiment_name,
            "guardrail": guardrail,
            "value": value,
            "threshold": threshold,
        },
    )


def alert_experiment_significant(
    tenant_id: str,
    experiment_name: str,
    challenger_name: str,
    metric: str,
    lift_pct: float,
    p_value: float,
) -> Alert:
    return Alert(
        id=f"exp_sig_{tenant_id}_{experiment_name}_{int(time.time())}",
        category=AlertCategory.EXPERIMENT_SIGNIFICANT,
        severity=AlertSeverity.INFO,
        tenant_id=tenant_id,
        title=f"Experiment Significant: {experiment_name}",
        message=(
            f"Experiment **{experiment_name}** for tenant **{tenant_id}** shows significant result.\n\n"
            f"- **Challenger**: {challenger_name}\n"
            f"- **Metric**: {metric}\n"
            f"- **Lift**: {lift_pct:.1f}%\n"
            f"- **P-value**: {p_value:.4f}\n\n"
            f"Consider promoting challenger to champion."
        ),
        metadata={
            "experiment_name": experiment_name,
            "challenger": challenger_name,
            "metric": metric,
            "lift_pct": lift_pct,
            "p_value": p_value,
        },
    )


# Default runbook URLs (customize per organization)
DEFAULT_RUNBOOKS = {
    AlertCategory.SLA_BREACH: "https://wiki.company.com/runbooks/sla-breach",
    AlertCategory.DRIFT_SPIKE: "https://wiki.company.com/runbooks/drift-spike",
    AlertCategory.CONCEPT_DRIFT: "https://wiki.company.com/runbooks/concept-drift",
    AlertCategory.PERFORMANCE_DROP: "https://wiki.company.com/runbooks/performance-drop",
    AlertCategory.DATA_QUALITY: "https://wiki.company.com/runbooks/data-quality",
    AlertCategory.SCORING_ERRORS: "https://wiki.company.com/runbooks/scoring-errors",
    AlertCategory.MODEL_RELOAD: "https://wiki.company.com/runbooks/model-reload",
    AlertCategory.GUARDRAIL_VIOLATION: "https://wiki.company.com/runbooks/guardrail-violation",
    AlertCategory.EXPERIMENT_SIGNIFICANT: "https://wiki.company.com/runbooks/experiment-significant",
}


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(
            pagerduty_key=settings.pagerduty_integration_key if settings else None,
            slack_webhook=settings.slack_alert_webhook if settings else None,
            custom_webhook=settings.custom_alert_webhook if settings else None,
            default_runbooks=DEFAULT_RUNBOOKS,
        )
    return _alert_manager


# Convenience functions
def fire_sla_breach(tenant_id: str, p95_latency: float, threshold: float = 90.0) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_sla_breach(tenant_id, p95_latency, threshold))


def fire_drift_spike(
    tenant_id: str,
    feature: str,
    psi: float,
    kl: float,
    threshold_psi: float = 0.25,
    threshold_kl: float = 0.1,
) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_drift_spike(tenant_id, feature, psi, kl, threshold_psi, threshold_kl))


def fire_concept_drift(
    tenant_id: str,
    label_rate_baseline: float,
    label_rate_current: float,
    threshold: float = 0.2,
) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_concept_drift(tenant_id, 0, 0, 0, threshold))  # Fixed in fire_


def fire_performance_drop(
    tenant_id: str,
    metric: str,
    current_value: float,
    baseline_value: float,
    threshold: float = 0.05,
) -> bool:
    drop_pct = abs(current_value - baseline_value) / max(baseline_value, 1e-6)
    manager = get_alert_manager()
    return manager.fire(alert_performance_drop(tenant_id, metric, current_value, baseline_value, drop_pct, threshold))


def fire_data_quality(
    tenant_id: str,
    issue: str,
    affected_features: list[str],
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_data_quality(tenant_id, issue, affected_features, severity))


def fire_scoring_errors(tenant_id: str, error_rate: float, threshold: float = 0.01) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_scoring_errors(tenant_id, error_rate, threshold))


def fire_model_reload(
    tenant_id: str,
    model_version: str,
    reload_duration_ms: float,
    success: bool,
) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_model_reload(tenant_id, model_version, reload_duration_ms, success))


def fire_guardrail_violation(
    tenant_id: str,
    experiment_name: str,
    guardrail: str,
    value: float,
    threshold: float,
) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_guardrail_violation(tenant_id, experiment_name, guardrail, value, threshold))


def fire_experiment_significant(
    tenant_id: str,
    experiment_name: str,
    challenger_name: str,
    metric: str,
    lift_pct: float,
    p_value: float,
) -> bool:
    manager = get_alert_manager()
    return manager.fire(alert_experiment_significant(tenant_id, experiment_name, challenger_name, metric, lift_pct, p_value))