"""
FraudTrap — Rules DSL Configuration
Defines the YAML/JSON schema for declarative rule definitions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class RuleType(str, Enum):
    BLOCKLIST = "blocklist"
    THRESHOLD = "threshold"
    EXPRESSION = "expression"
    VELOCITY = "velocity"
    GEO = "geo"


class RuleAction(str, Enum):
    HARD_BLOCK = "hard_block"
    SOFT_BOOST = "soft_boost"
    REVIEW = "review"
    TAG = "tag"


class RuleSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Operator(str, Enum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


@dataclass
class BlocklistConfig:
    """Configuration for blocklist rule."""

    entity: Literal["account", "device", "ip", "merchant", "country"]
    list_name: str  # Redis set name (e.g., "fraud_accounts")


@dataclass
class ThresholdConfig:
    """Configuration for threshold rule."""

    feature: str
    operator: Operator
    threshold: float


@dataclass
class ExpressionConfig:
    """Configuration for expression rule."""

    expression: str  # e.g., "acct_v_24h_count < 5 and amount_vs_mean_ratio > 10"


@dataclass
class VelocityConfig:
    """Configuration for velocity rule."""

    entity_type: Literal["account", "device", "ip"]
    window_seconds: int
    threshold: int
    operator: Operator = Operator.GT


@dataclass
class GeoConfig:
    """Configuration for geo-based rule."""

    rule_type: Literal["impossible_travel", "cross_border", "sanctioned_country"]
    threshold_kmh: float = 900.0  # For impossible travel


@dataclass
class RuleDefinition:
    """
    Complete rule definition.

    Can be loaded from YAML/JSON configuration.
    """

    id: str
    description: str
    type: RuleType
    action: RuleAction
    severity: RuleSeverity = RuleSeverity.HIGH

    # Config by type (only one should be set)
    blocklist: Optional[BlocklistConfig] = None
    threshold: Optional[ThresholdConfig] = None
    expression: Optional[ExpressionConfig] = None
    velocity: Optional[VelocityConfig] = None
    geo: Optional[GeoConfig] = None

    # Boost/override settings
    boost: float = 0.05
    max_boost: float = 0.20
    score_override: Optional[float] = None

    # Metadata
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = RuleType(self.type)
        if isinstance(self.action, str):
            self.action = RuleAction(self.action)
        if isinstance(self.severity, str):
            self.severity = RuleSeverity(self.severity)

    def to_dict(self) -> dict:
        """Serialize to dictionary for YAML/JSON output."""
        result = {
            "id": self.id,
            "description": self.description,
            "type": self.type.value,
            "action": self.action.value,
            "severity": self.severity.value,
            "boost": self.boost,
            "max_boost": self.max_boost,
            "enabled": self.enabled,
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.score_override is not None:
            result["score_override"] = self.score_override

        if self.blocklist:
            result["blocklist"] = {
                "entity": self.blocklist.entity,
                "list_name": self.blocklist.list_name,
            }
        if self.threshold:
            result["threshold"] = {
                "feature": self.threshold.feature,
                "operator": self.threshold.operator.value,
                "threshold": self.threshold.threshold,
            }
        if self.expression:
            result["expression"] = {"expression": self.expression.expression}
        if self.velocity:
            result["velocity"] = {
                "entity_type": self.velocity.entity_type,
                "window_seconds": self.velocity.window_seconds,
                "threshold": self.velocity.threshold,
                "operator": self.velocity.operator.value,
            }
        if self.geo:
            result["geo"] = {
                "rule_type": self.geo.rule_type,
                "threshold_kmh": self.geo.threshold_kmh,
            }

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "RuleDefinition":
        """Deserialize from dictionary."""
        # Parse nested configs
        if "blocklist" in data:
            data["blocklist"] = BlocklistConfig(**data["blocklist"])
        if "threshold" in data:
            data["threshold"] = ThresholdConfig(**data["threshold"])
        if "expression" in data:
            data["expression"] = ExpressionConfig(**data["expression"])
        if "velocity" in data:
            data["velocity"] = VelocityConfig(**data["velocity"])
        if "geo" in data:
            data["geo"] = GeoConfig(**data["geo"])

        return cls(**data)


# Default ruleset (used when no config file provided)
DEFAULT_RULESET = [
    RuleDefinition(
        id="BLOCKLIST_ACCOUNT",
        description="Account on fraud blocklist",
        type=RuleType.BLOCKLIST,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.CRITICAL,
        blocklist=BlocklistConfig(entity="account", list_name="fraud_accounts"),
    ),
    RuleDefinition(
        id="BLOCKLIST_DEVICE",
        description="Device on fraud blocklist",
        type=RuleType.BLOCKLIST,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.CRITICAL,
        blocklist=BlocklistConfig(entity="device", list_name="fraud_devices"),
    ),
    RuleDefinition(
        id="BLOCKLIST_IP",
        description="IP hash on fraud blocklist",
        type=RuleType.BLOCKLIST,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.CRITICAL,
        blocklist=BlocklistConfig(entity="ip", list_name="fraud_ips"),
    ),
    RuleDefinition(
        id="BLOCKLIST_MERCHANT",
        description="Merchant on fraud blocklist",
        type=RuleType.BLOCKLIST,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.CRITICAL,
        blocklist=BlocklistConfig(entity="merchant", list_name="fraud_merchants"),
    ),
    RuleDefinition(
        id="SANCTIONED_COUNTRY",
        description="Transaction country on sanctions list",
        type=RuleType.BLOCKLIST,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.CRITICAL,
        blocklist=BlocklistConfig(entity="country", list_name="sanctioned_countries"),
    ),
    RuleDefinition(
        id="IMPOSSIBLE_TRAVEL",
        description="Geo velocity > 900 km/h (impossible travel)",
        type=RuleType.GEO,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.CRITICAL,
        geo=GeoConfig(rule_type="impossible_travel", threshold_kmh=900.0),
    ),
    RuleDefinition(
        id="VELOCITY_SPIKE_1M",
        description="Velocity spike: > 10 transactions in 1 minute",
        type=RuleType.VELOCITY,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.HIGH,
        velocity=VelocityConfig(
            entity_type="account",
            window_seconds=60,
            threshold=10,
            operator=">",
        ),
    ),
    RuleDefinition(
        id="NEW_ACCT_HIGH_VALUE",
        description="New account (< 5 txns/24h) with amount > 10x mean",
        type=RuleType.EXPRESSION,
        action=RuleAction.HARD_BLOCK,
        severity=RuleSeverity.HIGH,
        expression=ExpressionConfig(
            expression="acct_v_24h_count < 5 and amount_vs_mean_ratio > 10"
        ),
    ),
    RuleDefinition(
        id="ROUND_AMT_BURST",
        description="Round-amount burst (card testing): 5+ round amounts in 5 min",
        type=RuleType.EXPRESSION,
        action=RuleAction.SOFT_BOOST,
        severity=RuleSeverity.MEDIUM,
        expression=ExpressionConfig(
            expression="is_round_amount == 1 and acct_v_5m_count >= 5"
        ),
        boost=0.15,
    ),
    RuleDefinition(
        id="HIGH_RISK_CHANNEL",
        description="High-risk channel (API, USSD) with large amount",
        type=RuleType.EXPRESSION,
        action=RuleAction.SOFT_BOOST,
        severity=RuleSeverity.MEDIUM,
        expression=ExpressionConfig(
            expression="channel_enc in [2, 5] and amount > 100000"
        ),
        boost=0.10,
    ),
    RuleDefinition(
        id="NEW_DEVICE_HIGH_VALUE",
        description="New device with amount > 5x account mean",
        type=RuleType.EXPRESSION,
        action=RuleAction.SOFT_BOOST,
        severity=RuleSeverity.MEDIUM,
        expression=ExpressionConfig(
            expression="is_new_device == 1 and amount_vs_mean_ratio > 5"
        ),
        boost=0.12,
    ),
    RuleDefinition(
        id="CROSS_BORDER_NEW_MERCHANT",
        description="Cross-border transaction with new merchant",
        type=RuleType.EXPRESSION,
        action=RuleAction.SOFT_BOOST,
        severity=RuleSeverity.MEDIUM,
        expression=ExpressionConfig(
            expression="cross_country_flag == 1 and is_new_merchant == 1"
        ),
        boost=0.08,
    ),
    RuleDefinition(
        id="VELOCITY_SPIKE_5M",
        description="Velocity spike: > 20 transactions in 5 minutes",
        type=RuleType.VELOCITY,
        action=RuleAction.SOFT_BOOST,
        severity=RuleSeverity.MEDIUM,
        velocity=VelocityConfig(
            entity_type="account",
            window_seconds=300,
            threshold=20,
            operator=">",
        ),
        boost=0.10,
    ),
    RuleDefinition(
        id="CROSS_BORDER_HIGH_VALUE",
        description="Cross-border transaction > 50x account mean",
        type=RuleType.EXPRESSION,
        action=RuleAction.SOFT_BOOST,
        severity=RuleSeverity.HIGH,
        expression=ExpressionConfig(
            expression="cross_country_flag == 1 and amount_vs_mean_ratio > 50"
        ),
        boost=0.20,
    ),
]


# Default rules file path (relative to project root)
DEFAULT_RULES_PATH = "config/rules.yaml"


def load_ruleset_from_file(path: str = None) -> list[RuleDefinition]:
    """Load ruleset from YAML or JSON file. Uses default path if not provided."""
    import yaml
    import json
    from pathlib import Path

    path = Path(path or DEFAULT_RULES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            import json

            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    rules = []
    for rule_data in data.get("rules", []):
        rules.append(RuleDefinition.from_dict(rule_data))

    return rules


def save_ruleset_to_file(rules: list[RuleDefinition], path: str = None) -> None:
    """Save ruleset to YAML or JSON file. Uses default path if not provided."""
    import yaml
    import json
    from pathlib import Path

    path = Path(path or DEFAULT_RULES_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rules": [r.to_dict() for r in rules]}

    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


# Example YAML configuration
EXAMPLE_YAML = """
rules:
  - id: "BLOCKLIST_ACCOUNT"
    description: "Account on fraud blocklist"
    type: "blocklist"
    action: "hard_block"
    severity: "critical"
    blocklist:
      entity: "account"
      list_name: "fraud_accounts"
  
  - id: "IMPOSSIBLE_TRAVEL"
    description: "Geo velocity > 900 km/h"
    type: "geo"
    action: "hard_block"
    severity: "critical"
    geo:
      rule_type: "impossible_travel"
      threshold_kmh: 900.0
  
  - id: "VELOCITY_SPIKE_1M"
    description: "Velocity spike: > 10 txns in 1 min"
    type: "velocity"
    action: "hard_block"
    severity: "high"
    velocity:
      entity_type: "account"
      window_seconds: 60
      threshold: 10
      operator: ">"
  
  - id: "NEW_ACCT_HIGH_VALUE"
    description: "New account with high value"
    type: "expression"
    action: "hard_block"
    severity: "high"
    expression:
      expression: "acct_v_24h_count < 5 and amount_vs_mean_ratio > 10"
  
  - id: "ROUND_AMT_BURST"
    description: "Round amount burst"
    type: "expression"
    action: "soft_boost"
    severity: "medium"
    boost: 0.15
    expression:
      expression: "is_round_amount == 1 and acct_v_5m_count >= 5"
"""
