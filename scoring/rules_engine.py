"""
RiskLens — Rules Engine (Tier 1)
Evaluates declarative rules from YAML/JSON configuration.
Supports blocklists, thresholds, expressions, velocity, and geo rules.
"""

from __future__ import annotations
import re
import ast
import operator
import hashlib
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
from pathlib import Path

import redis
from loguru import logger

from scoring.rules_config import (
    RuleDefinition,
    RuleType,
    RuleAction,
    RuleSeverity,
    BlocklistConfig,
    ThresholdConfig,
    ExpressionConfig,
    VelocityConfig,
    GeoConfig,
    Operator,
    DEFAULT_RULESET,
    load_ruleset_from_file,
)


@dataclass
class RuleContribution:
    """Individual rule's contribution to the risk score."""

    rule_id: str
    description: str
    triggered: bool
    contribution: float  # 0.0 to 1.0
    hard_block: bool
    rule_type: str
    severity: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RuleResult:
    """Aggregated result of rule evaluation."""

    triggered: bool
    rule_ids: list[str] = field(default_factory=list)
    hard_block: bool = False
    score_override: Optional[float] = None
    risk_boost: float = 0.0
    contributions: list[RuleContribution] = field(default_factory=list)


class SafeExpressionEvaluator:
    """
    Safe expression evaluator using AST parsing.
    Only allows: comparisons, boolean ops, arithmetic, feature access.
    """

    _ALLOWED_NODES = {
        "Expression",
        "Expr",
        "Compare",
        "BoolOp",
        "BinOp",
        "UnaryOp",
        "Name",
        "Constant",
        "Load",
        "And",
        "Or",
        "Not",
        "Eq",
        "NotEq",
        "Lt",
        "LtE",
        "Gt",
        "GtE",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "Mod",
        "Pow",
        "USub",
        "UAdd",
        "In",
        "NotIn",
        "List",
        "Tuple",
    }

    _ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    def __init__(self):
        self._cache: dict[str, ast.AST] = {}

    def evaluate(self, expression: str, features: dict[str, float]) -> bool:
        """Safely evaluate expression with feature values."""
        try:
            tree = self._parse(expression)
            return bool(self._eval_node(tree, features))
        except Exception as exc:
            logger.warning(
                "Expression evaluation failed: {} | expr={}", exc, expression
            )
            return False

    def _parse(self, expression: str) -> ast.AST:
        if expression in self._cache:
            return self._cache[expression]

        tree = ast.parse(expression, mode="eval")
        self._validate(tree)
        self._cache[expression] = tree
        return tree

    def _validate(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if type(child).__name__ not in self._ALLOWED_NODES:
                raise ValueError(f"Disallowed AST node: {type(child).__name__}")

    def _eval_node(self, node: ast.AST, features: dict[str, float]) -> float:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, features)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return features.get(node.id, 0.0)
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, features)
            for op, right in zip(node.ops, node.comparators):
                right_val = self._eval_node(right, features)
                if not self._ALLOWED_OPS[type(op)](left, right_val):
                    return 0.0
                left = right_val
            return 1.0
        elif isinstance(node, ast.BoolOp):
            values = [self._eval_node(v, features) for v in node.values]
            if isinstance(node.op, ast.And):
                return 1.0 if all(values) else 0.0
            elif isinstance(node.op, ast.Or):
                return 1.0 if any(values) else 0.0
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, features)
            right = self._eval_node(node.right, features)
            return self._ALLOWED_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, features)
            return self._ALLOWED_OPS[type(node.op)](operand)
        elif isinstance(node, ast.List):
            return [self._eval_node(e, features) for e in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(e, features) for e in node.elts)
        return 0.0


class RulesEngine:
    """
    Tier 1 deterministic rule evaluation.
    Runs in < 1ms. Fires before ML models.
    Rules loaded from YAML/JSON with hot-reload support.
    """

    # ISO 3166-1 alpha-2 sanctioned countries (illustrative subset)
    _SANCTIONED_COUNTRIES = {"KP", "IR", "CU", "SY", "RU"}

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        rules_path: Optional[str] = None,
    ):
        self._redis = redis_client
        self._rules_path = rules_path
        self._rules: list[RuleDefinition] = []
        self._evaluator = SafeExpressionEvaluator()
        self._rules_cache: list[RuleDefinition] = []
        self._cache_ttl = 60  # seconds
        self._last_load = 0.0
        self._rules_hash = ""

        # Load initial rules
        self.reload_rules()

    def reload_rules(self) -> None:
        """Reload rules from file or Redis, falling back to defaults."""
        if self._rules_path:
            try:
                self._rules = load_ruleset_from_file(self._rules_path)
                logger.info(
                    "Loaded {} rules from {}", len(self._rules), self._rules_path
                )
                return
            except Exception as exc:
                logger.warning("Failed to load rules from file: {}", exc)

        if self._redis:
            try:
                raw = self._redis.get("fraudtrap:rules:definitions")
                if raw:
                    data = json.loads(raw)
                    self._rules = [
                        RuleDefinition.from_dict(r) for r in data.get("rules", [])
                    ]
                    logger.info("Loaded {} rules from Redis", len(self._rules))
                    return
            except Exception as exc:
                logger.warning("Failed to load rules from Redis: {}", exc)

        # Fallback to defaults
        self._rules = DEFAULT_RULESET.copy()
        logger.info("Using default ruleset ({} rules)", len(self._rules))

    def maybe_reload(self) -> bool:
        """Check if rules file changed and reload if needed."""
        if not self._rules_path:
            return False

        try:
            stat = Path(self._rules_path).stat()
            current_hash = hashlib.sha256(
                f"{stat.st_mtime}:{stat.st_size}".encode()
            ).hexdigest()

            if current_hash != self._rules_hash:
                self._rules_hash = current_hash
                self.reload_rules()
                return True
        except Exception:
            pass
        return False

    def save_rules(self, rules: list[RuleDefinition]) -> None:
        """Save rules to Redis and/or file."""
        data = {
            "rules": [r.to_dict() for r in rules],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._redis:
            self._redis.set("fraudtrap:rules:definitions", json.dumps(data))
            logger.info("Saved {} rules to Redis", len(rules))

        if self._rules_path:
            from scoring.rules_config import save_ruleset_to_file

            save_ruleset_to_file(rules, self._rules_path)
            logger.info("Saved {} rules to {}", len(rules), self._rules_path)

        self._rules = rules

    # ─── Rule Evaluation ─────────────────────────────────────────────────────

    def evaluate(
        self, txn: TransactionRequest, features: dict[str, float]
    ) -> RuleResult:
        """Evaluate all rules and return aggregated result."""
        triggered: list[str] = []
        hard_block = False
        total_boost = 0.0
        score_override: Optional[float] = None
        contributions: list[RuleContribution] = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            try:
                rule_triggered, contribution, is_hard_block, override = (
                    self._evaluate_rule(rule, txn, features)
                )

                if rule_triggered:
                    triggered.append(rule.id)
                    contributions.append(contribution)

                    if rule.action == RuleAction.HARD_BLOCK:
                        hard_block = True
                    if rule.score_override is not None:
                        score_override = rule.score_override
                    elif rule.action == RuleAction.SOFT_BOOST:
                        total_boost = min(rule.max_boost, total_boost + rule.boost)

            except Exception as exc:
                logger.warning("Rule evaluation error for {}: {}", rule.id, exc)

        return RuleResult(
            triggered=len(triggered) > 0,
            rule_ids=triggered,
            hard_block=hard_block,
            score_override=score_override,
            risk_boost=min(0.30, total_boost),
            contributions=contributions,
        )

    def _evaluate_rule(
        self, rule: RuleDefinition, txn: TransactionRequest, features: dict[str, float]
    ) -> tuple[bool, RuleContribution, bool, Optional[float]]:
        """Evaluate a single rule. Returns (triggered, contribution, hard_block, score_override)."""
        triggered = False
        contribution = 0.0
        is_hard_block = False
        override = None

        if rule.type == RuleType.BLOCKLIST:
            triggered = self._check_blocklist(rule.blocklist, txn)
            if triggered:
                contribution = 1.0
                is_hard_block = True

        elif rule.type == RuleType.THRESHOLD:
            triggered = self._check_threshold(rule.threshold, features)
            if triggered:
                contribution = 1.0
                is_hard_block = rule.action == RuleAction.HARD_BLOCK

        elif rule.type == RuleType.EXPRESSION:
            triggered = self._evaluator.evaluate(rule.expression.expression, features)
            if triggered:
                if rule.action == RuleAction.HARD_BLOCK:
                    contribution = 1.0
                    is_hard_block = True
                else:
                    contribution = rule.boost

        elif rule.type == RuleType.VELOCITY:
            triggered = self._check_velocity(rule.velocity, features)
            if triggered:
                if rule.action == RuleAction.HARD_BLOCK:
                    contribution = 1.0
                    is_hard_block = True
                else:
                    contribution = rule.boost

        elif rule.type == RuleType.GEO:
            triggered = self._check_geo(rule.geo, txn, features)
            if triggered:
                if rule.action == RuleAction.HARD_BLOCK:
                    contribution = 1.0
                    is_hard_block = True
                else:
                    contribution = rule.boost

        # Sanctioned country check (always runs)
        if not triggered and rule.id == "SANCTIONED_COUNTRY":
            if txn.country_code in self._SANCTIONED_COUNTRIES:
                triggered = True
                contribution = 1.0
                is_hard_block = True

        # Score override
        if triggered and rule.score_override is not None:
            override = rule.score_override

        contr = RuleContribution(
            rule_id=rule.id,
            description=rule.description,
            triggered=triggered,
            contribution=contribution,
            hard_block=is_hard_block,
            rule_type=rule.type.value,
            severity=rule.severity.value,
            metadata={
                "action": rule.action.value,
                "boost": rule.boost if rule.action == RuleAction.SOFT_BOOST else 0,
            },
        )

        return triggered, contr, is_hard_block, override

    def _check_blocklist(
        self, config: BlocklistConfig, txn: TransactionRequest
    ) -> bool:
        if not self._redis or not config:
            return False

        value_map = {
            "account": txn.account_id,
            "device": txn.device_id,
            "ip": txn.ip_address_hash,
            "merchant": txn.merchant_id,
            "country": txn.country_code,
        }
        value = value_map.get(config.entity)
        if not value:
            return False

        try:
            return bool(
                self._redis.sismember(f"ft:blocklist:{config.list_name}", value)
            )
        except Exception:
            return False

    def _check_threshold(
        self, config: ThresholdConfig, features: dict[str, float]
    ) -> bool:
        val = features.get(config.feature, 0.0)
        return self._compare(val, config.operator, config.threshold)

    def _check_velocity(
        self, config: VelocityConfig, features: dict[str, float]
    ) -> bool:
        feature_map = {
            "account": "acct",
            "device": "dev",
            "ip": "ip",
        }
        prefix = feature_map.get(config.entity_type, "acct")
        feature_name = f"{prefix}_v_{config.window_seconds//60 if config.window_seconds >= 60 else config.window_seconds}m_count"

        # Handle special naming
        if config.window_seconds == 60:
            feature_name = f"{prefix}_v_1m_count"
        elif config.window_seconds == 300:
            feature_name = f"{prefix}_v_5m_count"
        elif config.window_seconds == 3600:
            feature_name = f"{prefix}_v_1h_count"
        elif config.window_seconds == 86400:
            feature_name = f"{prefix}_v_24h_count"
        elif config.window_seconds == 604800:
            feature_name = f"{prefix}_v_7d_count"

        val = features.get(feature_name, 0.0)
        return self._compare(val, config.operator, config.threshold)

    def _check_geo(
        self, config: GeoConfig, txn: TransactionRequest, features: dict[str, float]
    ) -> bool:
        if config.rule_type == "impossible_travel":
            return features.get("impossible_travel", 0.0) == 1.0
        elif config.rule_type == "cross_border":
            return features.get("cross_country_flag", 0.0) == 1.0
        elif config.rule_type == "sanctioned_country":
            return txn.country_code in self._SANCTIONED_COUNTRIES
        return False

    @staticmethod
    def _compare(val: float, op: Operator, threshold: float) -> bool:
        ops = {
            Operator.GT: lambda v, t: v > t,
            Operator.GTE: lambda v, t: v >= t,
            Operator.LT: lambda v, t: v < t,
            Operator.LTE: lambda v, t: v <= t,
            Operator.EQ: lambda v, t: v == t,
            Operator.NEQ: lambda v, t: v != t,
        }
        return ops.get(op, lambda v, t: False)(val, threshold)

    # ─── Blocklist Management (Admin API) ─────────────────────────────────────

    def add_to_blocklist(self, list_name: str, value: str) -> None:
        if self._redis:
            self._redis.sadd(f"ft:blocklist:{list_name}", value)
            logger.info("Added {} to blocklist:{}", value, list_name)

    def remove_from_blocklist(self, list_name: str, value: str) -> None:
        if self._redis:
            self._redis.srem(f"ft:blocklist:{list_name}", value)
            logger.info("Removed {} from blocklist:{}", value, list_name)

    # ─── Explainability ───────────────────────────────────────────────────────

    def explain(self, txn: TransactionRequest, features: dict[str, float]) -> dict:
        """Return detailed rule contributions for explainability."""
        result = self.evaluate(TransactionRequest, features)

        return {
            "model_type": "rules",
            "base_value": 0.0,
            "prediction_value": min(
                1.0, result.risk_boost + (1.0 if result.hard_block else 0.0)
            ),
            "top_features": [
                {
                    "feature": c.rule_id,
                    "value": 1.0 if c.triggered else 0.0,
                    "contribution": c.contribution,
                    "method": "rule_weight",
                }
                for c in result.contributions
            ],
            "components": {
                "hard_block": float(result.hard_block),
                "total_boost": result.risk_boost,
                "triggered_rules": result.rule_ids,
            },
        }


# ─── Blocklist Management with TTL & Audit ────────────────────────────────────


@dataclass
class BlocklistEntry:
    """Blocklist entry with metadata and TTL."""

    value: str
    list_name: str
    tenant_id: str
    added_by: str
    reason: str
    added_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

    def to_redis_hash(self) -> dict:
        return {
            "value": self.value,
            "list_name": self.list_name,
            "tenant_id": self.tenant_id,
            "added_by": self.added_by,
            "reason": self.reason,
            "added_at": self.added_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else "",
            "is_active": "1" if self.is_active else "0",
        }

    @classmethod
    def from_redis_hash(cls, data: dict) -> "BlocklistEntry":
        return cls(
            value=data["value"],
            list_name=data["list_name"],
            tenant_id=data["tenant_id"],
            added_by=data["added_by"],
            reason=data["reason"],
            added_at=datetime.fromisoformat(data["added_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            is_active=data.get("is_active", "1") == "1",
        )


class BlocklistManager:
    """
    Manages blocklists with TTL, audit trail, and tenant isolation.
    """

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    def _entry_key(self, tenant_id: str, list_name: str, value: str) -> str:
        return f"ft:blocklist:entry:{tenant_id}:{list_name}:{value}"

    def _list_key(self, tenant_id: str, list_name: str) -> str:
        return f"ft:blocklist:set:{tenant_id}:{list_name}"

    def _audit_key(self, tenant_id: str) -> str:
        return f"ft:blocklist:audit:{tenant_id}"

    def add(
        self,
        list_name: str,
        value: str,
        tenant_id: str,
        added_by: str,
        reason: str,
        ttl_days: Optional[int] = None,
    ) -> BlocklistEntry:
        """Add entry to blocklist with TTL and audit trail."""
        entry = BlocklistEntry(
            value=value,
            list_name=list_name,
            tenant_id=tenant_id,
            added_by=added_by,
            reason=reason,
            added_at=datetime.now(timezone.utc),
            expires_at=(
                datetime.now(timezone.utc) + timedelta(days=ttl_days)
                if ttl_days
                else None
            ),
        )

        pipe = self._redis.pipeline(transaction=False)

        # Store entry metadata
        pipe.hset(
            self._entry_key(tenant_id, list_name, value), mapping=entry.to_redis_hash()
        )

        # Add to set for fast lookup
        pipe.sadd(self._list_key(tenant_id, list_name), value)

        # Audit log
        audit = {
            "action": "add",
            "value": value,
            "list_name": list_name,
            "added_by": added_by,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        pipe.lpush(self._audit_key(tenant_id), json.dumps(audit))
        pipe.ltrim(self._audit_key(tenant_id), 0, 9999)

        if entry.expires_at:
            ttl_seconds = int(
                (entry.expires_at - datetime.now(timezone.utc)).total_seconds()
            )
            pipe.expire(self._entry_key(tenant_id, list_name, value), ttl_seconds)

        pipe.execute()

        logger.info(
            "Blocklist add: tenant={} list={} value={} by={}",
            tenant_id,
            list_name,
            value,
            added_by,
        )
        return entry

    def remove(
        self, list_name: str, value: str, tenant_id: str, removed_by: str, reason: str
    ) -> bool:
        """Remove entry from blocklist with audit."""
        removed = self._redis.srem(self._list_key(tenant_id, list_name), value)
        self._redis.delete(self._entry_key(tenant_id, list_name, value))

        if removed:
            audit = {
                "action": "remove",
                "value": value,
                "list_name": list_name,
                "removed_by": removed_by,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._redis.lpush(self._audit_key(tenant_id), json.dumps(audit))
            logger.info(
                "Blocklist remove: tenant={} list={} value={} by={}",
                tenant_id,
                list_name,
                value,
                removed_by,
            )
        return bool(removed)

    def get_entry(
        self, list_name: str, value: str, tenant_id: str
    ) -> Optional[BlocklistEntry]:
        """Get blocklist entry metadata."""
        data = self._redis.hgetall(self._entry_key(tenant_id, list_name, value))
        if not data:
            return None
        return BlocklistEntry.from_redis_hash(data)

    def list_entries(
        self, list_name: str, tenant_id: str, active_only: bool = True
    ) -> list[BlocklistEntry]:
        """List all entries in a blocklist."""
        values = self._redis.smembers(self._list_key(tenant_id, list_name))
        entries = []
        for value in values:
            entry = self.get_entry(list_name, value, tenant_id)
            if entry and (not active_only or entry.is_active):
                entries.append(entry)
        return entries

    def get_audit_log(self, tenant_id: str, limit: int = 100) -> list[dict]:
        """Get audit log for blocklist changes."""
        logs = self._redis.lrange(self._audit_key(tenant_id), 0, limit - 1)
        return [json.loads(log) for log in logs]

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        # This would need a SCAN loop in production
        # For now, rely on Redis TTL expiration
        return 0


# ─── Heuristic Score Configuration ────────────────────────────────────────────


@dataclass
class HeuristicConfig:
    """Configurable heuristic score components."""

    base_score: float = 0.08
    components: dict[str, dict] = field(
        default_factory=lambda: {
            "amount_zscore": {"weight": 0.04, "transform": "max(0, x)", "cap": 0.30},
            "amount": {"weight": 4e-7, "transform": "x", "cap": 0.22},  # 1/2_500_000
            "is_new_device": {"weight": 0.16, "transform": "x", "cap": 0.16},
            "is_new_merchant": {"weight": 0.12, "transform": "x", "cap": 0.12},
            "impossible_travel": {"weight": 0.22, "transform": "x", "cap": 0.22},
            "acct_v_1m_count": {"weight": 0.015, "transform": "x", "cap": 0.18},
            "is_night": {"weight": 0.05, "transform": "x", "cap": 0.05},
            "high_risk_channel": {
                "weight": 0.05,
                "transform": "x in [2, 5]",
                "cap": 0.05,
            },
        }
    )
    max_score: float = 1.0

    def score(self, features: dict[str, float]) -> float:
        """Compute heuristic score from features."""
        score = self.base_score

        for name, config in self.components.items():
            val = features.get(name, 0.0)

            # Apply transform
            if config["transform"] == "max(0, x)":
                val = max(val, 0.0)
            elif config["transform"] == "x in [2, 5]":
                val = 1.0 if val in (2.0, 5.0) else 0.0
            elif config["transform"] != "x":
                val = 0.0

            contrib = min(config["cap"], config["weight"] * val)
            score += contrib

        return min(score, self.max_score)


DEFAULT_HEURISTIC = HeuristicConfig()


# ─── Convenience Functions ────────────────────────────────────────────────────


def create_rules_engine(
    redis_client: Optional[redis.Redis] = None,
    rules_path: Optional[str] = None,
) -> RulesEngine:
    """Factory to create rules engine."""
    return RulesEngine(redis_client=redis_client, rules_path=rules_path)


def create_blocklist_manager(redis_client: redis.Redis) -> BlocklistManager:
    """Factory to create blocklist manager."""
    return BlocklistManager(redis_client)
