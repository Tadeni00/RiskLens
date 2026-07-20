"""
FraudTrap — Feature Engineering
Computes all feature families from raw transaction payloads.
Features are read from Redis (online) or computed on-the-fly.
Results are written back to Redis for the scoring path.
"""
from __future__ import annotations
import hashlib
import math
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import redis

from config.settings import get_settings
from ingestion.schema import TransactionRequest

settings = get_settings()

# ── Redis connection ──────────────────────────────────────────────────────────

def get_redis() -> redis.Redis:
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password or None,
        db=settings.redis_db,
        decode_responses=True,
        socket_timeout=0.050,          # 50ms hard timeout — must not blow the SLA
        socket_connect_timeout=1.0,
    )


# ── Key helpers ───────────────────────────────────────────────────────────────

def _key(tenant: str, entity_type: str, entity_id: str, feature: str) -> str:
    """
    Tenant-namespaced Redis key.
    Format: ft:{tenant_hash}:{entity_type}:{entity_id}:{feature}
    The tenant_hash ensures cross-tenant isolation — keys are unguessable.
    """
    tenant_hash = hashlib.sha256(tenant.encode()).hexdigest()[:12]
    return f"ft:{tenant_hash}:{entity_type}:{entity_id}:{feature}"


# ── Velocity feature computation ──────────────────────────────────────────────

def compute_velocity_features(
    txn: TransactionRequest,
    r: redis.Redis,
) -> dict[str, float]:
    """
    Compute rolling velocity counts and amounts across time windows.
    Uses Redis sorted sets with timestamps as scores for O(log N) range queries.
    Windows: 1min, 5min, 1hr, 24hr, 7day (in seconds).
    """
    now_ts = txn.timestamp.timestamp()
    windows = {
        "1m":  60,
        "5m":  300,
        "1h":  3_600,
        "24h": 86_400,
        "7d":  604_800,
    }
    features: dict[str, float] = {}

    for entity_type, entity_id in [
        ("acct", txn.account_id),
        ("dev",  txn.device_id or "UNKNOWN"),
        ("ip",   txn.ip_address_hash or "UNKNOWN"),
    ]:
        # Sorted set key for this entity's transactions
        ts_key = _key(txn.tenant_id, entity_type, entity_id, "txn_ts")
        amt_key = _key(txn.tenant_id, entity_type, entity_id, "txn_amt")

        pipe = r.pipeline(transaction=False)
        for name, window_sec in windows.items():
            cutoff = now_ts - window_sec
            pipe.zcount(ts_key, cutoff, "+inf")
            pipe.zrangebyscore(amt_key, cutoff, "+inf")
        results = pipe.execute()

        for i, (name, _) in enumerate(windows.items()):
            count = results[i * 2] or 0
            amounts = [float(a) for a in (results[i * 2 + 1] or [])]
            total_amt = sum(amounts)
            prefix = f"{entity_type}_v_{name}"
            features[f"{prefix}_count"] = float(count)
            features[f"{prefix}_total_amt"] = total_amt
            features[f"{prefix}_mean_amt"] = (
                total_amt / count if count > 0 else 0.0
            )

    # Write new transaction to sorted sets (pipeline for atomicity + speed)
    pipe = r.pipeline(transaction=False)
    for entity_type, entity_id in [
        ("acct", txn.account_id),
        ("dev",  txn.device_id or "UNKNOWN"),
        ("ip",   txn.ip_address_hash or "UNKNOWN"),
    ]:
        ts_key = _key(txn.tenant_id, entity_type, entity_id, "txn_ts")
        amt_key = _key(txn.tenant_id, entity_type, entity_id, "txn_amt")
        member = txn.transaction_id
        pipe.zadd(ts_key, {member: now_ts})
        pipe.zadd(amt_key, {txn.transaction_id: txn.amount})
        # Prune entries older than 7 days to bound memory
        pipe.zremrangebyscore(ts_key, "-inf", now_ts - 604_800)
        pipe.zremrangebyscore(amt_key, "-inf", now_ts - 604_800)
        pipe.expire(ts_key, 604_800 + 3_600)
        pipe.expire(amt_key, 604_800 + 3_600)
    pipe.execute()

    return features


# ── Transaction features ──────────────────────────────────────────────────────

def compute_transaction_features(
    txn: TransactionRequest,
    r: redis.Redis,
) -> dict[str, float]:
    """Amount deviation, time-of-day, channel encoding, etc."""
    features: dict[str, float] = {}

    # Amount features
    features["amount"] = txn.amount
    features["amount_log"] = math.log1p(txn.amount)

    # Historical mean/std from Redis
    hist_key = _key(txn.tenant_id, "acct", txn.account_id, "hist_stats")
    hist = r.hgetall(hist_key)
    if hist:
        mean_amt = float(hist.get("mean_amt", txn.amount))
        std_amt = float(hist.get("std_amt", 1.0))
        features["amount_zscore"] = (txn.amount - mean_amt) / max(std_amt, 1.0)
        features["amount_vs_mean_ratio"] = txn.amount / max(mean_amt, 0.01)
    else:
        features["amount_zscore"] = 0.0
        features["amount_vs_mean_ratio"] = 1.0

    # Time features
    hour = txn.timestamp.hour
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    features["day_of_week"] = float(txn.timestamp.weekday())
    features["is_weekend"] = float(txn.timestamp.weekday() >= 5)
    features["is_night"] = float(hour < 6 or hour > 22)

    # Round amount flag (common fraud pattern)
    features["is_round_amount"] = float(txn.amount % 100 == 0)
    features["is_very_round_amount"] = float(txn.amount % 1000 == 0)

    # Channel encoding
    channel_map = {"WEB": 0, "MOBILE": 1, "API": 2, "POS": 3, "ATM": 4, "USSD": 5}
    features["channel_enc"] = float(channel_map.get(txn.channel, -1))

    # Transaction type encoding
    type_map = {
        "PAYMENT": 0, "TRANSFER": 1, "WITHDRAWAL": 2,
        "TOP_UP": 3, "REFUND": 4, "LOAN_DISBURSEMENT": 5
    }
    features["txn_type_enc"] = float(type_map.get(txn.transaction_type, -1))

    # New merchant flag
    merch_key = _key(txn.tenant_id, "acct", txn.account_id, "seen_merchants")
    if txn.merchant_id:
        is_new_merch = not r.sismember(merch_key, txn.merchant_id)
        features["is_new_merchant"] = float(is_new_merch)
        r.sadd(merch_key, txn.merchant_id)
        r.expire(merch_key, 604_800)
    else:
        features["is_new_merchant"] = 0.0

    return features


# ── Device & geo features ─────────────────────────────────────────────────────

def compute_device_geo_features(
    txn: TransactionRequest,
    r: redis.Redis,
) -> dict[str, float]:
    features: dict[str, float] = {}

    # New device for this account
    dev_key = _key(txn.tenant_id, "acct", txn.account_id, "seen_devices")
    if txn.device_id:
        is_new_device = not r.sismember(dev_key, txn.device_id)
        features["is_new_device"] = float(is_new_device)
        r.sadd(dev_key, txn.device_id)
        r.expire(dev_key, 2_592_000)  # 30 days
    else:
        features["is_new_device"] = 1.0   # unknown device = new

    # Device sharing: how many accounts have used this device?
    if txn.device_id:
        dev_acct_key = _key(txn.tenant_id, "dev", txn.device_id, "accounts")
        r.sadd(dev_acct_key, txn.account_id)
        r.expire(dev_acct_key, 604_800)
        device_account_count = r.scard(dev_acct_key)
        features["device_account_count"] = float(device_account_count)
        features["device_shared_flag"] = float(device_account_count > 3)
    else:
        features["device_account_count"] = 0.0
        features["device_shared_flag"] = 0.0

    # Geo velocity: impossible travel detection
    last_loc_key = _key(txn.tenant_id, "acct", txn.account_id, "last_loc")
    last_loc = r.hgetall(last_loc_key)
    if last_loc and txn.latitude and txn.longitude:
        prev_lat = float(last_loc.get("lat", txn.latitude))
        prev_lon = float(last_loc.get("lon", txn.longitude))
        prev_ts  = float(last_loc.get("ts",  txn.timestamp.timestamp()))
        distance_km = _haversine(prev_lat, prev_lon, txn.latitude, txn.longitude)
        time_hours = max((txn.timestamp.timestamp() - prev_ts) / 3_600, 0.001)
        speed_kmh = distance_km / time_hours
        features["geo_distance_km"] = distance_km
        features["geo_speed_kmh"] = speed_kmh
        features["impossible_travel"] = float(speed_kmh > 900)  # faster than a plane
    else:
        features["geo_distance_km"] = 0.0
        features["geo_speed_kmh"] = 0.0
        features["impossible_travel"] = 0.0

    # Update last location
    if txn.latitude and txn.longitude:
        r.hset(last_loc_key, mapping={
            "lat": txn.latitude,
            "lon": txn.longitude,
            "ts":  txn.timestamp.timestamp(),
        })
        r.expire(last_loc_key, 604_800)

    # Cross-country flag
    last_country_key = _key(txn.tenant_id, "acct", txn.account_id, "home_country")
    home_country = r.get(last_country_key)
    if home_country and txn.country_code:
        features["cross_country_flag"] = float(home_country != txn.country_code)
    else:
        features["cross_country_flag"] = 0.0
        if txn.country_code:
            r.set(last_country_key, txn.country_code, ex=2_592_000)

    return features


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6_371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ── Behavioural biometrics ────────────────────────────────────────────────────

def compute_behavioural_features(
    txn: TransactionRequest,
    r: redis.Redis,
) -> dict[str, float]:
    features: dict[str, float] = {}

    features["has_biometrics"] = float(txn.typing_cadence_ms is not None)
    features["session_duration"] = txn.session_duration_seconds or 0.0
    features["field_visit_count"] = float(txn.field_visit_count or 0)

    if txn.typing_cadence_ms is not None:
        # Compare to baseline
        baseline_key = _key(txn.tenant_id, "acct", txn.account_id, "typing_baseline")
        baseline = r.hgetall(baseline_key)
        if baseline:
            b_mean = float(baseline.get("mean", txn.typing_cadence_ms))
            b_std  = float(baseline.get("std", 50.0))
            features["typing_zscore"] = (txn.typing_cadence_ms - b_mean) / max(b_std, 1.0)
        else:
            features["typing_zscore"] = 0.0
            r.hset(baseline_key, mapping={
                "mean": txn.typing_cadence_ms,
                "std":  50.0,
                "n":    1,
            })
            r.expire(baseline_key, 2_592_000)
        features["typing_cadence_ms"] = txn.typing_cadence_ms
    else:
        features["typing_zscore"] = 0.0
        features["typing_cadence_ms"] = 0.0

    return features


# ── Master feature assembler ──────────────────────────────────────────────────

def assemble_feature_vector(
    txn: TransactionRequest,
    r: Optional[redis.Redis] = None,
) -> dict[str, float]:
    """
    Entry point: assembles all feature families into a single flat dict.
    Falls back gracefully if Redis is unavailable (cold features = zeros).
    """
    if r is None:
        try:
            r = get_redis()
        except Exception:
            r = None

    features: dict[str, float] = {}

    if r:
        try:
            features.update(compute_velocity_features(txn, r))
            features.update(compute_transaction_features(txn, r))
            features.update(compute_device_geo_features(txn, r))
            features.update(compute_behavioural_features(txn, r))
        except Exception as exc:
            # Degrade gracefully — score with zero features rather than fail
            from loguru import logger
            logger.warning("Feature computation partial failure: {}", exc)

    # Always-computable features (no Redis needed)
    hour = txn.timestamp.hour
    features["amount"] = txn.amount
    features["amount_log"] = math.log1p(txn.amount)
    features.setdefault("amount_zscore", 0.0)
    features.setdefault("amount_vs_mean_ratio", 1.0)
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    features["day_of_week"] = float(txn.timestamp.weekday())
    features["is_weekend"] = float(txn.timestamp.weekday() >= 5)
    features["is_night"] = float(hour < 6 or hour > 22)
    features["is_round_amount"] = float(txn.amount % 100 == 0)
    features["is_very_round_amount"] = float(txn.amount % 1000 == 0)
    channel_map = {"WEB": 0, "MOBILE": 1, "API": 2, "POS": 3, "ATM": 4, "USSD": 5}
    type_map = {
        "PAYMENT": 0, "TRANSFER": 1, "WITHDRAWAL": 2,
        "TOP_UP": 3, "REFUND": 4, "LOAN_DISBURSEMENT": 5
    }
    features["channel_enc"] = float(channel_map.get(txn.channel, -1))
    features["txn_type_enc"] = float(type_map.get(txn.transaction_type, -1))
    features.setdefault("is_new_device", float(txn.device_id is None))
    features.setdefault("is_new_merchant", 0.0)
    features.setdefault("impossible_travel", 0.0)
    features.setdefault("acct_v_1m_count", 0.0)

    return features
