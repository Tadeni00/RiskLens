import os
from clickhouse_driver import Client
import time

def init_db():
    host = os.getenv("CLICKHOUSE_HOST", "clickhouse")
    port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    
    print(f"Connecting to ClickHouse at {host}:{port}...")
    
    # Wait for ClickHouse to be ready
    for _ in range(30):
        try:
            client = Client(host=host, port=port)
            client.execute("SELECT 1")
            break
        except Exception:
            time.sleep(1)
    else:
        raise Exception("ClickHouse failed to start")

    print("Connected. Creating schema...")

    # Create Database
    client.execute("CREATE DATABASE IF NOT EXISTS fraudtrap")

    # Create Main Table
    client.execute("""
    CREATE TABLE IF NOT EXISTS fraudtrap.transactions (
        transaction_id String,
        tenant_id String,
        account_id String,
        session_id Nullable(String),
        amount Float32,
        currency String,
        timestamp DateTime64(3, 'UTC'),
        transaction_type String,
        channel String,
        merchant_id Nullable(String),
        merchant_category_code Nullable(String),
        merchant_country Nullable(String),
        counterparty_account_id Nullable(String),
        device_id Nullable(String),
        device_type Nullable(String),
        ip_address_hash Nullable(String),
        user_agent_hash Nullable(String),
        latitude Nullable(Float32),
        longitude Nullable(Float32),
        country_code Nullable(String),
        typing_cadence_ms Nullable(Float32),
        session_duration_seconds Nullable(Float32),
        field_visit_count Nullable(Int32),
        risk_score Float32,
        decision String,
        model_phase String,
        model_version String,
        latency_ms Float32,
        triggered_rules Array(String),
        trace_id String,
        scored_at DateTime64(3, 'UTC'),
        is_fraud UInt8
    ) ENGINE = MergeTree()
    ORDER BY (tenant_id, timestamp)
    """)

    # Create Kafka Queue Table
    # NOTE: KAFKA_BROKER_LIST must be the internal docker hostname 'kafka:29092'
    client.execute("""
    CREATE TABLE IF NOT EXISTS fraudtrap.transactions_queue (
        transaction_id String,
        tenant_id String,
        account_id String,
        session_id Nullable(String),
        amount Float32,
        currency String,
        timestamp String,
        transaction_type String,
        channel String,
        merchant_id Nullable(String),
        merchant_category_code Nullable(String),
        merchant_country Nullable(String),
        counterparty_account_id Nullable(String),
        device_id Nullable(String),
        device_type Nullable(String),
        ip_address_hash Nullable(String),
        user_agent_hash Nullable(String),
        latitude Nullable(Float32),
        longitude Nullable(Float32),
        country_code Nullable(String),
        typing_cadence_ms Nullable(Float32),
        session_duration_seconds Nullable(Float32),
        field_visit_count Nullable(Int32),
        risk_score Float32,
        decision String,
        model_phase String,
        model_version String,
        latency_ms Float32,
        triggered_rules Array(String),
        trace_id String,
        scored_at String,
        is_fraud UInt8
    ) ENGINE = Kafka('kafka:29092', 'fraudtrap.audit.decisions', 'clickhouse_group', 'JSONEachRow')
    SETTINGS kafka_skip_broken_messages = 100
    """)

    # Create Materialized View to pipe from Queue to Main Table
    client.execute("""
    CREATE MATERIALIZED VIEW IF NOT EXISTS fraudtrap.transactions_mv 
    TO fraudtrap.transactions AS
    SELECT
        transaction_id,
        tenant_id,
        account_id,
        session_id,
        amount,
        currency,
        parseDateTime64BestEffort(timestamp) AS timestamp,
        transaction_type,
        channel,
        merchant_id,
        merchant_category_code,
        merchant_country,
        counterparty_account_id,
        device_id,
        device_type,
        ip_address_hash,
        user_agent_hash,
        latitude,
        longitude,
        country_code,
        typing_cadence_ms,
        session_duration_seconds,
        field_visit_count,
        risk_score,
        decision,
        model_phase,
        model_version,
        latency_ms,
        triggered_rules,
        trace_id,
        parseDateTime64BestEffort(scored_at) AS scored_at,
        is_fraud
    FROM fraudtrap.transactions_queue
    """)
    
    print("ClickHouse schema initialized.")

if __name__ == "__main__":
    init_db()
