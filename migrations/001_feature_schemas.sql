-- FraudTrap — Feature Schema Registry Migration
-- Version: 001
-- Description: Creates feature_schemas table for per-tenant feature schema versioning

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Feature schemas table
CREATE TABLE IF NOT EXISTS feature_schemas (
    tenant_id       VARCHAR(64) NOT NULL,
    version         INT NOT NULL,
    feature_hash    CHAR(16) NOT NULL,      -- SHA256[:16] of ordered feature names
    type_hash       CHAR(16) NOT NULL,      -- SHA256[:16] of feature types JSON
    feature_names   JSONB NOT NULL,         -- ["amount", "amount_zscore", ...]
    feature_types   JSONB NOT NULL,         -- {"amount": "float", "amount_zscore": "float"}
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    created_by      VARCHAR(64) DEFAULT 'system',
    is_active       BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (tenant_id, version)
);

-- Index for fast active schema lookup
CREATE INDEX IF NOT EXISTS idx_feature_schemas_active 
    ON feature_schemas(tenant_id) 
    WHERE is_active = TRUE;

-- Trigger to ensure only one active schema per tenant
CREATE OR REPLACE FUNCTION enforce_single_active_schema()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active THEN
        UPDATE feature_schemas 
        SET is_active = FALSE 
        WHERE tenant_id = NEW.tenant_id AND is_active = TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_enforce_single_active_schema ON feature_schemas;
CREATE TRIGGER trigger_enforce_single_active_schema
    BEFORE INSERT OR UPDATE ON feature_schemas
    FOR EACH ROW EXECUTE FUNCTION enforce_single_active_schema();

-- Grant permissions (adjust roles as needed)
-- GRANT SELECT, INSERT, UPDATE ON feature_schemas TO fraudtrap_app;

-- Comment
COMMENT ON TABLE feature_schemas IS 
'Stores per-tenant feature schemas with versioning. Only one active schema per tenant.';

COMMENT ON COLUMN feature_schemas.feature_hash IS 
'SHA256[:16] of sorted feature names joined by "|". Used for quick compatibility checks.';

COMMENT ON COLUMN feature_schemas.type_hash IS 
'SHA256[:16] of feature_types JSON (sorted keys).';

COMMENT ON COLUMN feature_schemas.feature_names IS 
'Ordered list of feature names as JSON array.';

COMMENT ON COLUMN feature_schemas.feature_types IS 
'Feature name -> type mapping as JSON object. Types: float, int, bool, unknown.';