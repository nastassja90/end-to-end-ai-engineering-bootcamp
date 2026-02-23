-- Connect to the tools_database to create the inventory table
\c tools_database

-- Warehouse inventory table
CREATE TABLE inventory (
    id SERIAL PRIMARY KEY,
    warehouse_id VARCHAR(255) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    total_quantity INTEGER NOT NULL DEFAULT 0,
    reserved_quantity INTEGER NOT NULL DEFAULT 0,
    available_quantity INTEGER GENERATED ALWAYS AS (total_quantity - reserved_quantity) STORED,
    warehouse_location VARCHAR(255),
    warehouse_name VARCHAR(255),
    -- estimated_processing_days INTEGER DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT positive_total_quantity CHECK (total_quantity >= 0),
    CONSTRAINT positive_reserved_quantity CHECK (reserved_quantity >= 0),
    CONSTRAINT valid_reservation CHECK (reserved_quantity <= total_quantity),
    CONSTRAINT unique_warehouse_product UNIQUE (warehouse_id, product_id)
);

-- Indexes for warehouse inventory
CREATE INDEX idx_inventory_warehouse ON inventory(warehouse_id);
CREATE INDEX idx_inventory_product ON inventory(product_id);
CREATE INDEX idx_inventory_warehouse_product ON inventory(warehouse_id, product_id);

-- Trigger to automatically update inventory timestamp
CREATE OR REPLACE FUNCTION update_inventory_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER inventory_update_timestamp
    BEFORE UPDATE ON inventory
    FOR EACH ROW
    EXECUTE FUNCTION update_inventory_timestamp();