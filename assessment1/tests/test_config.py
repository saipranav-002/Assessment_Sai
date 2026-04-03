"""Tests for configuration validation."""

import json
import pytest
from pathlib import Path


class TestConfigValidation:
    """Test configuration file structure and content."""
    
    @pytest.fixture
    def config_path(self):
        """Path to config.json file."""
        return Path(__file__).parent.parent / "src" / "assessment1_etl" / "config.json"
    
    @pytest.fixture
    def config(self, config_path):
        """Load config.json."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def test_config_file_exists(self, config_path):
        """Test that config.json exists."""
        assert config_path.exists(), "config.json file not found"
    
    def test_config_has_catalogs(self, config):
        """Test that config has catalogs defined."""
        assert "catalogs" in config, "Config missing 'catalogs' key"
        assert isinstance(config["catalogs"], dict), "Catalogs should be a dictionary"
    
    def test_config_has_environments(self, config):
        """Test that config has dev, qa, prod environments."""
        catalogs = config.get("catalogs", {})
        assert "dev" in catalogs, "Missing 'dev' environment"
        assert "qa" in catalogs, "Missing 'qa' environment"
        assert "prod" in catalogs, "Missing 'prod' environment"
    
    def test_catalog_names(self, config):
        """Test that catalog names match expected values."""
        catalogs = config["catalogs"]
        assert catalogs["dev"] == "dev1", "Dev catalog should be 'dev1'"
        assert catalogs["qa"] == "uat1", "QA catalog should be 'uat1'"
        assert catalogs["prod"] == "prod1", "Prod catalog should be 'prod1'"
    
    def test_config_has_schemas(self, config):
        """Test that config has bronze, silver, gold schemas."""
        assert "schemas" in config, "Config missing 'schemas' key"
        schemas = config["schemas"]
        assert "bronze" in schemas, "Missing 'bronze' schema"
        assert "silver" in schemas, "Missing 'silver' schema"
        assert "gold" in schemas, "Missing 'gold' schema"
    
    def test_config_has_tables(self, config):
        """Test that config has table definitions."""
        assert "tables" in config, "Config missing 'tables' key"
        tables = config["tables"]
        expected_tables = ["customers", "orders", "order_items", "products"]
        for table in expected_tables:
            assert table in tables, f"Missing table definition: {table}"
    
    def test_table_primary_keys(self, config):
        """Test that each table has a primary key defined."""
        tables = config["tables"]
        for table_name, table_config in tables.items():
            assert "primary_key" in table_config, f"Table {table_name} missing primary_key"
            assert table_config["primary_key"], f"Primary key for {table_name} is empty"
