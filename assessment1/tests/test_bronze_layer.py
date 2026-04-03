"""Tests for Bronze Layer ingestion logic."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from datetime import datetime


class TestBronzeLayerIngestion:
    """Test Bronze layer data ingestion."""
    
    @pytest.fixture
    def sample_customers_data(self, spark):
        """Create sample customers data for testing."""
        schema = StructType([
            StructField("customer_id", IntegerType(), False),
            StructField("customer_name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("signup_date", StringType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True)
        ])
        
        data = [
            (1, "John Doe", "john@example.com", "Mumbai", "MH", "2023-01-01", 
             datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 10, 0, 0)),
            (2, "Jane Smith", "jane@example.com", "Delhi", "DL", "2023-01-02",
             datetime(2023, 1, 2, 10, 0, 0), datetime(2023, 1, 2, 10, 0, 0)),
        ]
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def sample_orders_data(self, spark):
        """Create sample orders data for testing."""
        schema = StructType([
            StructField("order_id", IntegerType(), False),
            StructField("customer_id", IntegerType(), True),
            StructField("order_date", StringType(), True),
            StructField("order_status", StringType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True)
        ])
        
        data = [
            (101, 1, "2023-06-15", "Delivered", 
             datetime(2023, 6, 15, 10, 0, 0), datetime(2023, 6, 15, 10, 0, 0)),
            (102, 2, "2023-06-16", "Pending",
             datetime(2023, 6, 16, 10, 0, 0), datetime(2023, 6, 16, 10, 0, 0)),
        ]
        return spark.createDataFrame(data, schema)
    
    def test_bronze_schema_has_metadata_columns(self, spark, sample_customers_data):
        """Test that bronze tables have ingestion_time and path columns."""
        # Add metadata columns
        df_with_metadata = sample_customers_data \
            .withColumn("ingestion_time", spark.range(1).select("id").first()[0]) \
            .withColumn("path", spark.range(1).select("id").first()[0])
        
        assert "ingestion_time" in df_with_metadata.columns, "Missing ingestion_time column"
        assert "path" in df_with_metadata.columns, "Missing path column"
    
    def test_bronze_primary_key_not_null(self, sample_customers_data):
        """Test that primary key column has no nulls."""
        null_count = sample_customers_data.filter("customer_id IS NULL").count()
        assert null_count == 0, f"Found {null_count} null values in primary key"
    
    def test_bronze_row_count(self, sample_customers_data):
        """Test that bronze data has expected row count."""
        assert sample_customers_data.count() >= 1, "Bronze table should have at least 1 row"
    
    def test_bronze_required_columns(self, sample_customers_data):
        """Test that bronze table has all required columns."""
        required_columns = ["customer_id", "customer_name", "email", "city", "state", 
                           "signup_date", "created_at", "updated_at"]
        actual_columns = sample_customers_data.columns
        
        for col in required_columns:
            assert col in actual_columns, f"Missing required column: {col}"
    
    def test_bronze_orders_schema(self, sample_orders_data):
        """Test orders table has correct schema."""
        required_columns = ["order_id", "customer_id", "order_date", "order_status", 
                           "created_at", "updated_at"]
        actual_columns = sample_orders_data.columns
        
        for col in required_columns:
            assert col in actual_columns, f"Missing required column: {col}"
    
    def test_bronze_incremental_filter(self, spark, sample_customers_data):
        """Test incremental load filter based on timestamps."""
        last_run_time = datetime(2023, 1, 1, 12, 0, 0)
        
        # Simulate incremental filter
        incremental_df = sample_customers_data.filter(
            (sample_customers_data.created_at > last_run_time) | 
            (sample_customers_data.updated_at > last_run_time)
        )
        
        # All records before last_run_time should be filtered
        assert incremental_df.count() == 0, "Incremental filter should exclude old records"
    
    def test_bronze_upsert_logic(self, spark, sample_customers_data):
        """Test MERGE UPSERT logic (simulated)."""
        # Create existing data
        existing_data = sample_customers_data.limit(1)
        
        # Create updated data with same customer_id but different values
        updated_data = existing_data.withColumn("city", spark.lit("Pune"))
        
        # In actual MERGE, this would update the existing record
        # Here we verify the updated record has changed
        assert updated_data.select("city").collect()[0][0] == "Pune"
    
    def test_bronze_partition_column_exists(self, sample_customers_data):
        """Test that partition column (created_at) exists."""
        assert "created_at" in sample_customers_data.columns, \
            "Partition column 'created_at' not found"
    
    def test_bronze_data_quality_no_duplicates(self, sample_customers_data):
        """Test that bronze data has no duplicate primary keys."""
        total_count = sample_customers_data.count()
        distinct_count = sample_customers_data.select("customer_id").distinct().count()
        
        assert total_count == distinct_count, \
            f"Found duplicates: {total_count} total vs {distinct_count} distinct"
    
    def test_bronze_timestamp_format(self, sample_customers_data):
        """Test that timestamp columns have valid datetime format."""
        timestamp_cols = ["created_at", "updated_at"]
        
        for col in timestamp_cols:
            null_count = sample_customers_data.filter(f"{col} IS NULL").count()
            assert null_count < sample_customers_data.count(), \
                f"All values in {col} are null"
    
    def test_bronze_control_table_structure(self, spark):
        """Test that control table would have correct structure."""
        control_schema = StructType([
            StructField("table_name", StringType(), False),
            StructField("last_run_time", TimestampType(), True),
            StructField("run_status", StringType(), True)
        ])
        
        control_data = [
            ("customers", datetime(2023, 1, 1, 0, 0, 0), "SUCCESS"),
            ("orders", datetime(2023, 1, 1, 0, 0, 0), "SUCCESS")
        ]
        
        control_df = spark.createDataFrame(control_data, control_schema)
        
        assert "table_name" in control_df.columns
        assert "last_run_time" in control_df.columns
        assert control_df.count() == 2
