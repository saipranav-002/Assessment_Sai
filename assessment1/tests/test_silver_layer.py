"""Tests for Silver Layer transformations and deduplication."""

import pytest
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import row_number, col, desc
from datetime import datetime


class TestSilverLayerTransformations:
    """Test Silver layer data transformations."""
    
    @pytest.fixture
    def duplicate_customers_data(self, spark):
        """Create customers data with duplicates for testing deduplication."""
        schema = StructType([
            StructField("customer_id", IntegerType(), False),
            StructField("customer_name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("updated_at", TimestampType(), True)
        ])
        
        # Same customer_id with different updated_at
        data = [
            (1, "John Doe", "john@example.com", "Mumbai", "MH", datetime(2023, 1, 1, 10, 0, 0)),
            (1, "John Doe Updated", "john.new@example.com", "Mumbai", "MH", datetime(2023, 6, 1, 10, 0, 0)),
            (2, "Jane Smith", "jane@example.com", "Delhi", "DL", datetime(2023, 1, 2, 10, 0, 0)),
        ]
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def bronze_orders_data(self, spark):
        """Create bronze orders data."""
        schema = StructType([
            StructField("order_id", IntegerType(), False),
            StructField("customer_id", IntegerType(), True),
            StructField("order_date", StringType(), True),
            StructField("order_status", StringType(), True),
            StructField("updated_at", TimestampType(), True)
        ])
        
        data = [
            (101, 1, "2023-06-15", "Delivered", datetime(2023, 6, 15, 10, 0, 0)),
            (102, 2, "2023-06-16", "Pending", datetime(2023, 6, 16, 10, 0, 0)),
            (103, 999, "2023-06-17", "Delivered", datetime(2023, 6, 17, 10, 0, 0)),  # Orphan record
        ]
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def bronze_order_items_data(self, spark):
        """Create bronze order_items data."""
        schema = StructType([
            StructField("order_item_id", IntegerType(), False),
            StructField("order_id", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("quantity", IntegerType(), True),
            StructField("item_price", DoubleType(), True),
            StructField("updated_at", TimestampType(), True)
        ])
        
        data = [
            (1001, 101, 5001, 2, 100.0, datetime(2023, 6, 15, 10, 0, 0)),
            (1002, 102, 5002, 1, 200.0, datetime(2023, 6, 16, 10, 0, 0)),
        ]
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def bronze_products_data(self, spark):
        """Create bronze products data."""
        schema = StructType([
            StructField("product_id", IntegerType(), False),
            StructField("product_name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("product_price", DoubleType(), True),
            StructField("updated_at", TimestampType(), True)
        ])
        
        data = [
            (5001, "Product_A", "Electronics", 50.0, datetime(2023, 1, 1, 10, 0, 0)),
            (5002, "Product_B", "Clothing", 200.0, datetime(2023, 1, 1, 10, 0, 0)),
        ]
        return spark.createDataFrame(data, schema)
    
    def test_deduplication_keeps_latest_record(self, duplicate_customers_data):
        """Test that deduplication keeps the record with latest updated_at."""
        # Apply deduplication logic
        window_spec = Window.partitionBy("customer_id").orderBy(col("updated_at").desc())
        deduped_df = duplicate_customers_data \
            .withColumn("row_num", row_number().over(window_spec)) \
            .filter(col("row_num") == 1) \
            .drop("row_num")
        
        # Should have only 2 records (customer_id 1 and 2)
        assert deduped_df.count() == 2, "Deduplication should result in 2 unique customers"
        
        # Check that latest record for customer_id 1 is kept
        customer_1 = deduped_df.filter(col("customer_id") == 1).collect()[0]
        assert customer_1["customer_name"] == "John Doe Updated", \
            "Should keep the latest record"
        assert customer_1["email"] == "john.new@example.com", \
            "Should have updated email"
    
    def test_deduplication_removes_older_records(self, duplicate_customers_data):
        """Test that older duplicate records are removed."""
        initial_count = duplicate_customers_data.count()
        
        window_spec = Window.partitionBy("customer_id").orderBy(col("updated_at").desc())
        deduped_df = duplicate_customers_data \
            .withColumn("row_num", row_number().over(window_spec)) \
            .filter(col("row_num") == 1) \
            .drop("row_num")
        
        final_count = deduped_df.count()
        
        assert final_count < initial_count, "Deduplication should reduce row count"
        assert final_count == 2, "Should have exactly 2 unique customers"
    
    def test_silver_customers_no_duplicates(self, duplicate_customers_data):
        """Test that silver customers table has no duplicate primary keys."""
        window_spec = Window.partitionBy("customer_id").orderBy(col("updated_at").desc())
        deduped_df = duplicate_customers_data \
            .withColumn("row_num", row_number().over(window_spec)) \
            .filter(col("row_num") == 1) \
            .drop("row_num")
        
        total_count = deduped_df.count()
        distinct_count = deduped_df.select("customer_id").distinct().count()
        
        assert total_count == distinct_count, "No duplicates should exist after deduplication"
    
    def test_silver_orders_enriched_with_customer_info(self, duplicate_customers_data, bronze_orders_data):
        """Test that orders are enriched with customer information via LEFT JOIN."""
        # Deduplicate customers first
        window_spec = Window.partitionBy("customer_id").orderBy(col("updated_at").desc())
        deduped_customers = duplicate_customers_data \
            .withColumn("row_num", row_number().over(window_spec)) \
            .filter(col("row_num") == 1) \
            .drop("row_num")
        
        # Join orders with customers
        enriched_orders = bronze_orders_data.alias("o").join(
            deduped_customers.alias("c"),
            col("o.customer_id") == col("c.customer_id"),
            "left"
        ).select(
            col("o.order_id"),
            col("o.customer_id"),
            col("o.order_date"),
            col("o.order_status"),
            col("c.customer_name"),
            col("c.city"),
            col("c.state")
        )
        
        # Check enrichment
        assert "customer_name" in enriched_orders.columns, "Should have customer_name"
        assert "city" in enriched_orders.columns, "Should have city"
        assert "state" in enriched_orders.columns, "Should have state"
        
        # Check row count (LEFT JOIN preserves all orders)
        assert enriched_orders.count() == bronze_orders_data.count(), \
            "LEFT JOIN should preserve all orders"
    
    def test_silver_orders_left_join_preserves_orphans(self, duplicate_customers_data, bronze_orders_data):
        """Test that LEFT JOIN preserves orders without matching customers."""
        window_spec = Window.partitionBy("customer_id").orderBy(col("updated_at").desc())
        deduped_customers = duplicate_customers_data \
            .withColumn("row_num", row_number().over(window_spec)) \
            .filter(col("row_num") == 1) \
            .drop("row_num")
        
        enriched_orders = bronze_orders_data.alias("o").join(
            deduped_customers.alias("c"),
            col("o.customer_id") == col("c.customer_id"),
            "left"
        ).select(
            col("o.order_id"),
            col("o.customer_id"),
            col("c.customer_name")
        )
        
        # Order 103 with customer_id 999 should exist with NULL customer_name
        orphan_order = enriched_orders.filter(col("order_id") == 103).collect()
        assert len(orphan_order) == 1, "Orphan order should be preserved"
        assert orphan_order[0]["customer_name"] is None, "Orphan should have NULL customer_name"
    
    def test_silver_order_items_enriched_with_product_info(self, bronze_order_items_data, bronze_products_data):
        """Test that order_items are enriched with product information."""
        enriched_items = bronze_order_items_data.alias("oi").join(
            bronze_products_data.alias("p"),
            col("oi.product_id") == col("p.product_id"),
            "left"
        ).select(
            col("oi.order_item_id"),
            col("oi.order_id"),
            col("oi.product_id"),
            col("oi.quantity"),
            col("oi.item_price"),
            col("p.product_name"),
            col("p.category")
        )
        
        assert "product_name" in enriched_items.columns
        assert "category" in enriched_items.columns
        assert enriched_items.count() == bronze_order_items_data.count()
    
    def test_silver_control_table_tracks_runs(self, spark):
        """Test that silver control table tracks last run times."""
        control_schema = StructType([
            StructField("table_name", StringType(), False),
            StructField("last_run_time", TimestampType(), True)
        ])
        
        control_data = [
            ("silver_customers", datetime(2023, 6, 1, 0, 0, 0)),
            ("silver_orders", datetime(2023, 6, 1, 0, 0, 0))
        ]
        
        control_df = spark.createDataFrame(control_data, control_schema)
        
        assert control_df.count() == 2
        assert "last_run_time" in control_df.columns
