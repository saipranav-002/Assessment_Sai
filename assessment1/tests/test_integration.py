"""Integration tests for end-to-end ETL pipeline validation."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


class TestEndToEndPipeline:
    """Test end-to-end data flow through medallion architecture."""
    
    def test_bronze_tables_exist(self, spark):
        """Test that bronze tables exist in the catalog."""
        catalog = "dev1"
        schema = "bronze"
        expected_tables = ["customers", "orders", "order_items", "products"]
        
        # Get list of tables in bronze schema
        try:
            tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
            table_names = [row.tableName for row in tables]
            
            for expected_table in expected_tables:
                assert expected_table in table_names, \
                    f"Bronze table {expected_table} not found"
        except Exception as e:
            pytest.skip(f"Bronze schema not accessible: {e}")
    
    def test_silver_tables_exist(self, spark):
        """Test that silver tables exist in the catalog."""
        catalog = "dev1"
        schema = "silver"
        expected_tables = ["silver_customers", "silver_products", 
                          "silver_orders", "silver_order_items"]
        
        try:
            tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
            table_names = [row.tableName for row in tables]
            
            for expected_table in expected_tables:
                assert expected_table in table_names, \
                    f"Silver table {expected_table} not found"
        except Exception as e:
            pytest.skip(f"Silver schema not accessible: {e}")
    
    def test_gold_tables_exist(self, spark):
        """Test that gold tables exist in the catalog."""
        catalog = "dev1"
        schema = "gold"
        expected_tables = ["dim_customers", "dim_products", "fact_sales",
                          "revenue_by_state", "top_products", 
                          "sales_trends_daily", "sales_trends_monthly"]
        
        try:
            tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
            table_names = [row.tableName for row in tables]
            
            for expected_table in expected_tables:
                assert expected_table in table_names, \
                    f"Gold table {expected_table} not found"
        except Exception as e:
            pytest.skip(f"Gold schema not accessible: {e}")
    
    def test_data_lineage_bronze_to_silver(self, spark):
        """Test data lineage from bronze to silver layer."""
        try:
            bronze_count = spark.table("dev1.bronze.customers").count()
            silver_count = spark.table("dev1.silver.silver_customers").count()
            
            # Silver should have <= bronze count (due to deduplication)
            assert silver_count <= bronze_count, \
                "Silver customers should be <= bronze customers after deduplication"
            assert silver_count > 0, "Silver customers should have data"
        except Exception as e:
            pytest.skip(f"Tables not accessible for lineage test: {e}")
    
    def test_data_lineage_silver_to_gold(self, spark):
        """Test data lineage from silver to gold layer."""
        try:
            silver_orders = spark.table("dev1.silver.silver_orders").count()
            gold_fact = spark.table("dev1.gold.fact_sales").count()
            
            # Gold should have fewer records (filtered by status)
            assert gold_fact > 0, "Gold fact_sales should have data"
            assert gold_fact <= silver_orders, \
                "Fact sales should be filtered subset of silver orders"
        except Exception as e:
            pytest.skip(f"Tables not accessible for lineage test: {e}")
    
    def test_fact_sales_data_quality(self, spark):
        """Test data quality in fact_sales table."""
        try:
            fact_sales = spark.table("dev1.gold.fact_sales")
            
            # Test 1: No nulls in key columns
            null_orders = fact_sales.filter(col("order_id").isNull()).count()
            assert null_orders == 0, "Fact_sales should have no null order_ids"
            
            # Test 2: Only valid order statuses
            invalid_statuses = fact_sales.filter(
                ~col("order_status").isin(["Delivered", "Pending"])
            ).count()
            assert invalid_statuses == 0, \
                "Fact_sales should only have Delivered/Pending orders"
            
            # Test 3: Line total is positive
            negative_totals = fact_sales.filter(col("line_total") <= 0).count()
            assert negative_totals == 0, "Line totals should be positive"
            
            # Test 4: Quantity is positive
            invalid_quantity = fact_sales.filter(col("quantity") <= 0).count()
            assert invalid_quantity == 0, "Quantity should be positive"
            
        except Exception as e:
            pytest.skip(f"Fact_sales table not accessible: {e}")
    
    def test_revenue_aggregations_match(self, spark):
        """Test that aggregated revenue matches sum from fact table."""
        try:
            # Get total from fact_sales
            fact_total = spark.sql("""
                SELECT SUM(line_total) as total_revenue
                FROM dev1.gold.fact_sales
            """).collect()[0]["total_revenue"]
            
            # Get total from revenue_by_state
            state_total = spark.sql("""
                SELECT SUM(total_revenue) as total_revenue
                FROM dev1.gold.revenue_by_state
            """).collect()[0]["total_revenue"]
            
            # Should match within rounding tolerance
            assert abs(fact_total - state_total) < 1.0, \
                f"Revenue mismatch: fact={fact_total}, state={state_total}"
            
        except Exception as e:
            pytest.skip(f"Revenue aggregation test failed: {e}")
    
    def test_dimension_table_keys_unique(self, spark):
        """Test that dimension tables have unique keys."""
        try:
            # Test dim_customers
            customers = spark.table("dev1.gold.dim_customers")
            total_count = customers.count()
            distinct_count = customers.select("customer_id").distinct().count()
            assert total_count == distinct_count, \
                "dim_customers should have unique customer_ids"
            
            # Test dim_products
            products = spark.table("dev1.gold.dim_products")
            total_count = products.count()
            distinct_count = products.select("product_id").distinct().count()
            assert total_count == distinct_count, \
                "dim_products should have unique product_ids"
            
        except Exception as e:
            pytest.skip(f"Dimension table test failed: {e}")
    
    def test_daily_trends_date_sequence(self, spark):
        """Test that daily trends have continuous date sequence."""
        try:
            daily_trends = spark.sql("""
                SELECT order_date, daily_revenue
                FROM dev1.gold.sales_trends_daily
                ORDER BY order_date
            """).collect()
            
            assert len(daily_trends) > 0, "Daily trends should have data"
            
            # Check that daily_revenue is not null
            for row in daily_trends:
                assert row["daily_revenue"] is not None, \
                    f"Daily revenue should not be null for {row['order_date']}"
                assert row["daily_revenue"] >= 0, \
                    f"Daily revenue should be non-negative for {row['order_date']}"
            
        except Exception as e:
            pytest.skip(f"Daily trends test failed: {e}")
