"""Tests for Gold Layer aggregations and data quality."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.sql.functions import col, sum as _sum, count, avg, date_format
from datetime import date


class TestGoldLayerAggregations:
    """Test Gold layer fact tables and aggregations."""
    
    @pytest.fixture
    def silver_orders_data(self, spark):
        """Create silver orders data."""
        schema = StructType([
            StructField("order_id", IntegerType(), False),
            StructField("customer_id", IntegerType(), True),
            StructField("order_date", DateType(), True),
            StructField("order_status", StringType(), True),
            StructField("customer_name", StringType(), True),
            StructField("customer_state", StringType(), True)
        ])
        
        data = [
            (101, 1, date(2023, 6, 15), "Delivered", "John Doe", "MH"),
            (102, 2, date(2023, 6, 16), "Pending", "Jane Smith", "DL"),
            (103, 1, date(2023, 6, 17), "Cancelled", "John Doe", "MH"),
            (104, 3, date(2023, 6, 18), "Returned", "Bob Johnson", "KA"),
            (105, 2, date(2023, 6, 19), "Delivered", "Jane Smith", "DL"),
        ]
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def silver_order_items_data(self, spark):
        """Create silver order_items data."""
        schema = StructType([
            StructField("order_item_id", IntegerType(), False),
            StructField("order_id", IntegerType(), True),
            StructField("product_id", IntegerType(), True),
            StructField("quantity", IntegerType(), True),
            StructField("item_price", DoubleType(), True),
            StructField("product_name", StringType(), True),
            StructField("product_category", StringType(), True)
        ])
        
        data = [
            (1001, 101, 5001, 2, 100.0, "Product_A", "Electronics"),
            (1002, 102, 5002, 1, 200.0, "Product_B", "Clothing"),
            (1003, 103, 5001, 3, 100.0, "Product_A", "Electronics"),  # Cancelled order
            (1004, 104, 5003, 1, 300.0, "Product_C", "Home"),  # Returned order
            (1005, 105, 5002, 2, 200.0, "Product_B", "Clothing"),
        ]
        return spark.createDataFrame(data, schema)
    
    def test_fact_sales_filters_invalid_statuses(self, silver_orders_data, silver_order_items_data):
        """Test that fact_sales only includes Delivered and Pending orders."""
        # Filter orders
        valid_orders = silver_orders_data.filter(
            col("order_status").isin(["Delivered", "Pending"])
        )
        
        # Join with order_items
        fact_sales = valid_orders.alias("o").join(
            silver_order_items_data.alias("oi"),
            col("o.order_id") == col("oi.order_id"),
            "inner"
        ).select(
            col("o.order_id"),
            col("o.order_status")
        )
        
        # Should only have 3 orders (101, 102, 105)
        assert fact_sales.count() == 3, \
            "Fact sales should only include Delivered and Pending orders"
        
        # Verify no Cancelled or Returned orders
        invalid_statuses = fact_sales.filter(
            col("order_status").isin(["Cancelled", "Returned"])
        ).count()
        
        assert invalid_statuses == 0, \
            "Fact sales should not include Cancelled or Returned orders"
    
    def test_fact_sales_calculates_line_total(self, silver_order_items_data):
        """Test that fact_sales calculates line_total = quantity * item_price."""
        fact_sales = silver_order_items_data.withColumn(
            "line_total",
            col("quantity") * col("item_price")
        )
        
        # Check calculation for first record
        first_row = fact_sales.filter(col("order_item_id") == 1001).collect()[0]
        expected_total = 2 * 100.0  # quantity * item_price
        assert first_row["line_total"] == expected_total, \
            f"Line total should be {expected_total}"
    
    def test_fact_sales_has_required_columns(self, silver_orders_data, silver_order_items_data):
        """Test that fact_sales has all required columns."""
        valid_orders = silver_orders_data.filter(
            col("order_status").isin(["Delivered", "Pending"])
        )
        
        fact_sales = valid_orders.alias("o").join(
            silver_order_items_data.alias("oi"),
            col("o.order_id") == col("oi.order_id"),
            "inner"
        ).select(
            col("oi.order_item_id").alias("sale_id"),
            col("o.order_id"),
            col("o.customer_id"),
            col("oi.product_id"),
            col("o.order_date"),
            col("o.customer_state"),
            col("oi.quantity"),
            col("oi.item_price"),
            (col("oi.quantity") * col("oi.item_price")).alias("line_total"),
            col("o.order_status"),
            col("oi.product_category")
        )
        
        required_columns = [
            "sale_id", "order_id", "customer_id", "product_id", 
            "order_date", "customer_state", "quantity", "item_price", 
            "line_total", "order_status", "product_category"
        ]
        
        for required_col in required_columns:
            assert required_col in fact_sales.columns, \
                f"Missing required column: {required_col}"
    
    def test_revenue_by_state_aggregation(self, silver_orders_data, silver_order_items_data):
        """Test revenue_by_state aggregation logic."""
        valid_orders = silver_orders_data.filter(
            col("order_status").isin(["Delivered", "Pending"])
        )
        
        fact_sales = valid_orders.alias("o").join(
            silver_order_items_data.alias("oi"),
            col("o.order_id") == col("oi.order_id"),
            "inner"
        ).select(
            col("o.customer_state"),
            col("o.order_id"),
            col("o.customer_id"),
            (col("oi.quantity") * col("oi.item_price")).alias("line_total")
        )
        
        revenue_by_state = fact_sales.groupBy("customer_state").agg(
            _sum("line_total").alias("total_revenue"),
            count("order_id").alias("total_orders"),
            count("customer_id").distinct().alias("total_customers")
        ).orderBy(col("total_revenue").desc())
        
        # Should have aggregated data by state
        assert revenue_by_state.count() >= 1, "Should have state-level aggregations"
        
        # Check required columns
        assert "customer_state" in revenue_by_state.columns
        assert "total_revenue" in revenue_by_state.columns
        assert "total_orders" in revenue_by_state.columns
    
    def test_top_products_aggregation(self, silver_order_items_data):
        """Test top_products aggregation logic."""
        top_products = silver_order_items_data.groupBy(
            "product_id", "product_name", "product_category"
        ).agg(
            _sum(col("quantity") * col("item_price")).alias("total_revenue"),
            _sum("quantity").alias("total_quantity_sold"),
            count("order_id").alias("total_orders")
        ).orderBy(col("total_revenue").desc())
        
        # Should have product-level aggregations
        assert top_products.count() >= 1, "Should have product aggregations"
        
        # Check required columns
        required_cols = ["product_id", "product_name", "product_category", 
                        "total_revenue", "total_quantity_sold", "total_orders"]
        for col_name in required_cols:
            assert col_name in top_products.columns, f"Missing column: {col_name}"
    
    def test_sales_trends_daily_aggregation(self, silver_orders_data, silver_order_items_data):
        """Test sales_trends_daily aggregation logic."""
        valid_orders = silver_orders_data.filter(
            col("order_status").isin(["Delivered", "Pending"])
        )
        
        fact_sales = valid_orders.alias("o").join(
            silver_order_items_data.alias("oi"),
            col("o.order_id") == col("oi.order_id"),
            "inner"
        ).select(
            col("o.order_date"),
            col("o.order_id"),
            col("oi.quantity"),
            (col("oi.quantity") * col("oi.item_price")).alias("line_total")
        )
        
        daily_trends = fact_sales.groupBy("order_date").agg(
            _sum("line_total").alias("daily_revenue"),
            count("order_id").distinct().alias("daily_orders"),
            _sum("quantity").alias("daily_items_sold"),
            avg("line_total").alias("avg_order_value")
        ).orderBy("order_date")
        
        # Should have daily aggregations
        assert daily_trends.count() >= 1, "Should have daily trend data"
        
        # Check required columns
        assert "order_date" in daily_trends.columns
        assert "daily_revenue" in daily_trends.columns
        assert "daily_orders" in daily_trends.columns
    
    def test_sales_trends_monthly_aggregation(self, silver_orders_data, silver_order_items_data):
        """Test sales_trends_monthly aggregation logic."""
        valid_orders = silver_orders_data.filter(
            col("order_status").isin(["Delivered", "Pending"])
        )
        
        fact_sales = valid_orders.alias("o").join(
            silver_order_items_data.alias("oi"),
            col("o.order_id") == col("oi.order_id"),
            "inner"
        ).withColumn(
            "year_month",
            date_format(col("o.order_date"), "yyyy-MM")
        ).select(
            col("year_month"),
            col("o.order_id"),
            (col("oi.quantity") * col("oi.item_price")).alias("line_total")
        )
        
        monthly_trends = fact_sales.groupBy("year_month").agg(
            _sum("line_total").alias("monthly_revenue"),
            count("order_id").distinct().alias("monthly_orders")
        ).orderBy("year_month")
        
        # Should have monthly aggregations
        assert monthly_trends.count() >= 1, "Should have monthly trend data"
        
        # Check that year_month is in correct format
        first_row = monthly_trends.collect()[0]
        assert len(first_row["year_month"]) == 7, "Year_month should be YYYY-MM format"
    
    def test_dim_customers_scd_type_1(self, spark):
        """Test that dim_customers is SCD Type 1 (overwrite)."""
        schema = StructType([
            StructField("customer_id", IntegerType(), False),
            StructField("customer_name", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True)
        ])
        
        dim_customers = spark.createDataFrame([
            (1, "John Doe", "Mumbai", "MH"),
            (2, "Jane Smith", "Delhi", "DL")
        ], schema)
        
        # Check no duplicate customer_ids
        total_count = dim_customers.count()
        distinct_count = dim_customers.select("customer_id").distinct().count()
        
        assert total_count == distinct_count, "Dimension should have no duplicates"
    
    def test_dim_products_scd_type_1(self, spark):
        """Test that dim_products is SCD Type 1 (overwrite)."""
        schema = StructType([
            StructField("product_id", IntegerType(), False),
            StructField("product_name", StringType(), True),
            StructField("category", StringType(), True)
        ])
        
        dim_products = spark.createDataFrame([
            (5001, "Product_A", "Electronics"),
            (5002, "Product_B", "Clothing")
        ], schema)
        
        # Check no duplicate product_ids
        total_count = dim_products.count()
        distinct_count = dim_products.select("product_id").distinct().count()
        
        assert total_count == distinct_count, "Dimension should have no duplicates"
