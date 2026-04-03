# ETL Pipeline Tests

Comprehensive test suite for the Bronze-Silver-Gold medallion architecture ETL pipeline.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_config.py           # Configuration validation tests
├── test_bronze_layer.py     # Bronze layer ingestion tests
├── test_silver_layer.py     # Silver layer transformation tests
├── test_gold_layer.py       # Gold layer aggregation tests
├── test_integration.py      # End-to-end integration tests
└── README.md               # This file
```

## Test Categories

### 1. Configuration Tests (`test_config.py`)
- Validates `config.json` structure
- Checks catalog names (dev1, uat1, prod1)
- Verifies schema definitions (bronze, silver, gold)
- Ensures table configurations and primary keys

### 2. Bronze Layer Tests (`test_bronze_layer.py`)
- Schema validation with metadata columns
- Primary key integrity checks
- Incremental load filter logic
- MERGE UPSERT simulation
- Control table structure
- Partition column validation

### 3. Silver Layer Tests (`test_silver_layer.py`)
- Deduplication logic (row_number over window)
- Keeps latest record based on `updated_at`
- LEFT JOIN enrichment with dimension tables
- Orphan record preservation
- Control table tracking

### 4. Gold Layer Tests (`test_gold_layer.py`)
- Fact table filters (Delivered, Pending only)
- Line total calculation (quantity × item_price)
- Revenue by state aggregation
- Top products aggregation
- Daily and monthly trends
- Dimension table SCD Type 1 validation

### 5. Integration Tests (`test_integration.py`)
- Bronze/Silver/Gold table existence
- Data lineage validation
- End-to-end data quality checks
- Revenue aggregation reconciliation
- Dimension table uniqueness

## Running Tests

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv (recommended for Databricks Connect)
uv pip install -r requirements.txt
```

### Run All Tests

```bash
# From project root
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=html --cov-report=term
```

### Run Specific Test Files

```bash
# Configuration tests only
pytest tests/test_config.py -v

# Bronze layer tests only
pytest tests/test_bronze_layer.py -v

# Silver layer tests only
pytest tests/test_silver_layer.py -v

# Gold layer tests only
pytest tests/test_gold_layer.py -v

# Integration tests only
pytest tests/test_integration.py -v
```

### Run Tests by Marker

```bash
# Unit tests only (no database access required)
pytest -m unit -v

# Integration tests only (requires database access)
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v
```

### Run Tests in Parallel

```bash
# Using pytest-xdist for parallel execution
pytest tests/ -n auto -v
```

## Test Fixtures

### `spark` Fixture
Provides a Databricks Connect Spark session for all tests.

```python
def test_example(spark):
    df = spark.createDataFrame([(1, "test")], ["id", "value"])
    assert df.count() == 1
```

### `load_fixture` Fixture
Loads JSON or CSV test data from `fixtures/` directory.

```python
def test_with_fixture(load_fixture):
    data = load_fixture("sample_data.json")
    assert data.count() > 0
```

## CI/CD Integration

Tests are automatically executed in the GitHub Actions pipeline:

1. **On Pull Requests**: All tests run
2. **Before Dev Deployment**: Tests must pass
3. **Before QA Deployment**: Tests + validation
4. **Before Prod Deployment**: Tests + smoke tests

See `.github/workflows/databricks-cicd.yml` for details.

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
from pyspark.sql import SparkSession

class TestMyFeature:
    """Test my feature description."""
    
    @pytest.fixture
    def sample_data(self, spark):
        """Create sample test data."""
        return spark.createDataFrame([(1, "test")], ["id", "value"])
    
    def test_feature_works(self, sample_data):
        """Test that feature works correctly."""
        assert sample_data.count() == 1
```

## Troubleshooting

### Common Issues

1. **Databricks Connect not configured**
   ```bash
   databricks configure --token
   ```

2. **Missing test dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Tables not found in integration tests**
   - Ensure you've run the ETL notebooks first
   - Check catalog permissions (dev1, uat1, prod1)
   - Integration tests will skip if tables are inaccessible

## Coverage Reports

After running tests with coverage, open the HTML report:

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html  # macOS
```

Target: **80%+ code coverage**
