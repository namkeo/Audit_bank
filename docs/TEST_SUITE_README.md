# Unit Test Suite for Bank Audit System

## Overview
This comprehensive unit test suite validates critical financial calculations, risk scoring algorithms, and data integrity throughout the bank audit system.

## Test File
- **Location**: `test_critical_calculations.py`
- **Framework**: pytest
- **Coverage**: 6 test classes, 40+ individual test cases

## Running the Tests

### Run all tests
```bash
pytest test_critical_calculations.py -v
```

### Run specific test class
```bash
pytest test_critical_calculations.py::TestFinancialRatioCalculator -v
```

### Run specific test
```bash
pytest test_critical_calculations.py::TestFinancialRatioCalculator::test_npl_ratio_calculation -v
```

### Run with coverage report
```bash
pytest test_critical_calculations.py --cov=utility_functions --cov=bank_audit_system -v
```

## Test Classes and Coverage

### 1. TestFinancialRatioCalculator
**Purpose**: Validates all financial ratio calculations
**Tests**:
- NPL (Non-Performing Loans) ratio calculation
- NPL ratio edge cases (zero loans, zero NPL, NPL=Loans)
- ROA (Return on Assets) calculation
- ROE (Return on Equity) calculation
- Capital Adequacy Ratio (CAR)
- Liquidity Ratio
- Provision Coverage Ratio
- Loan-to-Deposit Ratio
- Growth metrics (deposit growth, asset growth)
- Growth calculation with edge cases (negative values, zero previous)
- Full ratio calculation method

**Critical Functions Tested**:
- `FinancialRatioCalculator.calculate_ratios()`
- `FinancialRatioCalculator.calculate_growth_metrics()`

### 2. TestRiskScoring
**Purpose**: Validates risk score aggregation and classification
**Tests**:
- Risk score aggregation from multiple indicators
- Risk level classification (LOW, MODERATE, HIGH, CRITICAL)
- Risk score boundary conditions (0, 0.5, 1.0)

**Critical Functions Tested**:
- `BaseRiskModel.calculate_risk_score()`
- `BaseRiskModel.classify_risk_level()`

### 3. TestHighRiskPeriodIdentification
**Purpose**: Validates detection of high-risk periods in time series
**Tests**:
- High-risk period identification with thresholds
- Threshold logic (NPL threshold, liquidity threshold)
- Multiple risk indicators combined
- Severity calculation

**Critical Functions Tested**:
- `identify_high_risk_periods()` standalone utility function
- Vectorized threshold detection

### 4. TestModelTraining
**Purpose**: Validates model training data validation
**Tests**:
- Sufficient training data detection
- Insufficient data handling
- NaN value detection
- Infinite value detection
- Empty data handling
- None data handling

**Critical Functions Tested**:
- `ModelTrainingPipeline.validate_training_data()`

### 5. TestEdgeCasesAndErrorHandling
**Purpose**: Validates robust error handling and edge case management
**Tests**:
- Division by zero handling
- NaN propagation in calculations
- Very large numbers (1e20 and beyond)
- Very small numbers (1e-20 and below)
- Negative values in ratios (loss-making banks)

### 6. TestDataIntegrity
**Purpose**: Validates data consistency and financial statement integrity
**Tests**:
- Ratio consistency (ROE = ROA × Asset Multiplier)
- Time series continuity (no gaps)
- Financial statement balance (Assets = Liabilities + Equity)
- Ratio bounds (0-1 ranges where applicable)

## Key Metrics and Thresholds

### Financial Ratios
| Ratio | Healthy Range | Warning Level | Critical |
|-------|---|---|---|
| NPL Ratio | < 3% | 3-5% | > 5% |
| Capital Adequacy Ratio | > 8% | 6-8% | < 6% |
| Liquidity Ratio | > 0.5 | 0.3-0.5 | < 0.3 |
| ROA | > 1% | 0-1% | < 0% |
| Loan-to-Deposit | 80-100% | 100-120% | > 120% |

### Risk Scores (0-1 scale)
- **LOW**: 0.0 - 0.25
- **MODERATE**: 0.25 - 0.50
- **HIGH**: 0.50 - 0.75
- **CRITICAL**: 0.75 - 1.0

## Expected Test Results

### Baseline Tests (Should Pass)
- All ratio calculations with valid data
- Risk classification logic
- Data validation with proper data
- Edge case handling for NaN/Inf

### Known Limitations
- Some tests may skip if optional functions aren't available
- Tests use approximate equality (`pytest.approx`) for floating-point comparisons with 0.001 tolerance
- Model training tests use synthetic data and don't validate actual model performance

## Integration with Main System

These tests validate the core calculation engine used by:
- `5_bank_audit_system.py` - Main audit orchestrator
- `2_model_credit_risk.py` - Credit risk assessment
- `2_model_liquidity_risk.py` - Liquidity risk assessment
- `2_model_anomaly_detection.py` - Anomaly/fraud detection
- `3_reporting_analysis.py` - Report generation

## Adding New Tests

To add new tests:

1. **Identify the function** to test (e.g., `new_calculation_function()`)
2. **Create a test class** following naming convention `TestNewFeature`
3. **Add test methods** with descriptive names starting with `test_`
4. **Use fixtures** for complex test data
5. **Add assertions** with appropriate tolerances for floats

Example:
```python
class TestNewFeature:
    @pytest.fixture
    def sample_data(self):
        return {...}
    
    def test_feature_basic(self, sample_data):
        result = new_calculation_function(sample_data)
        assert result == expected_value
    
    def test_feature_edge_case(self):
        # Test edge case
        assert condition
```

## Logging Integration

All tests respect the logging configuration from `6_logging_config.py`. To debug test failures:
- Run with logging enabled: `pytest test_critical_calculations.py -v -s`
- Check console output for debug/warning/error messages
- Use `logger` in your test code: `logger.debug("Test message")`

## Performance Notes

- Full test suite runs in ~5-10 seconds (depending on system)
- Each test class can be run independently for faster iteration
- Use `-k` flag to filter tests: `pytest -k "ratio" -v`

## Maintenance

### Regular Updates Needed When:
1. New ratio calculations are added
2. Risk scoring weights change
3. Threshold values are updated
4. New model types are integrated
5. Data validation rules change

### Regression Testing
Run the full suite before merging changes to ensure no calculations were accidentally modified:
```bash
pytest test_critical_calculations.py -v --tb=short
```

## References

- [pytest Documentation](https://docs.pytest.org/)
- [NumPy Testing Guide](https://numpy.org/doc/stable/reference/testing.html)
- [Pandas Testing Utilities](https://pandas.pydata.org/docs/reference/testing.html)
