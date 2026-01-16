# -*- coding: utf-8 -*-
"""
Test script to verify all imports are working correctly
Run this to diagnose any import issues
"""

import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("\n" + "="*80)
print("Testing imports...")
print("="*80 + "\n")

def test_import(description, import_func):
    """Test an import and report results"""
    try:
        import_func()
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}")
        print(f"   Error: {e}")
        return False

results = []

# Test 1: Import utility functions
def test_utility():
    import importlib.util
    spec = importlib.util.spec_from_file_location("utils", "4_utility_functions.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    assert hasattr(utils, 'FinancialRatioCalculator')
    assert hasattr(utils, 'TimeSeriesFeaturePreparation')
    assert hasattr(utils, 'ModelTrainingPipeline')

results.append(test_import("4_utility_functions.py imports", test_utility))

# Test 2: Import data preparation
def test_data_prep():
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_prep", "1_data_preparation.py")
    data_prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_prep)
    assert hasattr(data_prep, 'DataPreparation')

results.append(test_import("1_data_preparation.py imports", test_data_prep))

# Test 3: Import base risk model
def test_base_risk():
    import importlib.util
    spec = importlib.util.spec_from_file_location("base_risk", "2_model_base_risk.py")
    base_risk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_risk)
    assert hasattr(base_risk, 'BaseRiskModel')

results.append(test_import("2_model_base_risk.py imports", test_base_risk))

# Test 4: Import credit risk model
def test_credit_risk():
    import importlib.util
    spec = importlib.util.spec_from_file_location("credit_risk", "2_model_credit_risk.py")
    credit_risk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(credit_risk)
    assert hasattr(credit_risk, 'CreditRiskModel')

results.append(test_import("2_model_credit_risk.py imports", test_credit_risk))

# Test 5: Import liquidity risk model
def test_liquidity_risk():
    import importlib.util
    spec = importlib.util.spec_from_file_location("liquidity_risk", "2_model_liquidity_risk.py")
    liquidity_risk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(liquidity_risk)
    assert hasattr(liquidity_risk, 'LiquidityRiskModel')

results.append(test_import("2_model_liquidity_risk.py imports", test_liquidity_risk))

# Test 6: Import anomaly detection model
def test_anomaly_detection():
    import importlib.util
    spec = importlib.util.spec_from_file_location("anomaly", "2_model_anomaly_detection.py")
    anomaly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anomaly)
    assert hasattr(anomaly, 'AnomalyDetectionModel')

results.append(test_import("2_model_anomaly_detection.py imports", test_anomaly_detection))

# Test 7: Import reporting analysis
def test_reporting():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reporting", "3_reporting_analysis.py")
    reporting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporting)
    assert hasattr(reporting, 'ReportingAnalysis')

results.append(test_import("3_reporting_analysis.py imports", test_reporting))

# Test 8: Import bank audit system
def test_bank_audit():
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit", "5_bank_audit_system.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert hasattr(audit, 'BankAuditSystem')

results.append(test_import("5_bank_audit_system.py imports", test_bank_audit))

# Test 9: Import example usage
def test_example():
    import importlib.util
    spec = importlib.util.spec_from_file_location("example", "0_example_usage.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

results.append(test_import("0_example_usage.py imports", test_example))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
passed = sum(results)
total = len(results)
print(f"\nPassed: {passed}/{total}")

if passed == total:
    print("\n✅ All imports successful! The system is ready to use.")
    sys.exit(0)
else:
    print(f"\n❌ {total - passed} import(s) failed. Please check the errors above.")
    sys.exit(1)
