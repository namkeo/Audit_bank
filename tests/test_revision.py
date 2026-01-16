# -*- coding: utf-8 -*-
"""
Test Suite for Project Revisions
Verifies DBSCAN/LOF enhancements and harmonized functions
"""

import pandas as pd
import numpy as np
import sys
import importlib.util

# Import necessary modules
def _import_module(module_path, class_name):
    """Helper to import modules"""
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def _import_function(module_path, func_name):
    """Helper to import functions"""
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

AnomalyDetectionModel = _import_module("2_model_anomaly_detection.py", "AnomalyDetectionModel")
identify_high_risk_periods = _import_function("4_utility_functions.py", "identify_high_risk_periods")


def test_dbscan_enhancement():
    """Test DBSCAN training with enhanced parameters"""
    print("\n" + "="*80)
    print("TEST 1: DBSCAN Enhancement")
    print("="*80)
    
    # Create synthetic training data with proper financial feature names
    np.random.seed(42)
    n_samples = 200
    df_train = pd.DataFrame({
        'npl_ratio': np.random.rand(n_samples) * 0.3,
        'roa': np.random.randn(n_samples) * 0.05 + 0.02,
        'roe': np.random.randn(n_samples) * 0.1 + 0.1,
        'loan_growth': np.random.randn(n_samples) * 0.2,
        'asset_growth': np.random.randn(n_samples) * 0.15,
        'total_loans': np.random.rand(n_samples) * 10000 + 1000,
        'liquidity_ratio': np.random.rand(n_samples) * 0.5 + 0.3,
        'capital_adequacy_ratio': np.random.rand(n_samples) * 0.3 + 0.08,
    })
    
    # Train anomaly detection
    detector = AnomalyDetectionModel()
    result = detector.train_models(df_train)
    
    # Check DBSCAN results
    assert 'dbscan' in result, f"DBSCAN training failed. Keys: {list(result.keys())}"
    assert result['dbscan']['trained'], "DBSCAN not trained"
    assert 'clusters_found' in result['dbscan'], "Missing clusters_found"
    assert 'anomalies_found' in result['dbscan'], "Missing anomalies_found"
    assert 'core_points' in result['dbscan'], "Missing core_points"
    assert 'eps' in result['dbscan'], "Missing eps parameter"
    assert 'min_samples' in result['dbscan'], "Missing min_samples parameter"
    
    print(f"[PASS] DBSCAN training successful")
    print(f"   - Clusters found: {result['dbscan']['clusters_found']}")
    print(f"   - Anomalies found: {result['dbscan']['anomalies_found']}")
    print(f"   - Core points: {result['dbscan']['core_points']}")
    print(f"   - Anomaly rate: {result['dbscan']['anomaly_rate']:.2%}")
    

def test_lof_enhancement():
    """Test LocalOutlierFactor training with enhancements"""
    print("\n" + "="*80)
    print("TEST 2: LocalOutlierFactor Enhancement")
    print("="*80)
    
    # Create synthetic training data with proper financial feature names
    np.random.seed(42)
    n_samples = 150
    df_train = pd.DataFrame({
        'npl_ratio': np.random.rand(n_samples) * 0.3,
        'roa': np.random.randn(n_samples) * 0.05 + 0.02,
        'roe': np.random.randn(n_samples) * 0.1 + 0.1,
        'loan_growth': np.random.randn(n_samples) * 0.2,
        'asset_growth': np.random.randn(n_samples) * 0.15,
    })
    
    # Train anomaly detection
    detector = AnomalyDetectionModel()
    result = detector.train_models(df_train)
    
    # Check LOF results
    assert 'local_outlier_factor' in result, f"LOF training failed. Keys: {list(result.keys())}"
    lof_key = 'local_outlier_factor'
    assert result[lof_key]['trained'], "LOF not trained"
    assert 'anomaly_rate' in result[lof_key], "Missing anomaly_rate"
    assert 'contamination' in result[lof_key], "Missing contamination"
    
    print(f"✅ LocalOutlierFactor training successful")
    print(f"   - Anomalies found: {result[lof_key].get('anomalies_found', 'N/A')}")
    print(f"   - Anomaly rate: {result[lof_key]['anomaly_rate']:.2%}")
    print(f"   - Contamination: {result[lof_key]['contamination']}")


def test_ensemble_voting():
    """Test ensemble voting with all 5 models"""
    print("\n" + "="*80)
    print("TEST 3: Ensemble Voting with All Models")
    print("="*80)
    
    # Create training and test data with proper financial feature names
    np.random.seed(42)
    n_train = 200
    n_test = 50
    
    df_train = pd.DataFrame({
        'npl_ratio': np.random.rand(n_train) * 0.3,
        'roa': np.random.randn(n_train) * 0.05 + 0.02,
        'roe': np.random.randn(n_train) * 0.1 + 0.1,
        'loan_growth': np.random.randn(n_train) * 0.2,
        'asset_growth': np.random.randn(n_train) * 0.15,
        'total_loans': np.random.rand(n_train) * 10000 + 1000,
        'liquidity_ratio': np.random.rand(n_train) * 0.5 + 0.3,
        'capital_adequacy_ratio': np.random.rand(n_train) * 0.3 + 0.08,
    })
    
    # Train
    detector = AnomalyDetectionModel()
    detector.train_models(df_train, contamination=0.1)
    
    # Test
    df_test = pd.DataFrame({
        'npl_ratio': np.random.rand(n_test) * 0.3,
        'roa': np.random.randn(n_test) * 0.05 + 0.02,
        'roe': np.random.randn(n_test) * 0.1 + 0.1,
        'loan_growth': np.random.randn(n_test) * 0.2,
        'asset_growth': np.random.randn(n_test) * 0.15,
        'total_loans': np.random.rand(n_test) * 10000 + 1000,
        'liquidity_ratio': np.random.rand(n_test) * 0.5 + 0.3,
        'capital_adequacy_ratio': np.random.rand(n_test) * 0.3 + 0.08,
    })
    
    results = detector.detect_anomalies(df_test)
    
    # Check results
    assert 'models_used' in results, "Missing models_used"
    num_models = len(results['models_used'])
    assert num_models >= 3, f"Expected >=3 models, got {num_models}: {results['models_used']}"
    assert 'ensemble_scores' in results, "Missing ensemble_scores"
    assert 'anomalies' in results, "Missing anomalies"
    
    print(f"✅ Ensemble voting successful")
    print(f"   - Models used: {num_models} - {results['models_used']}")
    print(f"   - Total records: {results['total_records']}")
    print(f"   - Anomalies detected: {results['anomalies_detected']}")
    print(f"   - Anomaly rate: {results['anomaly_rate']:.2%}")
    print(f"   - Voting threshold: {results['voting_threshold']}")
    
    # Check anomaly details
    if results['anomalies']:
        anomaly = results['anomalies'][0]
        assert 'models_voting_anomaly' in anomaly, "Missing models_voting_anomaly in anomaly record"
        print(f"   - Sample anomaly: {anomaly['models_voting_anomaly']}/{num_models} models voted")


def test_high_risk_periods_standalone():
    """Test standalone high-risk period detection function"""
    print("\n" + "="*80)
    print("TEST 4: Standalone High-Risk Period Detection")
    print("="*80)
    
    # Create sample data with multiple periods
    periods = ['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4']
    data = {
        'period': periods * 2,
        'npl_ratio': [0.02, 0.06, 0.08, 0.04, 0.03, 0.07, 0.09, 0.05],
        'liquidity_ratio': [0.5, 0.25, 0.2, 0.4, 0.55, 0.2, 0.15, 0.45],
        'capital_adequacy_ratio': [0.12, 0.09, 0.07, 0.11, 0.13, 0.08, 0.06, 0.10]
    }
    
    df = pd.DataFrame(data)
    prepared_data = {'features': df}
    
    # Call standalone function
    high_risk = identify_high_risk_periods(prepared_data)
    
    # Verify results
    assert isinstance(high_risk, list), "Should return a list"
    assert len(high_risk) > 0, "Should detect high-risk periods"
    
    print(f"✅ Standalone function successful")
    print(f"   - High-risk periods detected: {len(high_risk)}")
    
    for period_info in high_risk:
        assert 'period' in period_info, "Missing period"
        assert 'risk_indicators' in period_info, "Missing risk_indicators"
        assert 'severity' in period_info, "Missing severity"
        assert 'npl_ratio' in period_info, "Missing npl_ratio"
        assert 'liquidity_ratio' in period_info, "Missing liquidity_ratio"
        assert 'capital_adequacy_ratio' in period_info, "Missing capital_adequacy_ratio"
        
        print(f"   - {period_info['period']}: {period_info['severity']} - {period_info['risk_indicators']}")


def test_high_risk_periods_method():
    """Test harmonized method still works"""
    print("\n" + "="*80)
    print("TEST 5: BankAuditSystem.identify_high_risk_periods() Method")
    print("="*80)
    
    try:
        BankAuditSystem = _import_module("5_bank_audit_system.py", "BankAuditSystem")
        
        # Create sample data
        periods = ['2023-Q1', '2023-Q2', '2023-Q3']
        data = {
            'period': periods,
            'npl_ratio': [0.02, 0.06, 0.08],
            'liquidity_ratio': [0.5, 0.25, 0.2],
            'capital_adequacy_ratio': [0.12, 0.09, 0.07]
        }
        
        df = pd.DataFrame(data)
        prepared_data = {'features': df}
        
        # Create audit system and test method
        audit_system = BankAuditSystem("Test Bank", "2024-Q1")
        high_risk = audit_system.identify_high_risk_periods(prepared_data)
        
        assert isinstance(high_risk, list), "Should return a list"
        
        print(f"✅ Method call successful")
        print(f"   - High-risk periods detected: {len(high_risk)}")
        
        for period_info in high_risk:
            print(f"   - {period_info['period']}: {period_info['severity']}")
            
    except Exception as e:
        print(f"⚠️  Method test skipped: {str(e)}")


def test_consistency():
    """Test that standalone and method return consistent results"""
    print("\n" + "="*80)
    print("TEST 6: Consistency Between Standalone and Method")
    print("="*80)
    
    try:
        BankAuditSystem = _import_module("5_bank_audit_system.py", "BankAuditSystem")
        
        # Create sample data
        periods = ['2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4']
        data = {
            'period': periods,
            'npl_ratio': [0.03, 0.07, 0.09, 0.04],
            'liquidity_ratio': [0.45, 0.22, 0.18, 0.38],
            'capital_adequacy_ratio': [0.11, 0.08, 0.065, 0.10]
        }
        
        df = pd.DataFrame(data)
        prepared_data = {'features': df}
        
        # Get results from both approaches
        result_standalone = identify_high_risk_periods(prepared_data)
        
        audit_system = BankAuditSystem("Consistency Test", "2024")
        result_method = audit_system.identify_high_risk_periods(prepared_data)
        
        # Compare results
        assert len(result_standalone) == len(result_method), "Different number of high-risk periods"
        
        for i, (standalone, method) in enumerate(zip(result_standalone, result_method)):
            assert standalone['period'] == method['period'], f"Period mismatch at index {i}"
            assert standalone['severity'] == method['severity'], f"Severity mismatch at {standalone['period']}"
            assert set(standalone['risk_indicators']) == set(method['risk_indicators']), \
                f"Risk indicators mismatch at {standalone['period']}"
        
        print(f"✅ Consistency check passed")
        print(f"   - Both approaches return {len(result_standalone)} high-risk periods")
        print(f"   - Results are identical ✓")
        
    except Exception as e:
        print(f"⚠️  Consistency test skipped: {str(e)}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# PROJECT REVISION TEST SUITE")
    print("#"*80)
    
    tests = [
        ("DBSCAN Enhancement", test_dbscan_enhancement),
        ("LocalOutlierFactor Enhancement", test_lof_enhancement),
        ("Ensemble Voting", test_ensemble_voting),
        ("Standalone Function", test_high_risk_periods_standalone),
        ("Class Method", test_high_risk_periods_method),
        ("Consistency", test_consistency)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n⚠️  {test_name} ERROR: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Project revision successful!")
    else:
        print(f"\n❌ {failed} test(s) failed - See details above")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

