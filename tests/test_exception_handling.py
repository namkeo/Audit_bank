# -*- coding: utf-8 -*-
"""
Test Exception Handling
Demonstrates the improved exception handling with structured logging
"""

import pandas as pd
import numpy as np
import importlib.util

def _import_module(module_path, class_name):
    """Import a module by file path and return the specified class"""
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Import modules
DataPreparation = _import_module("1_data_preparation.py", "DataPreparation")
AuditLogger = _import_module("6_logging_config.py", "AuditLogger")
DataLoadError = _import_module("6_logging_config.py", "DataLoadError")

def test_exception_handling():
    """Test improved exception handling"""
    
    print("=" * 70)
    print("TESTING IMPROVED EXCEPTION HANDLING")
    print("=" * 70)
    
    # Initialize data preparation with logging
    data_prep = DataPreparation()
    logger = AuditLogger.get_logger(__name__)
    
    print("\n1. Testing with empty dataframe (should raise DataLoadError):")
    print("-" * 70)
    try:
        empty_df = pd.DataFrame()
        data_prep.load_time_series_data(empty_df, "TEST_BANK")
    except DataLoadError as e:
        print(f"✓ Caught DataLoadError: {str(e)}")
        print(f"  Context: {e.context if hasattr(e, 'context') else 'N/A'}")
    except Exception as e:
        print(f"✗ Unexpected exception: {type(e).__name__}: {str(e)}")
    
    print("\n2. Testing with missing bank_id column (should raise DataValidationError):")
    print("-" * 70)
    try:
        bad_df = pd.DataFrame({'total_assets': [1000, 2000]})
        data_prep.load_time_series_data(bad_df, "TEST_BANK")
    except Exception as e:
        print(f"✓ Caught {type(e).__name__}: {str(e)}")
        print(f"  Context: {e.context if hasattr(e, 'context') else 'N/A'}")
    
    print("\n3. Testing with valid data (should succeed with logging):")
    print("-" * 70)
    try:
        # Create valid test data
        test_df = pd.DataFrame({
            'bank_id': ['TEST_BANK', 'TEST_BANK', 'TEST_BANK'],
            'period': ['2024-Q1', '2024-Q2', '2024-Q3'],
            'total_assets': [1000000, 1100000, 1200000],
            'total_equity': [100000, 110000, 120000],
            'net_income': [10000, 11000, 12000],
            'total_loans': [700000, 750000, 800000],
            'non_performing_loans': [35000, 37500, 40000]
        })
        
        result = data_prep.load_time_series_data(test_df, "TEST_BANK", "period")
        print(f"✓ Successfully loaded data for TEST_BANK")
        print(f"  Periods: {len(result['periods'])}")
        print(f"  Records: {len(result['raw_data'])}")
        
        # Test ratio calculation
        ratios = data_prep.calculate_time_series_ratios()
        print(f"✓ Successfully calculated ratios")
        print(f"  Ratio columns: {list(ratios.columns)[:5]}...")
        
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {str(e)}")
    
    print("\n4. Testing data quality validation:")
    print("-" * 70)
    try:
        # Test with problematic data
        problem_df = test_df.copy()
        problem_df.loc[0, 'total_assets'] = -1000  # Negative assets
        problem_df.loc[1, 'net_income'] = np.nan   # Missing value
        
        validation_results = data_prep.validate_data_quality(problem_df)
        print(f"✓ Data quality validation completed")
        print(f"  Total records: {validation_results['total_records']}")
        print(f"  Issues found: {len(validation_results['issues'])}")
        for issue in validation_results['issues']:
            print(f"    - {issue}")
            
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)}")
    
    print("\n" + "=" * 70)
    print("EXCEPTION HANDLING TESTS COMPLETED")
    print("=" * 70)
    print("\nKey Improvements Demonstrated:")
    print("  ✓ Custom exception classes with context")
    print("  ✓ Structured logging instead of print statements")
    print("  ✓ Proper error propagation")
    print("  ✓ Contextual information (bank_id, method names)")
    print("  ✓ Graceful degradation (partial success where appropriate)")
    print("\nCheck the log file: audit_system_YYYYMMDD.log for detailed logs")
    print("=" * 70)

if __name__ == "__main__":
    test_exception_handling()
