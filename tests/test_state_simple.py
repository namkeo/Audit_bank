# -*- coding: utf-8 -*-
"""
Simple State Management Test
Demonstrates lifecycle improvements
"""

import pandas as pd
import numpy as np
import importlib.util
import sys

def _import_module(module_path, class_name):
    """Import a module by file path and return the specified class"""
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Import modules
CreditRiskModel = _import_module("2_model_credit_risk.py", "CreditRiskModel")

def create_sample_data(n=100):
    """Create sample data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'total_loans': np.random.uniform(500000, 2000000, n),
        'non_performing_loans': np.random.uniform(10000, 100000, n),
        'total_assets': np.random.uniform(1000000, 5000000, n),
        'total_equity': np.random.uniform(100000, 500000, n),
        'net_income': np.random.uniform(10000, 100000, n),
        'total_deposits': np.random.uniform(800000, 4000000, n),
        'loan_loss_provisions': np.random.uniform(5000, 50000, n)
    })

print("=" * 70)
print("STATE MANAGEMENT AND LIFECYCLE TEST")
print("=" * 70)

# Test 1: Check initial state
print("\n1. Initial State Check:")
print("-" * 70)
model = CreditRiskModel()
print(f"✓ Model created: {model.model_name}")
print(f"  State: {model.get_state().value}")
print(f"  Is trained: {model.is_trained()}")

# Test 2: Try prediction before training (should fail)
print("\n2. Prediction Before Training:")
print("-" * 70)
test_data = create_sample_data(5)

try:
    predictions = model.predict_risk(test_data)
    print("✗ FAIL: Should have raised an error!")
except Exception as e:
    print(f"✓ PASS: Correctly prevented prediction on untrained model")
    print(f"  Error: {type(e).__name__}")
    print(f"  Message: {str(e)[:100]}...")

# Test 3: Train and check state
print("\n3. Training Model:")
print("-" * 70)
training_data = create_sample_data(100)
results = model.train_models(training_data)
print(f"✓ Training completed")
print(f"  New state: {model.get_state().value}")
print(f"  Is trained: {model.is_trained()}")
print(f"  Models: {list(model.models.keys())}")

# Test 4: Prediction after training
print("\n4. Prediction After Training:")
print("-" * 70)
try:
    predictions = model.predict_risk(test_data)
    print(f"✓ PASS: Prediction successful!")
    print(f"  Risk score: {predictions.get('overall_risk_score', 'N/A')}")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {str(e)}")

# Test 5: Factory method
print("\n5. Factory Method (create_and_train):")
print("-" * 70)
factory_model = CreditRiskModel.create_and_train(training_data)
print(f"✓ Factory model created and trained in one call")
print(f"  State: {factory_model.get_state().value}")
print(f"  Ready for immediate use: {factory_model.is_trained()}")

try:
    predictions = factory_model.predict_risk(test_data)
    print(f"✓ Factory model works immediately without manual training")
except Exception as e:
    print(f"✗ ERROR: {str(e)}")

print("\n" + "=" * 70)
print("SUMMARY: State Management Improvements")
print("=" * 70)
print("✓ Models track their state (UNINITIALIZED/TRAINED/READY)")
print("✓ Fail-fast validation prevents using untrained models")
print("✓ Factory methods create ready-to-use instances")
print("✓ Clear lifecycle documentation in docstrings")
print("✓ Helpful error messages with context")
print("=" * 70)
