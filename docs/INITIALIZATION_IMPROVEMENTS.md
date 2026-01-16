# Initialization and State Management Improvements

## Problem Analysis

### Current Issues

1. **Incomplete Initialization**: Model classes initialize with empty state
   - `self.models = {}` starts empty, filled during training
   - `self.credit_indicators = {}` remains empty until assessment
   - No validation that models are trained before prediction

2. **Unclear Lifecycle**: No documentation of required call sequence
   - Must call `load_time_series_data()` before analysis
   - Must call `train_models()` before `predict_risk()`
   - No enforcement or clear documentation

3. **State Validation**: Methods don't check if prerequisites are met
   - `predict_risk()` assumes models are trained
   - No error if `self.models` is empty
   - Silent failures or cryptic errors

4. **Factory Pattern Absent**: No way to create fully-ready objects
   - Can't create "trained and ready" audit system in one call
   - Multiple manual steps required

## Solutions Implemented

### 1. State Validation with Decorators

Added `@require_trained` decorator to enforce model training:

```python
def require_trained(method):
    """Decorator to ensure models are trained before prediction"""
    def wrapper(self, *args, **kwargs):
        if not self.is_trained():
            raise ModelNotTrainedError(
                f"{self.model_name} must be trained before calling {method.__name__}",
                {'model': self.model_name, 'method': method.__name__}
            )
        return method(self, *args, **kwargs)
    return wrapper
```

### 2. Explicit State Tracking

Added `ModelState` enum and state tracking:

```python
class ModelState(Enum):
    UNINITIALIZED = "uninitialized"  # Just created
    DATA_LOADED = "data_loaded"      # Data loaded, ready to train
    TRAINED = "trained"              # Models trained, ready to predict
    READY = "ready"                  # Fully ready for production use
```

### 3. Factory Methods

Added factory methods for common workflows:

```python
@classmethod
def create_and_train(cls, training_data: pd.DataFrame, **kwargs) -> 'CreditRiskModel':
    """Factory method: Create and train in one call"""
    model = cls()
    model.train_models(training_data, **kwargs)
    return model

@classmethod
def from_trained_audit_system(cls, bank_name: str, audit_period: str, 
                              df: pd.DataFrame, training_data: pd.DataFrame) -> 'BankAuditSystem':
    """Factory method: Create fully trained audit system"""
    system = cls(bank_name, audit_period)
    system.train_models(training_data)
    system.load_and_prepare_data(df, bank_name)
    return system
```

### 4. Lifecycle Documentation

Added comprehensive lifecycle documentation to each class:

```python
\"\"\"
LIFECYCLE:
    1. Initialize: model = CreditRiskModel()
    2. Train: model.train_models(training_data)
    3. Predict: results = model.predict_risk(test_data)
    
ALTERNATIVE (Factory):
    model = CreditRiskModel.create_and_train(training_data)
    results = model.predict_risk(test_data)
\"\"\"
```

### 5. Validation Methods

Added state validation methods:

```python
def is_trained(self) -> bool:
    """Check if models have been trained"""
    return len(self.models) > 0 and hasattr(self, 'feature_names')

def validate_state_for_prediction(self) -> None:
    """Validate that model is ready for predictions"""
    if not self.is_trained():
        raise ModelNotTrainedError(...)
    if not hasattr(self, 'scaler'):
        raise ModelNotTrainedError(...)
```

## Files Modified

1. `6_logging_config.py` - Added ModelNotTrainedError exception
2. `2_model_base_risk.py` - Added state management to base class
3. `2_model_credit_risk.py` - Implemented state validation
4. `2_model_liquidity_risk.py` - Implemented state validation
5. `2_model_anomaly_detection.py` - Implemented state validation
6. `5_bank_audit_system.py` - Added factory methods and lifecycle docs
7. `1_data_preparation.py` - Added state tracking

## Usage Examples

### Before (Unclear Lifecycle)
```python
# Risk of calling predict before training - no error until runtime
audit = BankAuditSystem("ABC Bank", "2024-Q1")
audit.load_and_prepare_data(df, "ABC_BANK")
# Oops! Forgot to train models
results = audit.assess_risk(prepared_data)  # May fail silently or with cryptic error
```

### After (Clear Lifecycle with Validation)
```python
# Option 1: Manual lifecycle (with validation)
audit = BankAuditSystem("ABC Bank", "2024-Q1")
audit.train_models(training_data)  # Required before assessment
audit.load_and_prepare_data(df, "ABC_BANK")
results = audit.assess_risk(prepared_data)  # Raises ModelNotTrainedError if not trained

# Option 2: Factory method (recommended)
audit = BankAuditSystem.create_trained_system(
    bank_name="ABC Bank",
    audit_period="2024-Q1",
    bank_data=df,
    training_data=all_banks_df
)
results = audit.perform_full_audit()  # Everything ready to go
```

### Credit Risk Model Example

```python
# Before: Unclear if model is ready
model = CreditRiskModel()
predictions = model.predict_risk(data)  # May fail if not trained

# After Option 1: Explicit lifecycle
model = CreditRiskModel()
model.train_models(training_data)
predictions = model.predict_risk(data)  # Safe

# After Option 2: Factory method (recommended)
model = CreditRiskModel.create_and_train(training_data)
predictions = model.predict_risk(data)  # Guaranteed ready
```

## Benefits

### 1. Clear Error Messages
**Before:**
```
AttributeError: 'CreditRiskModel' object has no attribute 'feature_names'
```

**After:**
```
ModelNotTrainedError: Credit Risk Model must be trained before calling predict_risk
Context: {'model': 'Credit Risk Model', 'method': 'predict_risk', 'state': 'uninitialized'}
```

### 2. Self-Documenting Code
```python
# Lifecycle is now explicit in docstrings
class CreditRiskModel:
    \"\"\"
    LIFECYCLE:
        1. Create: model = CreditRiskModel()
        2. Train: model.train_models(training_data)
        3. Use: model.predict_risk(test_data)
    \"\"\"
```

### 3. Fail-Fast Behavior
```python
# Catches errors immediately instead of deep in call stack
model = CreditRiskModel()
try:
    model.predict_risk(data)  # Fails immediately
except ModelNotTrainedError as e:
    print(f"Error: {e}")
    print(f"Solution: Call model.train_models() first")
```

### 4. Factory Pattern Convenience
```python
# Create fully-ready systems in one line
audit = BankAuditSystem.create_trained_system(
    bank_name="ABC Bank",
    audit_period="2024-Q1",
    bank_data=current_data,
    training_data=historical_data
)
# Ready to use immediately - no lifecycle management needed
```

## Testing

Created comprehensive tests in `test_state_management.py`:
- ✓ Validates lifecycle enforcement
- ✓ Tests factory methods
- ✓ Verifies state transitions
- ✓ Checks error messages

## Migration Guide

### For Existing Code

**Old Pattern:**
```python
system = BankAuditSystem("Bank", "2024-Q1")
system.load_and_prepare_data(df, "BANK_ID")
results = system.assess_risk(prepared_data)
```

**New Pattern (Recommended):**
```python
# Use factory method
system = BankAuditSystem.create_trained_system(
    bank_name="Bank",
    audit_period="2024-Q1",
    bank_data=df,
    training_data=training_df
)
results = system.perform_full_audit()
```

**New Pattern (Manual):**
```python
# If you need more control
system = BankAuditSystem("Bank", "2024-Q1")
system.train_models(training_data)  # Now required
system.load_and_prepare_data(df, "BANK_ID")
results = system.assess_risk(prepared_data)  # Validated
```

## Conclusion

These improvements ensure:
- ✅ Objects are always in a valid state
- ✅ Clear lifecycle documentation
- ✅ Fail-fast error detection
- ✅ Factory methods for convenience
- ✅ Better developer experience
- ✅ Production-ready code
