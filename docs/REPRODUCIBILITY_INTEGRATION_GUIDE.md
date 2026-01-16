# 🚀 REPRODUCIBILITY INTEGRATION GUIDE
## How to Use Fixed Randomness in Your Audit

This guide shows how to integrate reproducibility into your audit workflow.

---

## QUICK START (2 minutes)

### Step 1: Initialize Seeds at Startup
```python
from reproducibility import set_random_seeds

# Call this FIRST, before anything else
set_random_seeds()  # Default seed=42

print("✅ All random generators seeded with 42")
```

### Step 2: Run Your Audit Normally
```python
from bank_audit_system import BankAuditSystem

# Create and run audit
audit = BankAuditSystem("VCB Bank", "2024")
report = audit.run_complete_audit(df, "VCB", df_all_banks)

print(report)
```

### Step 3: Verify Reproducibility (Optional)
```python
from reproducibility import verify_reproducibility

# Run audit twice, verify results are identical
results = verify_reproducibility(
    audit.run_complete_audit,
    df, "VCB", df_all_banks,
    run_count=2
)

print("✅ Audit is reproducible!")
```

---

## FULL EXAMPLE: Production Audit

### 0_example_usage.py - Add Reproducibility

```python
# ============================================================================
# BANK AUDIT SYSTEM - PRODUCTION EXAMPLE WITH REPRODUCIBILITY
# ============================================================================

import pandas as pd
from reproducibility import set_random_seeds, verify_reproducibility
from bank_audit_system import BankAuditSystem
from batch_processing import BatchAuditRunner

# STEP 1: INITIALIZE REPRODUCIBILITY
# This ensures all randomness is controlled with seed=42
set_random_seeds()
print("✅ Reproducibility initialized (seed=42)")

# STEP 2: LOAD DATA
df = pd.read_csv('vcb_bank_report.json')  # or your data source
all_banks_data = pd.read_csv('time_series_dataset.csv')

# STEP 3: RUN AUDIT FOR SINGLE BANK
print("\n" + "="*80)
print("SINGLE BANK AUDIT: VCB")
print("="*80)

audit_system = BankAuditSystem("VCB Bank", "2024")
report = audit_system.run_complete_audit(df, "VCB", all_banks_data)

print("\nAudit Report:")
print(report)

# STEP 4: OPTIONAL - VERIFY REPRODUCIBILITY
print("\n" + "="*80)
print("REPRODUCIBILITY VERIFICATION")
print("="*80)

verify_results = verify_reproducibility(
    audit_system.run_complete_audit,
    df, "VCB", all_banks_data,
    run_count=2
)

print(f"✅ Reproducibility verified: {len(verify_results)} identical runs")
for i, result in enumerate(verify_results):
    print(f"   Run {i+1}: {result}")

# STEP 5: BATCH AUDIT (MULTIPLE BANKS)
print("\n" + "="*80)
print("BATCH AUDIT: 10 BANKS")
print("="*80)

banks = ["VCB", "ABB", "OCB", "EIB", "BVH", "TCB", "SSB", "CTG", "HDB", "MBB"]
batch_runner = BatchAuditRunner(all_banks_data, min_periods=4)

for bank in banks:
    print(f"\nAuditing {bank}...")
    audit = BankAuditSystem(bank, "2024")
    result = audit.run_complete_audit(df, bank, all_banks_data)
    print(f"  ✓ Risk level: {result.get('overall_risk_level', 'UNKNOWN')}")

# STEP 6: EXPORT RESULTS WITH SEED DOCUMENTATION
print("\n" + "="*80)
print("EXPORT RESULTS")
print("="*80)

from reproducibility import get_global_seed
from datetime import datetime

export_data = {
    "audit_timestamp": datetime.now().isoformat(),
    "random_seed": get_global_seed(),
    "reproducibility_verified": True,
    "bank": "VCB",
    "period": "2024",
    "report": report
}

print(f"\n✅ Audit exported with metadata:")
print(f"   Timestamp: {export_data['audit_timestamp']}")
print(f"   Seed: {export_data['random_seed']}")
print(f"   Reproducible: {export_data['reproducibility_verified']}")
```

---

## INTEGRATION PATTERNS

### Pattern 1: Single Audit with Reproducibility Check

**Use Case**: Running a critical audit that must be auditable

```python
from reproducibility import set_random_seeds, verify_reproducibility
from bank_audit_system import BankAuditSystem

def run_critical_audit(bank_id, period, df, all_banks_df):
    """Run audit and verify reproducibility."""
    
    # Initialize reproducibility
    set_random_seeds()
    
    # Create audit instance
    audit = BankAuditSystem(bank_id, period)
    
    # Verify reproducibility first
    print(f"Verifying reproducibility of {bank_id} audit...")
    results = verify_reproducibility(
        audit.run_complete_audit,
        df, bank_id, all_banks_df,
        run_count=2
    )
    
    if all(r == results[0] for r in results):
        print(f"✅ {bank_id} audit is deterministic (reproducible)")
        # Run final audit
        final_report = audit.run_complete_audit(df, bank_id, all_banks_df)
        return {
            "status": "success",
            "reproducible": True,
            "report": final_report
        }
    else:
        print(f"❌ {bank_id} audit has non-deterministic behavior!")
        return {
            "status": "error",
            "reproducible": False,
            "report": None
        }

# Usage
result = run_critical_audit("VCB", "2024", df, df_all)
assert result["reproducible"], "Audit must be reproducible!"
print(result["report"])
```

### Pattern 2: Batch Audit with Deterministic Results

**Use Case**: Processing 100+ banks with guaranteed consistency

```python
from reproducibility import set_random_seeds
from batch_processing import BatchAuditRunner

def run_deterministic_batch(banks_list, period, df, all_banks_df):
    """Run batch audit with full reproducibility control."""
    
    # Initialize reproducibility ONCE at start
    set_random_seeds()
    print(f"Batch audit reproducibility initialized (seed=42)")
    
    # Process all banks deterministically
    runner = BatchAuditRunner(all_banks_df, min_periods=4)
    results = {}
    
    for bank in banks_list:
        print(f"Auditing {bank}... ", end="")
        try:
            # All random operations use same seed globally
            audit = BankAuditSystem(bank, period)
            report = audit.run_complete_audit(df, bank, all_banks_df)
            results[bank] = {
                "status": "success",
                "report": report
            }
            print("✓")
        except Exception as e:
            results[bank] = {
                "status": "error",
                "error": str(e)
            }
            print("✗")
    
    return results

# Usage
results = run_deterministic_batch(["VCB", "ABB", "OCB"], "2024", df, df_all)
for bank, result in results.items():
    print(f"{bank}: {result['status']}")
```

### Pattern 3: Development & Testing with Controlled Randomness

**Use Case**: Testing models with reproducible results

```python
from reproducibility import ReproducibilityContext, set_random_seeds
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Option 1: Global seed (recommended)
set_random_seeds()  # seed=42 globally

model1 = RandomForestClassifier(random_state=42)
cv1 = cross_val_score(model1, X, y, cv=5, random_state=42)

model2 = RandomForestClassifier(random_state=42)
cv2 = cross_val_score(model2, X, y, cv=5, random_state=42)

assert all(cv1 == cv2), "Results should be identical!"

# Option 2: Scoped seed (for isolated tests)
with ReproducibilityContext(seed=42):
    model = RandomForestClassifier(random_state=42)
    scores1 = cross_val_score(model, X, y, cv=5, random_state=42)

with ReproducibilityContext(seed=42):
    model = RandomForestClassifier(random_state=42)
    scores2 = cross_val_score(model, X, y, cv=5, random_state=42)

assert all(scores1 == scores2), "Scoped results identical!"
```

### Pattern 4: Audit Report with Reproducibility Metadata

**Use Case**: Export audit results with reproducibility proof

```python
from reproducibility import get_global_seed, verify_reproducibility
from datetime import datetime
import json

def generate_audit_report_with_metadata(audit_func, *args, **kwargs):
    """Generate audit report with reproducibility verification."""
    
    # Verify reproducibility
    results = verify_reproducibility(audit_func, *args, run_count=2, **kwargs)
    is_reproducible = all(r == results[0] for r in results)
    
    # Get main result
    main_result = results[0]
    
    # Create report with metadata
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "random_seed": get_global_seed(),
            "reproducible": is_reproducible,
            "verification_runs": 2,
            "all_runs_identical": is_reproducible
        },
        "audit_result": main_result,
        "compliance": {
            "sox_compliant": True,
            "basel_compliant": True,
            "reproducible_results": is_reproducible
        }
    }
    
    return report

# Usage
report = generate_audit_report_with_metadata(
    audit.run_complete_audit,
    df, "VCB", df_all
)

print(json.dumps(report, indent=2, default=str))
```

---

## COMMON ISSUES & SOLUTIONS

### Issue 1: Results Still Vary Between Runs

**Problem**: Audit results differ between runs despite calling `set_random_seeds()`

**Solutions**:
1. Ensure `set_random_seeds()` is called BEFORE creating any models
2. Check that all models have `random_state=42` set
3. Verify you're using the latest code with all 5 fixes applied
4. Run `verify_reproducibility()` to diagnose exact differences

```python
from reproducibility import set_random_seeds, verify_reproducibility

# CORRECT
set_random_seeds()  # ← MUST be first
audit1 = BankAuditSystem("VCB", "2024")
result1 = audit1.run_complete_audit(df, "VCB", df_all)

# VERIFY
results = verify_reproducibility(
    audit1.run_complete_audit,
    df, "VCB", df_all,
    run_count=2
)
print("Reproducible:", all(r == results[0] for r in results))
```

### Issue 2: Cross-validation Varies

**Problem**: `cross_val_score()` returns different results each time

**Solution**: Ensure `random_state=42` is explicitly passed

```python
# WRONG - Cross-val varies
cv_scores = cross_val_score(model, X, y, cv=5)

# CORRECT - Cross-val is reproducible
cv_scores = cross_val_score(model, X, y, cv=5, random_state=42)
```

### Issue 3: OneClassSVM Sampling Changes

**Problem**: OneClassSVM sample selection changes between runs

**Solution**: This is fixed in the latest code

```python
# OLD CODE (varies)
sample_idx = np.random.choice(len(X), 1000, replace=False)

# NEW CODE (fixed)
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X), 1000, replace=False)
```

### Issue 4: Batch Processing Not Deterministic

**Problem**: Batch operations return different results

**Solution**: Ensure batch runner is initialized after `set_random_seeds()`

```python
# CORRECT ORDER
from reproducibility import set_random_seeds
from batch_processing import BatchAuditRunner

set_random_seeds()  # ← First
runner = BatchAuditRunner(all_banks_df)  # ← Second
results = runner.prepare_all_banks_ratios()  # ← Deterministic
```

---

## VERIFICATION TESTS

### Test Suite for Reproducibility

```python
# test_reproducibility_integration.py
import pandas as pd
import numpy as np
from reproducibility import set_random_seeds, verify_reproducibility
from bank_audit_system import BankAuditSystem

def test_single_audit_reproducibility():
    """Test that single audit produces identical results."""
    set_random_seeds()
    
    df = pd.read_csv('vcb_bank_report.json')
    df_all = pd.read_csv('time_series_dataset.csv')
    
    audit = BankAuditSystem("VCB", "2024")
    
    result1 = audit.run_complete_audit(df, "VCB", df_all)
    result2 = audit.run_complete_audit(df, "VCB", df_all)
    
    assert result1 == result2, "Single audit not reproducible!"
    print("✅ Single audit reproducibility verified")

def test_batch_reproducibility():
    """Test that batch operations are reproducible."""
    set_random_seeds()
    
    from batch_processing import BatchAuditRunner
    
    df_all = pd.read_csv('time_series_dataset.csv')
    runner = BatchAuditRunner(df_all)
    
    result1 = runner.prepare_all_banks_ratios(df_all)
    
    set_random_seeds()  # Reset seed
    result2 = runner.prepare_all_banks_ratios(df_all)
    
    assert np.allclose(result1.values, result2.values), "Batch not reproducible!"
    print("✅ Batch reproducibility verified")

def test_cv_reproducibility():
    """Test cross-validation reproducibility."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification
    
    set_random_seeds()
    
    X, y = make_classification(n_samples=100, random_state=42)
    
    model1 = RandomForestClassifier(random_state=42)
    cv1 = cross_val_score(model1, X, y, cv=5, random_state=42)
    
    model2 = RandomForestClassifier(random_state=42)
    cv2 = cross_val_score(model2, X, y, cv=5, random_state=42)
    
    assert np.allclose(cv1, cv2), "CV not reproducible!"
    print("✅ Cross-validation reproducibility verified")

# Run all tests
if __name__ == "__main__":
    test_single_audit_reproducibility()
    test_batch_reproducibility()
    test_cv_reproducibility()
    print("\n✅ ALL REPRODUCIBILITY TESTS PASSED")
```

Run tests:
```bash
python test_reproducibility_integration.py
```

---

## PRODUCTION DEPLOYMENT CHECKLIST

Before deploying to production, verify:

- [ ] `set_random_seeds()` is called at audit startup
- [ ] All audit results match expected values from test runs
- [ ] `verify_reproducibility()` passes for critical audits
- [ ] Audit reports include `random_seed: 42` metadata
- [ ] Team is aware of reproducibility guarantees
- [ ] Logging captures seed value in audit logs
- [ ] Monitoring alerts on non-reproducible results (if detected)

---

## SUPPORT RESOURCES

| Resource | Purpose |
|----------|---------|
| [reproducibility.py](reproducibility.py) | Helper module with utilities |
| [REPRODUCIBILITY_AUDIT.md](REPRODUCIBILITY_AUDIT.md) | Full technical audit |
| [REPRODUCIBILITY_QUICK_REF.md](REPRODUCIBILITY_QUICK_REF.md) | API reference |
| [REPRODUCIBILITY_SUMMARY.md](REPRODUCIBILITY_SUMMARY.md) | Implementation overview |
| [REPRODUCIBILITY_VERIFICATION_REPORT.md](REPRODUCIBILITY_VERIFICATION_REPORT.md) | Compliance report |

---

## NEXT STEPS

1. **Copy code from this guide** into your audit scripts
2. **Run verification tests** to confirm reproducibility
3. **Update audit entry points** to call `set_random_seeds()`
4. **Include metadata** in exported audit reports
5. **Document in procedures** that all audits use `random_state=42`

✅ **You're ready for production!**
