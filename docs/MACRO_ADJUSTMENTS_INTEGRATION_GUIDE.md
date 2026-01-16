# MACRO-ADJUSTMENTS INTEGRATION GUIDE

## Quick Start

### Step 1: Enable Macro-Adjustments in Audit System

```python
from 5_bank_audit_system import BankAuditSystem
from macro_adjustments import MacroAdjustmentCalculator

# Initialize audit system
audit_system = BankAuditSystem()

# Load all banks data for benchmarking
import pandas as pd
all_banks = pd.read_csv('time_series_dataset_enriched.csv')

# Run audit with macro-adjustments
audit_report = audit_system.audit_bank(
    bank_data=single_bank_data,
    apply_macro_adjustments=True,
    all_banks_for_context=all_banks
)

# Report now includes macro-adjustment context
print(f"Absolute Credit Risk: {audit_report['credit_risk']['score']:.1f}")
print(f"Adjusted Credit Risk: {audit_report['macro_adjustments']['adjusted_indicators']['credit_risk']['adjusted_score']:.1f}")
print(f"Industry Context: {audit_report['macro_adjustments']['systemic_stress_level']:.1%} stress")
```

### Step 2: Understand the Adjustment

```
BEFORE (Absolute Only):
  NPL Ratio: 4%
  Credit Risk Score: 62 (high risk)
  
AFTER (With Macro-Adjustment):
  NPL Ratio: 4%
  Industry Average NPL: 3%
  Relative Position: +0.5 standard deviations (slightly above average)
  Systemic Stress: 25% of banks in distress
  Adjusted Credit Risk Score: 58 (moderate risk due to context)
```

---

## Key Benefits in Different Scenarios

### Scenario 1: Normal Economic Conditions

**Problem Without Adjustments:**
- Bank A: 3.5% NPL → Score 55
- Bank B: 3.5% NPL → Score 55 (same score)

**Solution With Adjustments:**
- Industry average NPL: 2.0%
- Bank A: NPL 3.5%, better credit portfolio → Adjusted score 50 (distinguishes quality)
- Bank B: NPL 3.5%, worse credit portfolio → Adjusted score 60 (proper differentiation)

### Scenario 2: Severe Recession

**Problem Without Adjustments:**
- 85% of banks have NPL > 5%
- All trigger same absolute threshold
- Regulators can't distinguish truly troubled from average performers
- May lead to aggressive intervention everywhere or nowhere

**Solution With Adjustments:**
- Systemic stress estimated at 85%
- All scores adjusted downward (not flagged as high risk when industry-wide)
- Banks performing better than peers still flagged as concerning
- Enables targeted intervention: "Focus on bottom 25% performers, not entire sector"

### Scenario 3: Recovery Phase

**Problem Without Adjustments:**
- Absolute metrics haven't fully recovered to pre-crisis levels
- Banks still flagged as risky even though improving relative to peers
- Discourages positive behavior

**Solution With Adjustments:**
- Industry is recovering, benchmarks adjust upward
- Banks improving faster than average get lower adjusted scores
- Positive reinforcement for risk reduction

---

## Integration Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│              Bank Audit System (Main)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Credit Risk     │         │  Liquidity Risk  │         │
│  │   Model          │         │   Model          │         │
│  │  ┌────────────┐  │         │  ┌────────────┐  │         │
│  │  │ Absolute   │  │         │  │ Absolute   │  │         │
│  │  │ Score      │  │         │  │ Score      │  │         │
│  │  └────────────┘  │         │  └────────────┘  │         │
│  │        │         │         │        │         │         │
│  │        └─────────┼─────────┴────────┘         │         │
│  │                │                             │         │
│  └────────────────┼─────────────────────────────┘         │
│                   │                                        │
│                   ▼                                        │
│  ┌──────────────────────────────────────────────┐         │
│  │  Macro-Adjustment Calculator                │         │
│  │  ┌──────────────────────────────────────┐   │         │
│  │  │ 1. Calculate Industry Benchmarks     │   │         │
│  │  │ 2. Estimate Systemic Stress         │   │         │
│  │  │ 3. Adjust Risk Scores               │   │         │
│  │  │ 4. Generate Explanations            │   │         │
│  │  └──────────────────────────────────────┘   │         │
│  └──────────────────────────────────────────────┘         │
│                   │                                        │
│                   ▼                                        │
│  ┌──────────────────────────────────────────────┐         │
│  │  Audit Report with Context                   │         │
│  │  ├─ Absolute scores                         │         │
│  │  ├─ Adjusted scores                         │         │
│  │  ├─ Z-scores (relative position)            │         │
│  │  ├─ Systemic stress level                   │         │
│  │  └─ Peer benchmarks                         │         │
│  └──────────────────────────────────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Data (Single Bank)
        │
        ▼
┌──────────────────────────┐
│ Train/Predict Models    │
│ (Credit, Liquidity, etc)│
└──────────────────────────┘
        │
        ▼
    Absolute Scores
    (Credit: 62, Liquidity: 45, etc.)
        │
        ▼
┌──────────────────────────────────────────────┐
│ Macro-Adjustment Module                      │
│                                              │
│ Input: All Banks Data (for benchmarking)     │
│ ├─ Calculate industry benchmarks            │
│ ├─ Estimate systemic stress                 │
│ └─ Adjust all scores relative to benchmarks│
│                                              │
└──────────────────────────────────────────────┘
        │
        ▼
    Adjusted Scores
    (Credit: 58, Liquidity: 41, etc.)
        │
        ▼
┌──────────────────────────────────────────────┐
│ Enriched Audit Report                        │
│ ├─ Absolute risk levels                     │
│ ├─ Adjusted risk levels                     │
│ ├─ Industry benchmarks                      │
│ ├─ Systemic stress assessment               │
│ └─ Comparative analysis                     │
└──────────────────────────────────────────────┘
```

---

## Measurement and Validation

### Historical Validation

Compare predictive power of adjusted vs. absolute scores:

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Collect historical data with known outcomes
historical_audits = [...]  # List of past audits with outcomes
actual_defaults = [...]    # Did bank default? (True/False)

absolute_scores = [a['credit_risk']['score'] for a in historical_audits]
adjusted_scores = [a['macro_adjustments']['adjusted_indicators']['credit_risk']['adjusted_score'] 
                   for a in historical_audits]

# Calculate ROC AUC
auc_absolute = roc_auc_score(actual_defaults, absolute_scores)
auc_adjusted = roc_auc_score(actual_defaults, adjusted_scores)

print(f"AUC (Absolute):  {auc_absolute:.3f}")
print(f"AUC (Adjusted):  {auc_adjusted:.3f}")
print(f"Improvement:     {(auc_adjusted - auc_absolute)*100:.1f} bps")

# Plot ROC curves
fpr_abs, tpr_abs, _ = roc_curve(actual_defaults, absolute_scores)
fpr_adj, tpr_adj, _ = roc_curve(actual_defaults, adjusted_scores)

plt.plot(fpr_abs, tpr_abs, label=f'Absolute (AUC={auc_absolute:.3f})')
plt.plot(fpr_adj, tpr_adj, label=f'Adjusted (AUC={auc_adjusted:.3f})')
plt.legend()
plt.savefig('roc_comparison.png')
```

### Stress Test Scenarios

Validate adjustments across different economic scenarios:

```python
from example_macro_adjustments import create_sample_industry_data

# Test scenarios
scenarios = {
    'normal': create_sample_industry_data('normal'),
    'stressed': create_sample_industry_data('stressed'),
    'severe_crisis': create_sample_industry_data('severe')  # If available
}

# For each scenario, audit same bank
test_bank = create_test_bank()

results = {}
for scenario_name, banks_data in scenarios.items():
    audit = audit_system.audit_bank(
        bank_data=test_bank,
        apply_macro_adjustments=True,
        all_banks_for_context=banks_data
    )
    
    results[scenario_name] = {
        'absolute_score': audit['credit_risk']['score'],
        'adjusted_score': audit['macro_adjustments']['adjusted_indicators']['credit_risk']['adjusted_score'],
        'stress_level': audit['macro_adjustments']['systemic_stress_level']
    }

# Verify that adjusted scores are more stable across scenarios
print("Scenario Comparison:")
print(f"{'Scenario':<20} {'Absolute':<15} {'Adjusted':<15} {'Stress':<15}")
for scenario, result in results.items():
    print(f"{scenario:<20} {result['absolute_score']:<14.1f} {result['adjusted_score']:<14.1f} {result['stress_level']:<14.1%}")
```

---

## Configuration and Tuning

### Adjustment Strength

Control how much industry context affects the final score:

```python
# Conservative: Trust absolute thresholds
calc = MacroAdjustmentCalculator()
adj_conservative = calc.adjust_risk_score(
    absolute_score=65,
    relative_z_score=1.5,
    adjustment_strength=0.2  # Weak context weighting
)

# Balanced: Mix absolute and relative
adj_balanced = calc.adjust_risk_score(
    absolute_score=65,
    relative_z_score=1.5,
    adjustment_strength=0.35  # Moderate context weighting
)

# Context-focused: Trust peer comparison more
adj_aggressive = calc.adjust_risk_score(
    absolute_score=65,
    relative_z_score=1.5,
    adjustment_strength=0.5  # Strong context weighting
)
```

### Stress Thresholds

Customize what counts as "bank in distress":

```python
# Basel III regulatory perspective
basel_thresholds = {
    'capital_adequacy_ratio': ('<', 0.08),
    'npl_ratio': ('>', 0.05),
    'liquidity_coverage_ratio': ('<', 1.0)
}

# Conservative supervisory view
conservative_thresholds = {
    'capital_adequacy_ratio': ('<', 0.10),
    'npl_ratio': ('>', 0.03),
    'liquidity_coverage_ratio': ('<', 1.25)
}

stress_basel = calc.estimate_systemic_stress(
    all_banks, 
    list(basel_thresholds.keys()),
    basel_thresholds
)

stress_conservative = calc.estimate_systemic_stress(
    all_banks,
    list(conservative_thresholds.keys()),
    conservative_thresholds
)
```

---

## Reporting and Communication

### Standard Report Format

```
BANK AUDIT REPORT: BANK_ABC_123
Date: 2024-Q4
────────────────────────────────────────────────────────────

RISK ASSESSMENT SUMMARY
────────────────────────────────────────────────────────────

CREDIT RISK
  Absolute Score:       62.3 (MODERATE)
  Adjusted Score:       58.1 (MODERATE)
  Adjustment:           -4.2 (due to positive industry context)
  Relative Position:    +0.8σ (slightly above industry avg)
  Industry Average:     58.5%
  Percentile Rank:      65th (in worse half of peers)
  Stress Level:         25% of banks in distress

Interpretation:
  Bank's credit risk metrics are slightly worse than industry average,
  but improving. In the current economic environment (25% systemic stress),
  the bank ranks in the 65th percentile. Recommended monitoring level: NORMAL.

LIQUIDITY RISK
  Absolute Score:       45.2 (LOW)
  Adjusted Score:       42.8 (LOW)
  Adjustment:           -2.4 (due to positive industry context)
  Relative Position:    -0.5σ (better than industry avg)
  Industry Average:     47.1%
  Percentile Rank:      35th (in better half of peers)
  Stress Level:         8% of banks in distress

Interpretation:
  Bank's liquidity position is stronger than industry average. Bank is
  in the 35th percentile (better side). Liquidity risk is low with no
  immediate concern. Recommended monitoring level: NORMAL.

────────────────────────────────────────────────────────────
MACRO-ECONOMIC CONTEXT
────────────────────────────────────────────────────────────
Overall Systemic Stress:  25.0%
Economy Status:           MODERATE STRESS
Number of Banks Assessed: 47
Banks in Distress:        12 (25%)

Interpretation:
  Current economic environment shows moderate stress. One quarter of banks
  are experiencing financial difficulty. This is reflected in adjusted
  scores through reduced penalization for industry-wide challenges.

────────────────────────────────────────────────────────────
RECOMMENDATIONS
────────────────────────────────────────────────────────────
1. Credit Risk: Monitor quarterly; focus on portfolio quality improvement
2. Liquidity Risk: No immediate action required; maintain current position
3. Overall Risk: ACCEPTABLE with NORMAL monitoring
4. Stress Scenario: In severe recession (stress > 60%), credit risk could
   rise to adjusted score of ~72. Recommend liquidity contingency planning.

────────────────────────────────────────────────────────────
Generated: 2024-01-09
System: Bank Audit System v2.1 (with Macro-Adjustments)
```

---

## Best Practices

### 1. Always Show Both Scores
```python
# ✓ Correct: Show absolute and adjusted for transparency
report = {
    'credit_risk_absolute': 62,
    'credit_risk_adjusted': 58,
    'macro_context': {
        'industry_average': 58.5,
        'systemic_stress': 0.25,
        'adjustment_reason': '...'
    }
}

# ✗ Avoid: Hiding absolute score
report = {
    'credit_risk': 58,  # Only adjusted, no transparency
}
```

### 2. Document Assumptions
```python
# When reporting, include:
# - Time period for benchmarking (e.g., "Industry data from 2024-Q3")
# - Sample size (e.g., "Benchmarked against 47 banks")
# - Adjustment methodology (e.g., "Z-score with 0.35 strength weighting")
# - Systemic stress definition (e.g., "% of banks with NPL > 5%")
```

### 3. Validate Regularly
```python
# Monthly validation
def validate_adjustments():
    """Check that adjustments are reasonable"""
    
    # Validation checks
    checks = {
        'z_scores_within_bounds': all(abs(z) < 5 for z in z_scores),
        'stress_level_in_range': all(0 <= s <= 1 for s in stress_levels),
        'adjustments_move_toward_average': all(
            (score > avg and adj < score) or (score < avg and adj > score)
            for score, adj, avg in zip(abs_scores, adj_scores, avg_scores)
        ),
        'confidence_reasonable': all(0 <= c <= 1 for c in confidences)
    }
    
    failed = [k for k, v in checks.items() if not v]
    if failed:
        logger.warning(f"Validation failed: {failed}")
        return False
    
    return True
```

### 4. Update Benchmarks Regularly
```python
# Recalculate industry benchmarks quarterly or after significant events

def update_benchmarks():
    """Update industry benchmarks with latest data"""
    
    latest_data = pd.read_csv('time_series_dataset_enriched.csv')
    
    # Keep only recent period (e.g., last 4 quarters for rolling window)
    recent_data = latest_data[latest_data['period'] >= '2024-Q1']
    
    benchmarks = calc.calculate_industry_benchmarks(
        recent_data,
        ['npl_ratio', 'liquidity_ratio', 'capital_adequacy_ratio']
    )
    
    # Save for future use
    import json
    with open('current_benchmarks.json', 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    logger.info(f"Benchmarks updated with {len(recent_data)} observations")
```

---

## Summary

**Macro-adjustments** transform the audit system from a one-dimensional (absolute threshold) approach to a multi-dimensional (context-aware) approach:

| Aspect | Absolute Only | With Macro-Adjustments |
|--------|---------------|------------------------|
| **Threshold** | Fixed (e.g., NPL ≤ 3%) | Dynamic (adjusted per cycle) |
| **Context** | Ignores peer performance | Includes industry benchmarks |
| **Systemic risk** | Treats all downturns equally | Differentiates systemic vs. idiosyncratic |
| **Economic cycle** | Same alerts in boom and bust | Calibrated to economic conditions |
| **False positives** | High during crises | Significantly reduced |
| **False negatives** | May miss early outliers | Early detection via z-scores |
| **Transparency** | Simple but opaque | Complex but explainable |

The result: **More accurate, context-aware risk assessment** that guides regulatory action appropriately across economic cycles.

---

**Next Steps:**
1. Run `python example_macro_adjustments.py` to understand the mechanics
2. Review [MACRO_ADJUSTMENTS.md](MACRO_ADJUSTMENTS.md) for technical details
3. Integrate `MacroAdjustmentCalculator` into your audit pipeline
4. Validate adjustments against historical defaults
5. Update benchmarks quarterly as new data arrives
