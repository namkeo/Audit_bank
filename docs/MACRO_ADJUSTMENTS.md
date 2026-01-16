# Macro-Adjustments Module
## Contextualizing Risk Scores Against Industry Benchmarks

### Overview

The **Macro-Adjustments Module** extends the bank audit system with industry-context awareness. Instead of using absolute thresholds alone, it contextualizes individual bank risk scores against industry benchmarks to distinguish **systemic risk** (industry-wide stress) from **idiosyncratic risk** (bank-specific problems).

---

## Problem Statement

### Without Macro-Adjustments
- **NPL Ratio = 4%** with industry avg = **1%** → Same risk score as
- **NPL Ratio = 4%** with industry avg = **3%**

Both fail the same absolute threshold (e.g., 3%), but the **relative context** differs dramatically:
- First scenario: Bank is 3% above average (high relative risk)
- Second scenario: Bank is 1% above average (moderate relative risk)

### With Macro-Adjustments
Risk scores are contextualized:
- **4% NPL, industry avg 1%** → **High risk** (top 25% worst performers)
- **4% NPL, industry avg 3%** → **Moderate risk** (only slightly above average)
- **3.5% NPL in severe recession** → **Acceptable risk** (most banks stressed, bank is median)

---

## Key Concepts

### 1. Absolute Risk
**Definition**: Bank's metric vs. fixed regulatory threshold
```
Credit Risk Threshold: NPL ≤ 3%
Bank's NPL: 4%
→ Status: FAILS threshold (absolute score increases)
```

### 2. Relative Risk
**Definition**: Bank's metric vs. industry average
```
Bank's NPL: 4%
Industry Average: 3%
Deviation: +1% (+0.5 standard deviations)
→ Bank slightly worse than average, but acceptable
```

### 3. Systemic Risk
**Definition**: Industry-wide stress affecting most banks
```
Scenario: Recession
- 80% of banks' NPL ratios exceed 3%
- Systemic stress level: 80%
→ High stress shouldn't flag individual banks at threshold
```

### 4. Z-Score (Standard Deviation Units)
```
Z-Score = (Bank Value - Industry Mean) / Industry Std Dev

Example:
- Bank NPL: 4%
- Industry Mean: 3%
- Industry Std Dev: 1%
- Z-Score: (4% - 3%) / 1% = +1.0 (1 std dev above average)

Interpretation:
- Z < -2: Best ~2.5% of peers (good)
- -1 < Z < +1: Within 1 std dev of average (normal)
- +1 < Z < +2: 1-2 std devs above average (concerning)
- Z > +2: Worst ~2.5% of peers (problematic)
```

---

## Module Components

### MacroAdjustmentCalculator

Main class providing adjustment calculations.

#### Key Methods

##### 1. `calculate_industry_benchmarks()`
Compute statistics for all banks in a period.

```python
benchmarks = calc.calculate_industry_benchmarks(
    all_banks_data=df,
    metrics=['npl_ratio', 'liquidity_ratio', 'capital_adequacy_ratio'],
    period='2024-Q4'  # Optional
)

# Returns:
# {
#   'npl_ratio': {
#     'mean': 0.028,
#     'std': 0.015,
#     'p25': 0.018,
#     'p50': 0.025,
#     'p75': 0.035,
#     'min': 0.005,
#     'max': 0.125
#   },
#   ...
# }
```

##### 2. `calculate_relative_deviation()`
Calculate z-score (standardized deviation from mean).

```python
z_score = calc.calculate_relative_deviation(
    bank_value=0.04,
    industry_mean=0.03,
    industry_std=0.01
)
# Returns: 1.0 (1 standard deviation above average)
```

##### 3. `adjust_risk_score()`
Apply macro-adjustment to a single risk score.

```python
adjustment = calc.adjust_risk_score(
    absolute_score=65,           # Original credit risk score
    relative_z_score=1.5,        # Deviation from industry avg
    systemic_stress_level=0.4,   # 40% of banks in distress
    adjustment_strength=0.35     # How much to weight context
)

# Returns:
# {
#   'adjusted_score': 59.2,
#   'adjustment_delta': -5.8,
#   'adjustment_reason': '...',
#   'adjustment_confidence': 0.85
# }
```

##### 4. `estimate_systemic_stress()`
Calculate industry-wide stress level.

```python
stress = calc.estimate_systemic_stress(
    all_banks_data=df,
    stress_metrics=['npl_ratio', 'liquidity_ratio'],
    thresholds={
        'npl_ratio': ('>', 0.05),
        'liquidity_ratio': ('<', 0.3)
    }
)
# Returns: 0.65 (65% of banks in distress → high stress)
```

---

## Integration with Risk Models

### Credit Risk Model Integration

```python
from 2_model_credit_risk import CreditRiskModel

# Train model
model = CreditRiskModel.create_and_train(training_data)

# Get base credit risk score
result = model.predict_risk(test_data)
credit_risk_score = result['credit_risk_score']  # e.g., 62

# Apply macro-adjustment
all_banks = pd.read_csv('time_series_dataset_enriched.csv')
adjustment = model.apply_macro_adjustments(
    credit_risk_score=credit_risk_score,
    bank_data={
        'npl_ratio': test_data['npl_ratio'].iloc[0],
        'capital_adequacy_ratio': test_data['capital_adequacy_ratio'].iloc[0],
        'loan_to_deposit_ratio': test_data['loan_to_deposit_ratio'].iloc[0]
    },
    all_banks_data=all_banks
)

# Result includes:
# - adjusted_score: 55.3 (down from 62)
# - adjustment_delta: -6.7
# - systemic_stress_level: 0.35
# - relative_z_score: 0.8
# - adjustment_reason: 'Bank's metrics above average but not severe...'
```

### Liquidity Risk Model Integration

```python
from 2_model_liquidity_risk import LiquidityRiskModel

model = LiquidityRiskModel.create_and_train(training_data)
result = model.predict_risk(test_data)

adjustment = model.apply_macro_adjustments(
    liquidity_risk_score=result['liquidity_risk_score'],
    bank_data={
        'liquidity_ratio': test_data['liquidity_ratio'].iloc[0],
        'lcr': test_data['lcr'].iloc[0],
        'net_stable_funding': test_data['net_stable_funding'].iloc[0]
    },
    all_banks_data=all_banks
)
```

---

## Usage Examples

### Example 1: Scenario Comparison

```python
from macro_adjustments import MacroAdjustmentCalculator

calc = MacroAdjustmentCalculator()

# NORMAL TIMES: Industry NPL = 1%
normal_benchmarks = {'npl_ratio': {'mean': 0.01, 'std': 0.005}}
normal_stress = 0.05  # 5% of banks in distress

# RECESSION: Industry NPL = 5%
recession_benchmarks = {'npl_ratio': {'mean': 0.05, 'std': 0.02}}
recession_stress = 0.75  # 75% of banks in distress

# Bank has 4% NPL in both scenarios
bank_npl = 0.04

# Adjustment 1: Normal times
adj_normal = calc.adjust_risk_score(
    absolute_score=70,
    relative_z_score=(0.04 - 0.01) / 0.005,  # +6 std devs!
    systemic_stress_level=0.05,
    adjustment_strength=0.35
)
# Result: Adjusted score ≈ 63 (high risk despite same NPL)

# Adjustment 2: Recession
adj_recession = calc.adjust_risk_score(
    absolute_score=65,
    relative_z_score=(0.04 - 0.05) / 0.02,  # -0.5 std devs (better!)
    systemic_stress_level=0.75,
    adjustment_strength=0.35
)
# Result: Adjusted score ≈ 58 (moderate risk, systemic stress mitigates)
```

### Example 2: Multi-Indicator Adjustment

```python
# Bank's risk indicators and metrics
indicator_scores = {
    'credit_risk': 62,
    'liquidity_risk': 45,
    'operational_risk': 38
}

bank_values = {
    'npl_ratio': 0.038,
    'liquidity_ratio': 0.42,
    'capital_adequacy_ratio': 0.11
}

benchmarks = {
    'npl_ratio': {'mean': 0.025, 'std': 0.01},
    'liquidity_ratio': {'mean': 0.48, 'std': 0.08},
    'capital_adequacy_ratio': {'mean': 0.125, 'std': 0.02}
}

systemic_stress = 0.35  # Elevated stress

# Apply adjustments
adjusted = calc.adjust_multiple_indicators(
    indicator_scores,
    benchmarks,
    bank_values,
    systemic_stress
)

# Output: Adjusted indicators with explanations
for indicator, adj in adjusted.items():
    print(f"{indicator}:")
    print(f"  Original: {adj['absolute_score']:.1f}")
    print(f"  Adjusted: {adj['adjusted_score']:.1f}")
    print(f"  Delta: {adj['adjustment_delta']:+.1f}")
    print(f"  Z-Score: {adj['relative_z_score']:.2f}σ")
    print(f"  Reason: {adj['adjustment_reason']}")
```

### Example 3: Generate Audit Report with Macro Context

```python
from macro_adjustments import apply_macro_adjustments_to_bank_audit

# Existing audit report
audit_report = {
    'bank_id': 'BANK_001',
    'credit_risk': {'score': 58},
    'liquidity_risk': {'score': 42},
    'bank_data': {
        'npl_ratio': 0.035,
        'liquidity_ratio': 0.45,
        'capital_adequacy_ratio': 0.115
    }
}

# Apply macro-adjustments
enriched_report = apply_macro_adjustments_to_bank_audit(
    audit_report,
    all_banks_data,
    indicators_to_adjust=['credit_risk', 'liquidity_risk']
)

# Result includes:
# enriched_report['macro_adjustments'] = {
#   'systemic_stress_level': 0.28,
#   'adjusted_indicators': {
#     'credit_risk': {...},
#     'liquidity_risk': {...}
#   },
#   'benchmarks': {...}
# }
```

---

## Adjustment Logic Details

### Formula: Adjustment Delta

```
z_score_adjustment = z_score × 10 × adjustment_strength
stress_mitigation = systemic_stress_level × z_score_adjustment × 0.5
total_adjustment = z_score_adjustment - stress_mitigation

adjusted_score = absolute_score + total_adjustment
```

### Example Calculation

**Scenario**: Bank NPL = 4%, Industry avg = 3%, std = 1%, stress = 40%, adjustment_strength = 0.35

```
z_score = (4% - 3%) / 1% = +1.0
z_score_adjustment = 1.0 × 10 × 0.35 = +3.5
stress_mitigation = 0.40 × 3.5 × 0.5 = +0.7
total_adjustment = 3.5 - 0.7 = +2.8

If absolute_score = 60:
adjusted_score = 60 + 2.8 = 62.8
```

### Interpretation

| Z-Score | Adjustment | Interpretation |
|---------|-----------|-----------------|
| -2.0 | Large negative | Much better than average (potential data quality issue?) |
| -1.0 | Moderate negative | Better than average (good) |
| 0.0 | No adjustment | Equal to industry average (neutral) |
| +1.0 | Moderate positive | Worse than average (concerning) |
| +2.0 | Large positive | Much worse than average (problematic) |

---

## Regulatory and Risk Management Benefits

### 1. **Systemic Risk Differentiation**
- Distinguishes true bank-specific problems from industry-wide stress
- Prevents false positives in crisis periods
- Example: In 2008 financial crisis, most banks had high NPL ratios—macro-adjustments prevent overreaction

### 2. **Procyclical Bias Mitigation**
- Without adjustments: Same metrics always trigger same alerts
- With adjustments: Alerts calibrated to economic cycle
- Result: More appropriate regulatory response during boom/bust

### 3. **Economic Cycle Accounting**
- **Boom periods**: Macro-adjustments increase sensitivity (catch exceptions)
- **Bust periods**: Macro-adjustments increase tolerance (focus on relative strength)
- Balance between missed risks and false alarms

### 4. **Peer Comparison**
- Banks can see how they perform vs. peers, not just vs. absolute standard
- Incentivizes relative improvement even when absolute threshold is missed

### 5. **Confidence Scoring**
- Adjustments include confidence levels
- High confidence when: bank is far from average AND systemic stress is low
- Low confidence when: bank near average OR systemic stress high

---

## Configuration and Tuning

### Adjustment Strength Parameter

Controls how much industry context influences final score (0 = absolute only, 1 = relative heavily weighted).

```python
# Conservative: Trust absolute thresholds more
calc.adjust_risk_score(..., adjustment_strength=0.2)

# Moderate: Balance absolute and relative
calc.adjust_risk_score(..., adjustment_strength=0.35)

# Aggressive: Trust industry context more
calc.adjust_risk_score(..., adjustment_strength=0.5)
```

### Stress Level Thresholds

Customize what counts as "distress" in your jurisdiction:

```python
thresholds = {
    'npl_ratio': ('>', 0.05),              # > 5% NPL = distress
    'liquidity_ratio': ('<', 0.3),         # < 30% liquidity = distress
    'capital_adequacy_ratio': ('<', 0.08), # < 8% CAR = distress
    'deposit_growth': ('<', -0.10)         # > 10% deposit decline = distress
}

stress = calc.estimate_systemic_stress(
    all_banks_data,
    list(thresholds.keys()),
    thresholds
)
```

---

## Best Practices

### 1. Always Calculate Benchmarks First
```python
# ✓ Correct: Calculate with sufficient data
benchmarks = calc.calculate_industry_benchmarks(
    all_banks_data[all_banks_data.index.size > 10],  # Minimum 10 banks
    metrics
)

# ✗ Avoid: Calculating with too few banks
benchmarks = calc.calculate_industry_benchmarks(
    all_banks_data[all_banks_data.index.size < 3],  # Too few
    metrics
)
```

### 2. Check Adjustment Confidence
```python
adjustment = calc.adjust_risk_score(...)

if adjustment['adjustment_confidence'] < 0.5:
    logger.warning(f"Low confidence adjustment: {adjustment['reason']}")
    # Consider reverting to absolute score
    adjusted_score = adjustment['absolute_score']
else:
    adjusted_score = adjustment['adjusted_score']
```

### 3. Document Macro-Adjustments in Reports
```python
report = f"""
Credit Risk Score: {absolute_score:.1f}
Industry Average: {benchmarks['mean']:.1f}
Bank Z-Score: {z_score:.2f}σ
Systemic Stress: {stress:.1%}
→ Adjusted Score: {adjusted_score:.1f} ({adjustment_delta:+.1f})

Interpretation: {reason}
"""
```

### 4. Validate with Historical Data
```python
# Compare adjusted vs. absolute scores on past defaults
historical_defaults = df[df['defaulted'] == True]

# Do adjusted scores better predict defaults?
from sklearn.metrics import roc_auc_score

auc_absolute = roc_auc_score(df['defaulted'], df['absolute_score'])
auc_adjusted = roc_auc_score(df['defaulted'], df['adjusted_score'])

print(f"AUC Absolute: {auc_absolute:.3f}")
print(f"AUC Adjusted: {auc_adjusted:.3f}")
```

---

## Testing and Validation

### Run the Macro-Adjustments Example

```bash
python example_macro_adjustments.py
```

This demonstrates:
- Normal vs. stressed scenario comparison
- Same bank, different risk scores based on context
- Percentile ranking within industry
- Systemic stress estimation

### Output Example

```
SCENARIO 1: NORMAL ECONOMIC CONDITIONS
──────────────────────────────────────────────────────────
Industry Benchmarks (Normal Conditions):
  npl_ratio:
    Mean: 1.45%, Std: 0.48%
    Range: 0.50% - 2.85%

  liquidity_ratio:
    Mean: 46.32%, Std: 8.21%
    Range: 32.10% - 62.15%

Systemic Stress Level: 3.2% (Normal)

Risk Score Adjustment (Normal Scenario):
  Absolute Score: 45.0
  Adjusted Score: 48.5
  Delta: +3.5
  Z-Score: +2.50σ
  Reason: Bank is 2.50σ above industry average...

────────────────────────────────────────────────────────────
SCENARIO 2: STRESSED ECONOMIC CONDITIONS (RECESSION)

Industry Benchmarks (Stressed Conditions):
  npl_ratio:
    Mean: 3.55%, Std: 1.18%
    Range: 1.85% - 6.20%

Systemic Stress Level: 68.3% (Elevated)

Risk Score Adjustment (Stressed Scenario):
  Absolute Score: 55.0
  Adjusted Score: 52.1
  Delta: -2.9
  Z-Score: +0.45σ
  Reason: Industry stress level 68.3% reduces relative concern...
```

---

## References and Further Reading

- **Z-Scores**: [Wikipedia - Standard Score](https://en.wikipedia.org/wiki/Standard_score)
- **Systemic Risk**: Basel Committee on Banking Supervision (BCBS) - G-SIB Framework
- **Procyclicality**: Basel III regulatory framework countercyclical capital buffer
- **Peer Analysis**: Typical practice in bank stress testing and peer grouping

---

## Troubleshooting

### Q: Adjusted scores are too similar to absolute scores
**A**: Increase `adjustment_strength` parameter (default 0.35). Try 0.5 or 0.6 for stronger context weighting.

### Q: Adjustment confidence is always low
**A**: Check if you have enough banks for benchmarking. Minimum 10-15 banks recommended. Low confidence occurs when:
- Bank value very close to mean
- Industry-wide stress is very high
- Standard deviation is small

### Q: Z-scores seem extreme
**A**: This is correct if:
- Few banks and one is an outlier
- Early in data collection (small sample)
**Solution**: Filter outliers or increase minimum bank count.

### Q: Should I use absolute or adjusted scores for regulatory reporting?
**A**: Use **both**:
- **Absolute** for regulatory compliance (vs. Basel thresholds)
- **Adjusted** for risk contextualization and stress assessment
- **Report both** with clear labeling

---

## Version History

- **v1.0 (Current)**: Initial macro-adjustment framework
  - Industry benchmarking
  - Z-score calculation
  - Systemic stress estimation
  - Multi-indicator adjustment
  - Integration with credit and liquidity models

---

**Last Updated**: 2024  
**Maintainer**: Bank Audit System Team
