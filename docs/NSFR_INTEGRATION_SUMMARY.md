# NSFR (Net Stable Funding Ratio) Integration Summary

## Overview
Net Stable Funding Ratio (NSFR) has been successfully incorporated into the bank audit system to complement existing liquidity metrics and assess structural funding risk per Basel III requirements.

## What is NSFR?

**Definition**: NSFR = Available Stable Funding (ASF) / Required Stable Funding (RSF)

**Basel III Requirement**: NSFR ≥ 100% over a one-year horizon

**Purpose**: 
- Measures structural funding stability over medium term (1 year)
- Complements LCR (Liquidity Coverage Ratio) which focuses on short-term stress (30 days)
- Ensures banks maintain stable funding profiles relative to asset composition

## Integration Points

### 1. Dataset Enrichment (`dataset_enrichment.py`)
**Lines**: 27-47 (function), 54-85 (calculation loop)

**Calculation Method**:
```python
# Available Stable Funding (ASF)
stable_deposits_pct = 0.65 + 0.20 * rng.rand()
asf = total_deposits * stable_deposits_pct + long_term_funding

# Required Stable Funding (RSF)
rsf_factor = 0.70 + 0.15 * rng.rand()
rsf = total_assets * rsf_factor

# NSFR calculation
nsfr = asf / rsf
nsfr = np.clip(nsfr, 0.80, 1.30)  # Capped at realistic range
```

**Features**:
- Deterministic generation (bank-seeded RNG ensures reproducibility)
- Correlation with liquidity_ratio (better short-term liquidity → higher NSFR)
- Realistic range: 80%-130%
- Mean ~105%, matching industry averages

**Dataset Status**:
- ✅ 200 rows regenerated with NSFR column
- ✅ 140 banks (70%) compliant with Basel III ≥100% requirement
- ✅ 60 banks (30%) below minimum (structural funding stress)

### 2. Expert Rules Configuration (`expert_rules.json`)
**Line**: ~11 (in liquidity section)

**Addition**:
```json
{
  "liquidity": {
    "min_lcr": 1.0,
    "max_loan_to_deposit": 0.9,
    "min_nsfr": 1.0
  }
}
```

**Purpose**: Centralized Basel III threshold configuration

### 3. Liquidity Risk Model Features (`2_model_liquidity_risk.py`)
**Line**: ~135 (in `_prepare_liquidity_features()`)

**Addition**:
```python
# Include NSFR (Net Stable Funding Ratio) for Basel III compliance
if 'nsfr' in data.columns:
    cols_to_add.append('nsfr')
```

**Purpose**: Ensure NSFR flows into model feature set

### 4. Liquidity Indicators Calculation (`2_model_liquidity_risk.py`)
**Lines**: ~430-455 (in `_calculate_liquidity_indicators()`)

**Addition**:
```python
# NSFR (Net Stable Funding Ratio) - Basel III structural funding requirement
nsfr = bank_data.get('nsfr', 1.0)
threshold = self.expert_rules['liquidity'].get('min_nsfr', 1.0)

indicators['nsfr'] = {
    'value': nsfr,
    'status': 'ADEQUATE' if nsfr >= threshold else 'INADEQUATE',
    'threshold': threshold,
    'description': f'Net Stable Funding Ratio: {nsfr:.2%} (min: {threshold:.0%})',
    'interpretation': (
        f"NSFR at {nsfr:.1%} {'meets' if nsfr >= threshold else 'BELOW'} "
        f"Basel III minimum of {threshold:.0%}. "
        "NSFR measures structural funding stability over 1-year horizon, "
        "complementing LCR's 30-day stress coverage."
    )
}
```

**Features**:
- Status: ADEQUATE / INADEQUATE based on 100% threshold
- Threshold loaded from expert_rules config
- Interpretation explains Basel III context and relationship to LCR

### 5. Macro-Adjustments Stress Detection (`2_model_liquidity_risk.py`)
**Lines**: ~269-273 (in `apply_macro_adjustments()`)

**Addition**:
```python
# Define stress metrics (including structural funding via NSFR)
stress_metrics = ['liquidity_ratio', 'lcr', 'nsfr']
thresholds = {
    'liquidity_ratio': ('<', 0.3),
    'lcr': ('<', 1.0),
    'nsfr': ('<', 1.0)  # Basel III minimum: NSFR >= 100%
}
```

**Purpose**: 
- Systemic stress detection includes NSFR failures
- Marks industry-wide structural funding crises
- Banks with NSFR < 100% contribute to systemic stress calculation

### 6. Macro-Adjustments Benchmarking (`2_model_liquidity_risk.py`)
**Lines**: ~286-289 (in `apply_macro_adjustments()`)

**Addition**:
```python
# Calculate benchmarks for key liquidity metrics (including NSFR for structural funding)
liquidity_metrics = ['liquidity_ratio', 'lcr', 'nsfr']
benchmarks = calc.calculate_industry_benchmarks(all_banks_data, liquidity_metrics)
```

**Purpose**: 
- NSFR included in industry benchmark calculations
- Enables relative positioning (z-scores) for NSFR
- Supports context-aware risk adjustments during structural funding crises

## Verification Results

### Dataset Statistics
```
Total rows: 200
NSFR range: 90.2% - 119.6%
NSFR mean: 105.3%
```

### Basel III Compliance
```
Banks ≥ 100% (compliant): 140 (70.0%)
Banks < 100% (at risk): 60 (30.0%)
```

### Integration Checklist
- ✅ NSFR generation in dataset enrichment
- ✅ NSFR threshold in expert rules (min_nsfr: 1.0)
- ✅ NSFR added to liquidity model features
- ✅ NSFR indicator calculation with Basel III interpretation
- ✅ NSFR in macro-adjustments stress detection
- ✅ NSFR in macro-adjustments benchmarking
- ✅ Dataset regenerated with 200 rows
- ✅ Verification test confirms all integration points

## How NSFR Complements LCR

| Metric | Time Horizon | Purpose | Stress Scenario |
|--------|--------------|---------|-----------------|
| **LCR** | 30 days | Short-term liquidity stress | Bank run, deposit withdrawal |
| **NSFR** | 1 year | Structural funding stability | Funding market disruption, maturity mismatch |

**Combined Assessment**:
- Low LCR + Adequate NSFR → Short-term stress, structurally sound
- Adequate LCR + Low NSFR → Can survive immediate stress, but maturity mismatch
- Low LCR + Low NSFR → Severe liquidity crisis (both immediate and structural)

## Basel III Context

**Regulatory Background**:
- Basel III introduced NSFR as complementary liquidity metric post-2008 crisis
- NSFR addresses structural vulnerabilities beyond short-term liquidity
- Minimum 100% requirement ensures banks can fund assets with stable sources over 1 year

**Components**:
- **Available Stable Funding (ASF)**: Retail deposits, long-term wholesale funding, equity
- **Required Stable Funding (RSF)**: Based on asset liquidity (illiquid assets require more stable funding)

## Usage in Audit Reports

When analyzing bank audit reports, NSFR now appears in:

1. **Liquidity Indicators Section**:
   ```json
   "nsfr": {
     "value": 1.053,
     "status": "ADEQUATE",
     "threshold": 1.0,
     "description": "Net Stable Funding Ratio: 105.30% (min: 100%)",
     "interpretation": "NSFR at 105.3% meets Basel III minimum..."
   }
   ```

2. **Macro-Adjustments Context**:
   - Systemic stress includes % of banks with NSFR < 100%
   - Industry benchmarks show relative NSFR positioning (z-score)
   - Risk score adjustments account for structural funding stress

3. **Violations/Flags**:
   - Banks with NSFR < 100% flagged for Basel III non-compliance
   - Structural funding risk highlighted in audit summary

## Files Modified

| File | Lines Modified | Purpose |
|------|----------------|---------|
| `dataset_enrichment.py` | 27-47, 54-85 | NSFR generation logic |
| `expert_rules.json` | ~11 | min_nsfr: 1.0 threshold |
| `2_model_liquidity_risk.py` | ~135 | Add NSFR to features |
| `2_model_liquidity_risk.py` | ~430-455 | NSFR indicator calculation |
| `2_model_liquidity_risk.py` | ~269-273 | NSFR in stress detection |
| `2_model_liquidity_risk.py` | ~286-289 | NSFR in benchmarking |

**Total**: 4 files modified with ~60 lines of new code

## Testing

**Verification Script**: `test_nsfr_integration.py`

**Test Coverage**:
- ✅ NSFR column exists in enriched dataset
- ✅ NSFR statistics (min, mean, max) are realistic
- ✅ Basel III compliance distribution (70% compliant)
- ✅ All integration points confirmed

**No Breaking Changes**: All modifications are additive; existing functionality preserved.

## Next Steps (Optional Enhancements)

1. **Documentation Updates**:
   - Add NSFR to `MACRO_ADJUSTMENTS_QUICK_REF.md`
   - Update `BATCH_PROCESSING_GUIDE.md` to mention NSFR indicators

2. **Example Updates**:
   - Update `example_audit_with_macro_adjustments.py` to show NSFR in stressed scenarios
   - Create NSFR-specific example showing structural funding crisis

3. **Advanced Analysis**:
   - Add NSFR decomposition (ASF/RSF breakdown) to detailed reports
   - Correlation analysis between NSFR and NPL ratios during stress

4. **Visualization**:
   - Time series charts showing NSFR trends
   - Industry distribution histograms comparing NSFR across sectors

---

**Status**: ✅ **COMPLETE** - NSFR fully integrated into bank audit system with Basel III compliance checking and macro-adjustment contextualization.
