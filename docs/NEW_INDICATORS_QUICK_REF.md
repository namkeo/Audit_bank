# New Risk Indicators - Quick Reference

## Summary

Added 21 new risk indicators across 3 categories to enhance risk detection:

| Category | Indicators | Purpose |
|----------|-----------|---------|
| **Net Interest Margin** | 3 | Profitability & interest rate risk |
| **Off-Balance Sheet** | 6 | Hidden leverage & liquidity risk |
| **Concentration** | 8 | Diversification risk |
| **Composite Risks** | 3 | Aggregate risk scores |
| **Total** | **21** | Comprehensive risk coverage |

---

## Quick Access

### 1. Net Interest Margin (NIM)

**Key Metric**: `net_interest_margin`  
**Threshold**: ≥1.5%  
**Formula**: (Interest Income - Interest Expense) / Total Assets

```python
if nim < 0.015:
    print("WARNING: NIM below minimum threshold")
```

### 2. Off-Balance Sheet Exposures

**Key Metrics**:
- `derivatives_to_assets_ratio` (max: 2.0)
- `unused_lines_to_loans_ratio` (max: 0.8)
- `obs_risk_indicator` (max: 0.7)

```python
if obs_risk_indicator > 0.7:
    print("HIGH RISK: Excessive off-balance sheet exposure")
```

### 3. Concentration Metrics

**Deposit Concentration**:
- `top20_depositors_ratio` (max: 50%)
- `top5_depositors_ratio` (max: 30%)

**Loan Concentration**:
- `sector_concentration_hhi` (max: 0.35)
- `top20_borrower_concentration` (max: 25%)
- `geographic_concentration` (max: 0.75)

```python
if top20_depositors_ratio > 0.5:
    print("HIGH RISK: Deposit concentration too high")
    
if sector_concentration_hhi > 0.35:
    print("CRITICAL: Insufficient loan diversification")
```

---

## Implementation Status

✅ **Dataset Enhanced**: `time_series_dataset_enriched_v2.csv` (45→66 columns)  
✅ **Ratios Calculated**: All new indicators in `4_utility_functions.py`  
✅ **Thresholds Defined**: Expert rules updated in `expert_rules.json`  
✅ **System Integrated**: Working with all risk models  
✅ **Documentation**: Complete guide in `NEW_INDICATORS_GUIDE.md`

---

## Usage

### Generate Enhanced Dataset
```bash
python add_new_indicators.py
```

### Run Audit with New Indicators
```bash
python 0_example_usage.py
```

### Access New Indicators
```python
# In risk assessment results
ratios = FinancialRatioCalculator.calculate_ratios(bank_data)

nim = ratios['net_interest_margin']
obs_risk = ratios['obs_risk_indicator']
deposit_conc = ratios['top20_depositors_ratio']
loan_conc = ratios['sector_concentration_hhi']
```

---

## Key Thresholds (expert_rules.json)

```json
{
  "profitability": {
    "min_net_interest_margin": 0.015
  },
  "concentration_limits": {
    "max_top20_depositors_ratio": 0.50,
    "max_sector_hhi": 0.35,
    "max_top20_borrower_concentration": 0.25
  },
  "off_balance_sheet": {
    "max_derivatives_to_assets": 2.0,
    "max_obs_to_assets": 1.5
  },
  "composite_risks": {
    "max_liquidity_concentration_risk": 0.60,
    "max_credit_concentration_risk": 0.50,
    "max_obs_risk_indicator": 0.70
  }
}
```

---

## Risk Interpretation

### Composite Risk Scores (0-1 scale)

| Score | Level | Action |
|-------|-------|--------|
| 0.0-0.3 | **Low** | Monitor |
| 0.3-0.5 | **Moderate** | Review |
| 0.5-0.7 | **High** | Investigate |
| >0.7 | **Critical** | Immediate action |

### Concentration HHI

| HHI | Interpretation |
|-----|----------------|
| <0.15 | Highly diversified |
| 0.15-0.25 | Moderate concentration |
| 0.25-0.35 | High concentration (warning) |
| >0.35 | Very high (critical) |

---

## Files Created/Modified

### New Files
1. `add_new_indicators.py` - Dataset enrichment script
2. `time_series_dataset_enriched_v2.csv` - Enhanced dataset
3. `NEW_INDICATORS_GUIDE.md` - Comprehensive documentation
4. `NEW_INDICATORS_QUICK_REF.md` - This file

### Modified Files
1. `4_utility_functions.py` - Added new ratio calculations
2. `expert_rules.json` - Added thresholds for new indicators
3. `0_example_usage.py` - Updated to use new dataset

---

## Benefits

✅ **Enhanced Risk Detection**: 21 additional risk signals  
✅ **Regulatory Alignment**: Basel III OBS treatment, concentration limits  
✅ **Early Warning**: Concentration risks detected before crisis  
✅ **Comprehensive Coverage**: Profitability + Hidden risks + Diversification  
✅ **Explainable**: All indicators have clear thresholds and meanings

---

## Example Output

```
NIM range: 1.50% - 5.50%
Derivatives/Assets: 0.15 - 2.88
Top 20 concentration: 24.6% - 61.4%
Sector HHI: 0.2677 - 0.4929
```

---

## For More Information

See [NEW_INDICATORS_GUIDE.md](NEW_INDICATORS_GUIDE.md) for:
- Detailed formulas
- Regulatory context
- Real-world examples
- Usage examples
- Full implementation details
