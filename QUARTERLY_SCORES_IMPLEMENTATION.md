# Quarterly Risk Scoring Implementation - Summary Report

## Overview
Successfully implemented quarterly-level risk scoring for the Bank Audit System. This feature calculates individual risk scores for each of the 20 quarterly periods independently, allowing for tracking of risk evolution over time.

## What Was Done

### 1. Created New Quarterly Scoring Module (`7_quarterly_scores.py`)
- **QuarterlyScoreCalculator class**: Calculates risk scores for each quarter independently
- **Calculation method**: 
  - Credit Risk Score (40% weight)
  - Liquidity Risk Score (35% weight)  
  - Anomaly Score (25% weight)
  - Composite Quarterly Risk Score = weighted average of three components

### 2. Quarter-Level Calculations

#### Credit Score Calculation (Per Quarter):
- Based on NPL ratio, Capital Adequacy Ratio (CAR), Return on Assets (ROA)
- NPL penalty: Higher NPL = Higher risk
- CAR reward: Higher CAR = Lower risk  
- ROA indicator: Profitability reflects credit quality

#### Liquidity Score Calculation (Per Quarter):
- Based on Liquidity Coverage Ratio (LCR), Loan-to-Deposit ratio (LTD), Net Stable Funding Ratio (NSFR)
- LCR benchmark: >= 1.5 optimal, < 0.5 critical
- LTD benchmark: 0.7-1.0 optimal
- NSFR benchmark: >= 1.2 optimal

#### Anomaly Score Calculation (Per Quarter):
- Based on OBS to Assets ratio, Derivatives exposure, Concentration risk
- Higher concentrations = Higher anomaly risk
- Geographic concentration considered

### 3. Output Files Generated

**File**: `outputs/quarterly_risk_scores.csv`
- **Rows**: 200 (10 banks × 20 quarters)
- **Columns**: 
  - `period` - Quarter date
  - `bank_name` - Bank identifier
  - `credit_score` - Quarterly credit risk score (0-100)
  - `liquidity_score` - Quarterly liquidity risk score (0-100)
  - `anomaly_score` - Quarterly anomaly score (0-100)
  - `quarterly_risk_score` - Composite quarterly score (0-100)
  - `risk_level` - Classification (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL)

**File**: `outputs/quarterly_risk_scores.xlsx`
Multiple sheets for comprehensive analysis:
- **Sheet 1: Quarterly Scores** - All 200 quarterly records
- **Sheet 2: Bank Summary** - Statistics by bank (mean, min, max, std, count)
- **Sheet 3: Risk Level Distribution** - Risk level breakdown by bank and quarter
- **Sheet 4: Trend Analysis** - Quarter-by-quarter trends across all banks

### 4. Generation Script (`generate_quarterly_scores.py`)
- Loads time-series data with 20 quarterly periods per bank
- Calculates scores for each quarter independently
- Exports to both CSV and Excel formats
- Generates comprehensive summary statistics

## Key Statistics

All 10 banks analyzed with 20 quarterly periods each:

| Bank | Avg Score | Range | Latest Score | Trend |
|------|-----------|-------|--------------|-------|
| VCB  | 55.45 | 54.95 - 57.45 | 57.45 | +2.50 |
| BIDV | 53.88 | 52.50 - 57.75 | 57.75 | +5.25 |
| CTG  | 58.70 | 56.95 - 62.25 | 59.75 | +2.80 |
| ACB  | 58.81 | 52.45 - 60.20 | 60.20 | +7.75 |
| MBB  | 55.62 | 55.25 - 57.75 | 57.75 | +2.50 |
| TCB  | 57.39 | 54.50 - 58.70 | 58.70 | +4.20 |
| VPB  | 48.99 | 46.50 - 51.45 | 51.45 | +2.45 |
| HDB  | 48.06 | 46.75 - 48.50 | 48.50 | +1.75 |
| EIB  | 58.32 | 54.45 - 60.45 | 59.70 | +1.25 |
| STB  | 58.08 | 55.25 - 61.50 | 61.50 | +3.75 |

## Risk Level Distribution

- **MINIMAL**: Most banks in 0% of quarters (none)
- **LOW**: Most banks in 0% of quarters (none)
- **MEDIUM**: Most banks in 75-100% of quarters
- **HIGH**: Some banks in 5-70% of quarters (ACB, CTG, EIB, STB)
- **CRITICAL**: 0% across all banks

## Trend Analysis

**Overall Risk Trends (Latest vs Earliest Quarter)**:
- **Positive trends** (deteriorating): All 10 banks show increasing scores
- **Largest increase**: ACB (+7.75 points)
- **Smallest increase**: HDB (+1.75 points)
- **Average increase**: +3.34 points across all banks

This indicates gradual risk escalation across the banking sector from 2018 Q1 to 2022 Q4.

## Preservation of Overall Risk Scores

**IMPORTANT**: The existing overall_risk_score calculation remains UNCHANGED:
- Overall scores still use ALL 20 time periods combined
- Overall scores are calculated as composite weighted average
- Early warning signals still applied to overall scores
- Quarterly scores are ADDITIVE new feature, not replacement

## Integration Points

### For Users:
1. **Quarterly file access**: `outputs/quarterly_risk_scores.csv` and `.xlsx`
2. **Trend tracking**: Use Trend Analysis sheet to visualize risk evolution
3. **Quarterly vs Overall**: 
   - Use quarterly scores to see period-by-period variations
   - Use overall scores for comprehensive risk assessment

### For Developers:
1. **Module import**: 
   ```python
   from 7_quarterly_scores import QuarterlyScoreCalculator
   from generate_quarterly_scores import generate_quarterly_scores_for_all_banks
   ```

2. **Standalone calculation**:
   ```python
   result = generate_quarterly_scores_for_all_banks(
       csv_path="time_series_dataset_enriched_v2.csv",
       output_dir="outputs"
   )
   ```

## Files Modified

- **New files created**:
  - `7_quarterly_scores.py` - Core quarterly scoring module
  - `generate_quarterly_scores.py` - Quarterly scores generation script
  - `verify_quarterly_implementation.py` - Verification script

- **Files unchanged**:
  - `0_example_usage.py` - Existing audit workflow unmodified
  - `3_reporting_analysis.py` - Overall score calculation unmodified
  - `5_bank_audit_system.py` - Audit system orchestration unmodified

## Verification Results

✓ Quarterly scores calculated for all 10 banks  
✓ All 20 periods processed for each bank (200 total rows)  
✓ CSV output file created successfully  
✓ Excel workbook with 4 sheets created successfully  
✓ Overall risk scores remain unchanged and preserved  
✓ No modifications to existing audit system or calculations  

## Usage Example

### Load and analyze quarterly scores:
```python
import pandas as pd

# Load quarterly scores
df = pd.read_csv('outputs/quarterly_risk_scores.csv')

# Filter by bank
vcb_quarterly = df[df['bank_name'] == 'VCB']

# Calculate statistics
print(f"VCB Average Score: {vcb_quarterly['quarterly_risk_score'].mean():.2f}")
print(f"VCB Score Trend: {vcb_quarterly['quarterly_risk_score'].iloc[-1] - vcb_quarterly['quarterly_risk_score'].iloc[0]:.2f}")

# Analyze risk levels
risk_distribution = df.groupby('risk_level')['bank_name'].value_counts()
print(risk_distribution)
```

## Next Steps (Optional Enhancements)

1. **Visualization**: Create time-series charts showing quarterly score trends
2. **Alerts**: Implement alerts for significant quarter-over-quarter changes
3. **Forecasting**: Use historical quarterly trends to forecast future scores
4. **Comparative Analysis**: Compare quarterly trends across banks
5. **Sensitivity Analysis**: Test impact of different component weights

## Notes

- Quarterly scores are calculated independently without using trained models
- Each quarter is treated as a separate snapshot for assessment
- Risk levels follow same classification as overall scores (0-100 scale)
- All 20 periods are equally weighted in the analysis
- Time period spans 2018 Q1 to 2022 Q4 (5 years of quarterly data)

---

**Generated**: 2026-01-16  
**Status**: ✓ Implementation Complete and Verified
