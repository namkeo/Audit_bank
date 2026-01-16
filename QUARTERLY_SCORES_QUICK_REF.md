# Quarterly Risk Scores - Quick Reference

## Output Files Location

**Folder:** `outputs/`

### Files Created:
1. **quarterly_risk_scores.csv** (8.7 KB)
   - 200 rows (10 banks × 20 quarters)
   - Period: 2018 Q1 to 2022 Q4

2. **quarterly_risk_scores.xlsx** (13.9 KB)
   - Sheet 1: Quarterly Scores (all 200 records)
   - Sheet 2: Bank Summary (statistics)
   - Sheet 3: Risk Level Distribution
   - Sheet 4: Trend Analysis (pivot table)

## How to View Results

### Option 1: Run Display Script
```bash
python show_quarterly_results.py
```

### Option 2: Regenerate and Display
```bash
python generate_quarterly_scores.py
python show_quarterly_results.py
```

### Option 3: Open Excel File
```bash
start outputs\quarterly_risk_scores.xlsx
```

## Key Findings

### All Banks Showing Worsening Trends (2018 Q1 → 2022 Q4):
- **ACB**: +7.75 points (52.45 → 60.20) - Largest deterioration
- **BIDV**: +5.25 points (52.50 → 57.75)
- **TCB**: +4.20 points (54.50 → 58.70)
- **STB**: +3.75 points (57.75 → 61.50) - Highest score in 2022 Q4
- **CTG**: +2.80 points (56.95 → 59.75)
- **VCB**: +2.50 points (54.95 → 57.45)
- **MBB**: +2.50 points (55.25 → 57.75)
- **VPB**: +2.45 points (49.00 → 51.45)
- **HDB**: +1.75 points (46.75 → 48.50) - Lowest score overall
- **EIB**: +1.25 points (58.45 → 59.70) - Smallest increase

### Risk Level Distribution:
- **HIGH Risk**: 22 quarters total (11% of all quarters)
  - ACB: 14 quarters (70% of ACB's quarters)
  - EIB: 5 quarters (25%)
  - STB: 2 quarters (10%)
  - CTG: 1 quarter (5%)
  
- **MEDIUM Risk**: 178 quarters total (89% of all quarters)
  - 6 banks have 100% MEDIUM risk (BIDV, HDB, MBB, TCB, VCB, VPB)

### Latest Quarter (2022 Q4) Rankings:
1. **STB**: 61.50 (HIGH)
2. **ACB**: 60.20 (HIGH)
3. **CTG**: 59.75 (MEDIUM)
4. **EIB**: 59.70 (MEDIUM)
5. **TCB**: 58.70 (MEDIUM)
6. **BIDV**: 57.75 (MEDIUM)
7. **MBB**: 57.75 (MEDIUM)
8. **VCB**: 57.45 (MEDIUM)
9. **VPB**: 51.45 (MEDIUM)
10. **HDB**: 48.50 (MEDIUM)

## Data Columns

- **period**: Quarter date (YYYY-MM-DD)
- **bank_name**: Bank identifier
- **credit_score**: Credit risk (0-100)
- **liquidity_score**: Liquidity risk (0-100)
- **anomaly_score**: Anomaly detection (0-100)
- **quarterly_risk_score**: Composite score (weighted average)
- **risk_level**: Classification (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)

## Score Calculation

Quarterly Risk Score = (Credit × 40%) + (Liquidity × 35%) + (Anomaly × 25%)

### Risk Level Classification:
- 0-20: MINIMAL
- 20-40: LOW
- 40-60: MEDIUM
- 60-80: HIGH
- 80-100: CRITICAL

## Files Reference

- **Generate scores**: `generate_quarterly_scores.py`
- **Display results**: `show_quarterly_results.py`
- **Core module**: `7_quarterly_scores.py`
- **Documentation**: `QUARTERLY_SCORES_IMPLEMENTATION.md`
