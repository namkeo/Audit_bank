# Quarterly Results Integration - Summary

## What Was Modified

The `0_example_usage.py` file has been updated to automatically generate quarterly risk scores when executed.

### Changes Made:

1. **Added Imports** (Lines 17-21)
   - Imported `generate_quarterly_scores.py` module
   - Imported `generate_quarterly_excel.py` module

2. **Added Quarterly Generation** (Lines 1088-1102)
   - Automatically calls `generate_quarterly_scores_for_all_banks()` to create CSV data
   - Automatically calls `create_quarterly_results_excel()` to format Excel file
   - Includes error handling with detailed logging

## How to Use

Simply run the example usage script:

```bash
python 0_example_usage.py
```

This will:
1. ✅ Run all bank audits
2. ✅ Export component analysis
3. ✅ **Generate quarterly risk scores automatically**

## Output Files Generated

After running `0_example_usage.py`, the following files are created in the `outputs/` folder:

### Quarterly Results Files:
- **quarterly_results.xlsx** (10.9 KB)
  - Clean, formatted single-sheet Excel file
  - 200 quarterly records with color-coded risk levels
  - Auto-filters and frozen headers
  - Professional formatting

- **quarterly_risk_scores.xlsx** (13.9 KB)
  - Multi-sheet analysis file
  - Sheet 1: Quarterly Scores (all 200 records)
  - Sheet 2: Bank Summary (statistics)
  - Sheet 3: Risk Level Distribution
  - Sheet 4: Trend Analysis

- **quarterly_risk_scores.csv** (8.8 KB)
  - Raw data export
  - 200 rows × 7 columns
  - For data import and analysis

## Data Included

- **Time Period**: 2018 Q1 to 2022 Q4 (20 quarters)
- **Banks**: 10 Vietnamese banks (VCB, BIDV, CTG, ACB, MBB, TCB, VPB, HDB, EIB, STB)
- **Records**: 200 total (10 banks × 20 quarters)

### Columns:
- Period (quarter end date)
- Bank Name
- Credit Score (0-100)
- Liquidity Score (0-100)
- Anomaly Score (0-100)
- Quarterly Risk Score (weighted composite)
- Risk Level (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)

## Calculation Method

Each quarterly score is calculated independently:

```
Quarterly Risk Score = 
  (Credit Score × 40%) + 
  (Liquidity Score × 35%) + 
  (Anomaly Score × 25%)
```

This adds quarterly-level risk analysis **without changing** the existing overall risk scores.

## Key Features

✅ Automatic execution when running 0_example_usage.py  
✅ Color-coded risk levels in Excel (green → red)  
✅ Professional formatting with filters  
✅ Independent quarterly calculations  
✅ Comprehensive trend analysis  
✅ Statistical summaries by bank  

## Files Modified

- **0_example_usage.py** - Added quarterly generation imports and execution calls

## Files Created Previously

- `generate_quarterly_scores.py` - Core generation script
- `generate_quarterly_excel.py` - Formatting script
- `7_quarterly_scores.py` - Core calculation module
- `QUARTERLY_SCORES_IMPLEMENTATION.md` - Technical documentation
- `QUARTERLY_SCORES_QUICK_REF.md` - Quick reference guide

## Testing

Run the script:
```bash
cd "d:\2026\KTNN\Research audit bank\new source vscode 1.1"
python 0_example_usage.py
```

Check outputs folder for the three quarterly files.

---

**Status**: ✅ Integration Complete  
**Date**: January 16, 2026
