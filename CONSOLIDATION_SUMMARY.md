# Excel Consolidation Summary

## Objective Completed ✅

All Excel files have been successfully merged into a **single consolidated file** called `banks_components_analysis.xlsx` in the outputs folder.

## What Was Done

### 1. **Merged Files**
The following separate files were consolidated into one Excel workbook:
- `all_banks_summary.csv` → Now integrated into Component Scores sheet
- `banks_component_scores.csv` → Component Scores sheet
- `banks_risk_analysis.csv` → Risk Analysis sheet  
- `banks_recommendations.csv` → Recommendations sheet

### 2. **Excel File Structure**
**File:** `outputs/banks_components_analysis.xlsx`

**Sheet 1: Component Scores** (10 rows, 8 columns)
- Bank ID
- Overall Score
- Risk Level
- **Quantile Classification** ⭐ (NEW - added as requested)
- Credit Risk Score
- Liquidity Risk Score
- Anomaly Score
- Highest Risk Area

**Sheet 2: Risk Analysis** (10 rows, 5 columns)
- Bank ID
- Credit Risk Analysis
- Liquidity Risk Analysis
- Anomaly Analysis
- Overall Assessment

**Sheet 3: Recommendations** (30 rows, 6 columns)
- Bank ID
- Priority
- Risk Area
- Issue
- Recommendation
- Timeline

### 3. **Quantile Classification Column** ✅
- Added to Component Scores sheet with values: HIGH, MEDIUM, LOW, MINIMAL
- Computed from the top 25%, 50%, 75% percentiles of overall risk scores
- Pre-loads from `all_banks_summary.csv` when available

### 4. **Code Changes Made**

**File: `0_example_usage.py`**
- Added openpyxl style imports at module level
- Created `_attach_quantile_classification()` function to add the new column
- Modified `export_component_analysis_to_excel()` to:
  - Merge all data into one Excel file
  - Clean up old CSV files after export
  - Call the quantile classification function
- Updated column widths in `_format_component_sheet()` to accommodate new column

**File: `generate_excel_analysis.py`**
- Created standalone `_attach_quantile_classification()` function
- Added cleanup code to remove old Excel files before creating new one
- Updated column formatting for new column order
- Fixed Unicode character encoding issues for cross-platform compatibility

### 5. **Cleanup Automation** 🧹
- Automatic removal of old CSV export files after successful Excel generation
- Automatic removal of old Excel files before creating the consolidated version
- Only `all_banks_summary.csv` and `banks_components_analysis.xlsx` remain in outputs

## Verification Results

```
Excel File: banks_components_analysis.xlsx
├── Sheet 1: Component Scores
│   ├── 10 banks analyzed
│   ├── Quantile Classification: ✅ PRESENT
│   ├── Values: HIGH, MEDIUM, LOW, MINIMAL
│   └── Status: FORMATTED & SORTED
│
├── Sheet 2: Risk Analysis
│   ├── 10 detailed bank assessments
│   └── Status: COMPLETE
│
└── Sheet 3: Recommendations
    ├── 30 actionable recommendations
    ├── Sorted by Priority
    └── Status: COMPLETE
```

## How to Run

### Option 1: Run Full Workflow (Recommended)
```bash
python 0_example_usage.py
```
This will:
1. Audit all 10 banks
2. Generate bank reports
3. Create merged Excel file
4. Clean up old CSV files
5. Output: `outputs/banks_components_analysis.xlsx`

### Option 2: Generate Excel from Existing Reports
```bash
python generate_excel_analysis.py
```
(Requires pre-existing bank report JSON files)

## Key Features Implemented

✅ Single consolidated Excel file  
✅ Quantile Classification column added  
✅ Proper column ordering and formatting  
✅ Automatic cleanup of old files  
✅ All 3 sheets (Component Scores, Risk Analysis, Recommendations)  
✅ Professional formatting with frozen headers  
✅ Cross-platform compatible (fixed Unicode issues)  

## Output Files in `outputs/` folder

**Excel:**
- `banks_components_analysis.xlsx` ← **MAIN FILE** (all data here)

**Supporting Files:**
- `all_banks_summary.csv` (keeps quantile data for reference)
- `*_bank_report.json` (10 individual bank JSON reports)
- `*_bank_dashboard.png` (10 bank dashboard images)

**Status:** ✅ Only 1 Excel file in outputs (plus supporting data files)
