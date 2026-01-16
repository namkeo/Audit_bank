#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify the banks_components_analysis.xlsx file structure
"""
import pandas as pd
import os

os.chdir(r"D:\2026\KTNN\Research audit bank\new source vscode 1.1")

path = "outputs/banks_components_analysis.xlsx"

print("="*80)
print(f"VERIFYING: {path}")
print("="*80)

try:
    # Check if file exists
    if not os.path.exists(path):
        print(f"ERROR: File does not exist: {path}")
        exit(1)
    
    # Read Excel file
    excel_file = pd.ExcelFile(path)
    
    print(f"\nExcel Sheet Names: {excel_file.sheet_names}")
    print()
    
    # Check Component Scores sheet
    print("="*80)
    print("SHEET 1: Component Scores")
    print("="*80)
    df_comp = pd.read_excel(path, sheet_name='Component Scores')
    print(f"Shape: {df_comp.shape}")
    print(f"Columns: {df_comp.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df_comp.head(3).to_string())
    
    # Verify quantile_classification exists
    if 'Quantile Classification' in df_comp.columns:
        print(f"\n[OK] 'Quantile Classification' column found!")
        print(f"Values: {df_comp['Quantile Classification'].unique()}")
    else:
        print(f"\n[ERROR] 'Quantile Classification' column NOT found!")
    
    # Check Risk Analysis sheet
    print("\n" + "="*80)
    print("SHEET 2: Risk Analysis")
    print("="*80)
    df_risk = pd.read_excel(path, sheet_name='Risk Analysis')
    print(f"Shape: {df_risk.shape}")
    print(f"Columns: {df_risk.columns.tolist()}")
    print(f"First row Bank ID: {df_risk['Bank ID'].iloc[0]}")
    
    # Check Recommendations sheet
    print("\n" + "="*80)
    print("SHEET 3: Recommendations")
    print("="*80)
    df_rec = pd.read_excel(path, sheet_name='Recommendations')
    print(f"Shape: {df_rec.shape}")
    print(f"Columns: {df_rec.columns.tolist()}")
    print(f"Priorities: {df_rec['Priority'].unique()}")
    
    print("\n" + "="*80)
    print("SUCCESS: All sheets verified!")
    print("="*80)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
