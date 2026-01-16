# -*- coding: utf-8 -*-
"""
Display Quarterly Risk Scores from Outputs Folder
"""

import pandas as pd
import os
import sys

def show_quarterly_results():
    """Display quarterly risk scores from outputs folder"""
    
    csv_file = "outputs/quarterly_risk_scores.csv"
    excel_file = "outputs/quarterly_risk_scores.xlsx"
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        print("Please run: python generate_quarterly_scores.py")
        return False
    
    # Load data
    df = pd.read_csv(csv_file)
    
    print("="*80)
    print("QUARTERLY RISK SCORES - OUTPUTS FOLDER")
    print("="*80)
    print(f"CSV File: {csv_file} ({os.path.getsize(csv_file):,} bytes)")
    if os.path.exists(excel_file):
        print(f"Excel File: {excel_file} ({os.path.getsize(excel_file):,} bytes)")
    print()
    
    print(f"Total Records: {len(df)}")
    print(f"Banks: {df['bank_name'].nunique()}")
    print(f"Bank IDs: {', '.join(sorted(df['bank_name'].unique()))}")
    print(f"Quarters: {df['period'].nunique()}")
    print(f"Date Range: {df['period'].min()} to {df['period'].max()}")
    print()
    
    print("COLUMNS:")
    for col in df.columns:
        print(f"  - {col}")
    print()
    
    print("="*80)
    print("SAMPLE DATA (First 15 rows)")
    print("="*80)
    print(df.head(15).to_string(index=False))
    print()
    
    print("="*80)
    print("SUMMARY STATISTICS BY BANK")
    print("="*80)
    summary = df.groupby('bank_name')['quarterly_risk_score'].agg([
        ('Mean', 'mean'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Std Dev', 'std'),
        ('Count', 'count')
    ]).round(2)
    print(summary.to_string())
    print()
    
    print("="*80)
    print("RISK LEVEL DISTRIBUTION BY BANK")
    print("="*80)
    risk_dist = pd.crosstab(df['bank_name'], df['risk_level'], margins=True)
    print(risk_dist.to_string())
    print()
    
    print("="*80)
    print("LATEST QUARTER SCORES (2022-12-31)")
    print("="*80)
    latest = df[df['period'] == '2022-12-31'][[
        'bank_name', 
        'quarterly_risk_score', 
        'credit_score', 
        'liquidity_score', 
        'anomaly_score', 
        'risk_level'
    ]].sort_values('quarterly_risk_score', ascending=False)
    print(latest.to_string(index=False))
    print()
    
    print("="*80)
    print("TREND ANALYSIS (Latest vs Earliest Quarter)")
    print("="*80)
    trend_data = []
    for bank in sorted(df['bank_name'].unique()):
        bank_df = df[df['bank_name'] == bank].sort_values('period')
        earliest = bank_df.iloc[0]['quarterly_risk_score']
        latest = bank_df.iloc[-1]['quarterly_risk_score']
        trend = latest - earliest
        trend_data.append({
            'Bank': bank,
            'Earliest (2018 Q1)': f"{earliest:.2f}",
            'Latest (2022 Q4)': f"{latest:.2f}",
            'Change': f"{trend:+.2f}",
            'Trend': '↑ Worsening' if trend > 0 else '↓ Improving'
        })
    
    trend_df = pd.DataFrame(trend_data)
    print(trend_df.to_string(index=False))
    print()
    
    print("="*80)
    print("FILES LOCATION")
    print("="*80)
    print(f"CSV:   {os.path.abspath(csv_file)}")
    if os.path.exists(excel_file):
        print(f"Excel: {os.path.abspath(excel_file)}")
        print()
        print("Excel file contains 4 sheets:")
        print("  1. Quarterly Scores - All 200 quarterly records")
        print("  2. Bank Summary - Statistics by bank")
        print("  3. Risk Level Distribution - Risk breakdown")
        print("  4. Trend Analysis - Time series pivot table")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = show_quarterly_results()
    sys.exit(0 if success else 1)
