# -*- coding: utf-8 -*-
"""
Dataset Enhancement Script - Phase 3.5
Adds new concentration and off-balance sheet risk indicators
"""

import pandas as pd
import numpy as np


def add_new_indicators_to_dataset(input_file: str, output_file: str):
    """
    Add new risk indicators to existing enriched dataset:
    1. Net Interest Margin (NIM) components
    2. Off-Balance Sheet ratios
    3. Concentration metrics
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original: {len(df)} rows, {len(df.columns)} columns")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ==========================================================================
    # 1. NET INTEREST MARGIN COMPONENTS
    # ==========================================================================
    print("\n1. Adding Net Interest Margin components...")
    
    if 'interest_income' not in df.columns:
        # Estimate interest income (~70% of operating income for banks)
        df['interest_income'] = df['operating_income'] * 0.70
        
        # Add realistic variation by bank type
        large_bank_mask = df['bank_type'] == 'large'
        df.loc[large_bank_mask, 'interest_income'] *= np.random.uniform(0.95, 1.05, large_bank_mask.sum())
        df.loc[~large_bank_mask, 'interest_income'] *= np.random.uniform(0.90, 1.10, (~large_bank_mask).sum())
    
    if 'interest_expense' not in df.columns:
        # Estimate interest expense (~30-40% of operating income)
        df['interest_expense'] = df['operating_income'] * 0.35
        
        # Higher for banks with higher wholesale funding
        if 'wholesale_dependency_ratio' in df.columns:
            funding_premium = 1 + (df['wholesale_dependency_ratio'] * 0.2)
            df['interest_expense'] *= funding_premium
    
    # Calculate NIM if not exists or recalculate
    df['net_interest_margin'] = (df['interest_income'] - df['interest_expense']) / df['total_assets']
    
    # Ensure realistic NIM bounds (1.5% - 5.5%)
    df['net_interest_margin'] = df['net_interest_margin'].clip(0.015, 0.055)
    
    print(f"   ✓ Added: interest_income, interest_expense, net_interest_margin")
    print(f"   NIM range: {df['net_interest_margin'].min():.2%} - {df['net_interest_margin'].max():.2%}")
    
    # ==========================================================================
    # 2. OFF-BALANCE SHEET INDICATORS
    # ==========================================================================
    print("\n2. Adding Off-Balance Sheet indicators...")
    
    # Derivatives notional value
    if 'derivatives_notional' not in df.columns:
        # Base on bank size - larger banks have more derivatives
        bank_size_percentile = df['total_assets'].rank(pct=True)
        
        # Large banks: 50-200% of assets, smaller banks: 20-80%
        derivatives_ratio = 0.2 + (bank_size_percentile * 1.8)
        df['derivatives_notional'] = df['total_assets'] * derivatives_ratio
        
        # Add variation by bank type
        large_mask = df['bank_type'] == 'large'
        df.loc[large_mask, 'derivatives_notional'] *= np.random.uniform(1.2, 1.5, large_mask.sum())
        df.loc[~large_mask, 'derivatives_notional'] *= np.random.uniform(0.6, 0.9, (~large_mask).sum())
    
    # Unused credit lines (committed but undrawn)
    if 'unused_credit_lines' not in df.columns:
        # Typically 40-70% of total loans
        df['unused_credit_lines'] = df['total_loans'] * np.random.uniform(0.4, 0.7, len(df))
    
    # Guarantees and letters of credit
    if 'guarantees_issued' not in df.columns:
        # Typically 15-25% of total loans
        df['guarantees_issued'] = df['total_loans'] * np.random.uniform(0.15, 0.25, len(df))
    
    # Calculate OBS ratios
    df['derivatives_to_assets_ratio'] = df['derivatives_notional'] / df['total_assets']
    df['unused_lines_to_loans_ratio'] = df['unused_credit_lines'] / df['total_loans']
    df['guarantees_to_loans_ratio'] = df['guarantees_issued'] / df['total_loans']
    
    # Update total OBS exposure with proper risk weights
    df['obs_exposure_total'] = (
        df['derivatives_notional'] * 0.05 +  # 5% conversion (credit equivalent)
        df['unused_credit_lines'] * 0.5 +     # 50% conversion
        df['guarantees_issued'] * 0.8         # 80% conversion
    )
    df['obs_to_assets_ratio'] = df['obs_exposure_total'] / df['total_assets']
    
    print(f"   ✓ Added: derivatives_notional, unused_credit_lines, guarantees_issued")
    print(f"   ✓ Added: derivatives_to_assets_ratio, unused_lines_to_loans_ratio, obs_to_assets_ratio")
    print(f"   Derivatives/Assets: {df['derivatives_to_assets_ratio'].min():.2f} - {df['derivatives_to_assets_ratio'].max():.2f}")
    
    # ==========================================================================
    # 3. DEPOSIT CONCENTRATION METRICS
    # ==========================================================================
    print("\n3. Adding Deposit Concentration metrics...")
    
    # Top 20 depositors
    if 'top20_depositors' not in df.columns:
        # Higher concentration for smaller banks
        bank_size_percentile = df['total_assets'].rank(pct=True)
        
        # Small banks: 40-60%, Large banks: 20-35%
        top20_ratio = 0.60 - (bank_size_percentile * 0.35)
        top20_ratio = top20_ratio.clip(0.20, 0.65)
        
        # Add variation
        top20_ratio *= np.random.uniform(0.95, 1.05, len(df))
        
        df['top20_depositors'] = df['total_deposits'] * top20_ratio
        df['top20_depositors_ratio'] = top20_ratio
    
    # Top 5 depositors (higher concentration)
    if 'top5_depositors' not in df.columns:
        # Top 5 typically 35-55% of top 20
        top5_ratio = df['top20_depositors_ratio'] * np.random.uniform(0.35, 0.55, len(df))
        
        df['top5_depositors'] = df['total_deposits'] * top5_ratio
        df['top5_depositors_ratio'] = top5_ratio
    
    print(f"   ✓ Added: top20_depositors, top5_depositors with ratios")
    print(f"   Top 20 concentration: {df['top20_depositors_ratio'].min():.1%} - {df['top20_depositors_ratio'].max():.1%}")
    
    # ==========================================================================
    # 4. LOAN CONCENTRATION METRICS
    # ==========================================================================
    print("\n4. Adding Loan Concentration metrics...")
    
    # Sector concentration (HHI)
    sector_cols = ['sector_loans_energy', 'sector_loans_real_estate', 
                  'sector_loans_construction', 'sector_loans_services', 
                  'sector_loans_agriculture']
    
    if all(col in df.columns for col in sector_cols):
        # Calculate Herfindahl-Hirschman Index
        sector_data = df[sector_cols].fillna(0)
        sector_shares = sector_data.div(df['total_loans'], axis=0)
        df['sector_concentration_hhi'] = (sector_shares ** 2).sum(axis=1)
        
        print(f"   ✓ Added: sector_concentration_hhi")
        print(f"   Sector HHI: {df['sector_concentration_hhi'].min():.4f} - {df['sector_concentration_hhi'].max():.4f}")
    
    # Top 20 borrowers concentration
    if 'top20_borrower_loans' not in df.columns and 'top10_borrower_loans' in df.columns:
        # Top 20 typically 1.6-2.0x top 10
        df['top20_borrower_loans'] = df['top10_borrower_loans'] * np.random.uniform(1.6, 2.0, len(df))
        df['top20_borrower_concentration'] = df['top20_borrower_loans'] / df['total_loans']
        
        print(f"   ✓ Added: top20_borrower_loans, top20_borrower_concentration")
    
    # Geographic concentration
    if 'region' in df.columns and 'geographic_concentration' not in df.columns:
        # Banks in single region = 1.0, multiple regions lower
        region_diversity = df.groupby('bank_id')['region'].nunique()
        df['geographic_concentration'] = df['bank_id'].map(region_diversity).map(
            {1: 1.0, 2: 0.65, 3: 0.45, 4: 0.30}
        ).fillna(0.30)
        
        # Add variation
        df['geographic_concentration'] *= np.random.uniform(0.95, 1.05, len(df))
        df['geographic_concentration'] = df['geographic_concentration'].clip(0.25, 1.0)
        
        print(f"   ✓ Added: geographic_concentration")
    
    # ==========================================================================
    # 5. COMPOSITE RISK INDICATORS
    # ==========================================================================
    print("\n5. Adding Composite Risk Indicators...")
    
    # Liquidity-Concentration Risk
    df['liquidity_concentration_risk'] = (
        df['top20_depositors_ratio'] * 0.4 +
        df.get('wholesale_dependency_ratio', 0) * 0.3 +
        (1 - df['liquidity_coverage_ratio'].clip(0, 2) / 2) * 0.3
    ).clip(0, 1)
    
    # Credit Concentration Risk
    df['credit_concentration_risk'] = (
        df.get('sector_concentration_hhi', 0) * 0.5 +
        df.get('top20_borrower_concentration', df.get('top_borrower_concentration', 0)) * 0.3 +
        df.get('geographic_concentration', 0.5) * 0.2
    ).clip(0, 1)
    
    # Off-Balance Sheet Risk
    df['obs_risk_indicator'] = (
        (df['derivatives_to_assets_ratio'].clip(0, 3) / 3) * 0.5 +
        df['unused_lines_to_loans_ratio'].clip(0, 1) * 0.3 +
        (df['obs_to_assets_ratio'].clip(0, 2) / 2) * 0.2
    ).clip(0, 1)
    
    print(f"   ✓ Added: liquidity_concentration_risk, credit_concentration_risk, obs_risk_indicator")
    
    # ==========================================================================
    # 6. SAVE ENHANCED DATASET
    # ==========================================================================
    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Saving to {output_file}...")
    
    df.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "="*80)
    print("ENHANCEMENT SUMMARY")
    print("="*80)
    
    new_indicators = {
        'Net Interest Margin': ['interest_income', 'interest_expense', 'net_interest_margin'],
        'Off-Balance Sheet': ['derivatives_notional', 'unused_credit_lines', 'guarantees_issued',
                             'derivatives_to_assets_ratio', 'unused_lines_to_loans_ratio', 
                             'obs_to_assets_ratio'],
        'Deposit Concentration': ['top20_depositors', 'top5_depositors', 'top20_depositors_ratio', 
                                 'top5_depositors_ratio'],
        'Loan Concentration': ['sector_concentration_hhi', 'top20_borrower_loans', 
                              'top20_borrower_concentration', 'geographic_concentration'],
        'Composite Risks': ['liquidity_concentration_risk', 'credit_concentration_risk', 
                           'obs_risk_indicator']
    }
    
    for category, indicators in new_indicators.items():
        print(f"\n{category}:")
        for ind in indicators:
            if ind in df.columns:
                print(f"  ✓ {ind}")
    
    print("\n" + "="*80)
    print("✓ Enhancement Complete!")
    print("="*80)
    
    return df


if __name__ == "__main__":
    input_file = "time_series_dataset_enriched.csv"
    output_file = "time_series_dataset_enriched_v2.csv"
    
    df = add_new_indicators_to_dataset(input_file, output_file)
    
    print(f"\nEnhanced dataset: {output_file}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows: {len(df)}")
