# -*- coding: utf-8 -*-
"""
Tập lệnh Tạo Điểm Quý - Tính toán và Xuất Điểm Rủi ro Quý
Tích hợp với Hệ thống Kiểm toán Ngân hàng
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import quarterly calculator
try:
    from importlib.util import spec_from_file_location, module_from_spec
    
    spec = spec_from_file_location("quarterly_scores", "7_quarterly_scores.py")
    quarterly_module = module_from_spec(spec)
    spec.loader.exec_module(quarterly_module)
    QuarterlyScoreCalculator = quarterly_module.QuarterlyScoreCalculator
    calculate_all_quarterly_scores = quarterly_module.calculate_all_quarterly_scores
except Exception as e:
    logger.error(f"Failed to import QuarterlyScoreCalculator: {e}")
    sys.exit(1)


def generate_quarterly_scores_for_all_banks(csv_path: str = "time_series_dataset_enriched_v2.csv",
                                           output_dir: str = "outputs") -> pd.DataFrame:
    """
    Tạo điểm quý cho tất cả các ngân hàng từ dữ liệu chuỗi thời gian
    
    Args:
        csv_path (str): Đường dẫn đến tệp CSV dữ liệu chuỗi thời gian
        output_dir (str): Thư mục để lưu kết quả
        
    Returns:
        pd.DataFrame: DataFrame chứa điểm quý cho tất cả ngân hàng
    """
    logger.info(f"Loading time series data from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        raise FileNotFoundError(f"Cannot find {csv_path}")
    
    # Load time series data
    try:
        time_series_df = pd.read_csv(csv_path)
        logger.info(f"Loaded time series data: {time_series_df.shape[0]} rows, {time_series_df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise
    
    # Verify required columns
    required_cols = ['bank_id', 'period']
    missing_cols = [col for col in required_cols if col not in time_series_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    logger.info("Calculating quarterly scores for all banks...")
    
    # Calculate quarterly scores
    result = calculate_all_quarterly_scores(
        time_series_df,
        credit_model=None,
        liquidity_model=None,
        anomaly_model=None,
        financial_calc=None
    )
    
    quarterly_df = result['quarterly_scores']
    
    if quarterly_df.empty:
        logger.error("No quarterly scores generated")
        return pd.DataFrame()
    
    logger.info(f"Successfully calculated quarterly scores:")
    logger.info(f"  - Total rows: {len(quarterly_df)}")
    logger.info(f"  - Banks processed: {result['num_banks']}")
    logger.info(f"  - Banks: {', '.join(result['banks_processed'])}")
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Save quarterly scores to Excel
    excel_output = os.path.join(output_dir, "quarterly_risk_scores.xlsx")
    try:
        with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
            # Sheet 1: All quarterly scores
            quarterly_df.to_excel(writer, sheet_name='Quarterly Scores', index=False)
            logger.info(f"[+] Sheet 'Quarterly Scores': {len(quarterly_df)} rows")
            
            # Sheet 2: Summary by bank
            summary_by_bank = quarterly_df.groupby('bank_name').agg({
                'quarterly_risk_score': ['mean', 'min', 'max', 'std', 'count'],
                'credit_score': 'mean',
                'liquidity_score': 'mean',
                'anomaly_score': 'mean'
            }).round(2)
            summary_by_bank.to_excel(writer, sheet_name='Bank Summary')
            logger.info(f"[+] Sheet 'Bank Summary': {len(summary_by_bank)} banks")
            
            # Sheet 3: Summary by risk level
            if 'risk_level' in quarterly_df.columns:
                risk_summary = quarterly_df.groupby(['bank_name', 'risk_level']).size().unstack(fill_value=0)
                risk_summary.to_excel(writer, sheet_name='Risk Level Distribution')
                logger.info(f"[+] Sheet 'Risk Level Distribution': Risk level summary by bank")
            
            # Sheet 4: Time series trends (for visualization)
            quarterly_pivot = quarterly_df.pivot_table(
                index='period',
                columns='bank_name',
                values='quarterly_risk_score',
                aggfunc='mean'
            )
            quarterly_pivot.to_excel(writer, sheet_name='Trend Analysis')
            logger.info(f"[+] Sheet 'Trend Analysis': {len(quarterly_pivot)} periods")
        
        logger.info(f"Successfully saved quarterly scores to: {excel_output}")
    except Exception as e:
        logger.error(f"Failed to save Excel file: {e}")
        raise
    
    # Save quarterly scores to CSV
    csv_output = os.path.join(output_dir, "quarterly_risk_scores.csv")
    try:
        quarterly_df.to_csv(csv_output, index=False)
        logger.info(f"Successfully saved quarterly scores to: {csv_output}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        raise
    
    # Display summary
    logger.info("\n" + "="*80)
    logger.info("QUARTERLY RISK SCORES SUMMARY")
    logger.info("="*80)
    
    # Summary statistics
    for bank in quarterly_df['bank_name'].unique():
        bank_data = quarterly_df[quarterly_df['bank_name'] == bank]
        scores = bank_data['quarterly_risk_score'].dropna()
        
        if len(scores) > 0:
            latest = bank_data['quarterly_risk_score'].iloc[-1]
            earliest = bank_data['quarterly_risk_score'].iloc[0]
            trend = latest - earliest
            
            logger.info(f"\n{bank}:")
            logger.info(f"  Quarters analyzed: {len(bank_data)}")
            logger.info(f"  Average Score: {scores.mean():.2f}")
            logger.info(f"  Score Range: {scores.min():.2f} - {scores.max():.2f}")
            logger.info(f"  Latest Score: {latest:.2f}")
            logger.info(f"  Trend (Latest - Earliest): {trend:+.2f}")
            
            # Risk level distribution
            risk_dist = bank_data['risk_level'].value_counts()
            logger.info(f"  Risk Level Distribution:")
            for level, count in risk_dist.items():
                pct = (count / len(bank_data)) * 100
                logger.info(f"    - {level}: {count} quarters ({pct:.1f}%)")
    
    logger.info("\n" + "="*80)
    logger.info(f"Output files created:")
    logger.info(f"  - Excel: {excel_output}")
    logger.info(f"  - CSV: {csv_output}")
    logger.info("="*80)
    
    return quarterly_df


def main():
    """Main entry point"""
    logger.info("Starting Quarterly Scores Generation Script")
    
    try:
        # Generate quarterly scores
        quarterly_df = generate_quarterly_scores_for_all_banks()
        
        if not quarterly_df.empty:
            logger.info("\n[SUCCESS] Quarterly scores generated successfully!")
            logger.info(f"Total quarterly records: {len(quarterly_df)}")
            return 0
        else:
            logger.error("\n[ERROR] No quarterly scores were generated")
            return 1
            
    except Exception as e:
        logger.error(f"\n[ERROR] Failed to generate quarterly scores: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
