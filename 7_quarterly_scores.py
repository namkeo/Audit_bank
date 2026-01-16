# -*- coding: utf-8 -*-
"""
Hệ thống Tính toán Điểm số Theo Quý
Tính toán điểm rủi ro cho từng kỳ hạn độc lập (20 quý cho mỗi ngân hàng)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Set up logger
logger = logging.getLogger(__name__)


class QuarterlyScoreCalculator:
    """
    Tính toán điểm rủi ro cho từng quý riêng biệt
    Cho phép theo dõi sự phát triển của rủi ro theo thời gian
    
    Mỗi quý được xử lý độc lập:
    - Tính toán các chỉ số tài chính cho quý
    - Đánh giá rủi ro tín dụng, thanh khoản, bất thường
    - Tổng hợp thành điểm rủi ro quý
    """
    
    def __init__(self, credit_risk_model=None, liquidity_risk_model=None, 
                 anomaly_model=None, financial_calculator=None):
        """
        Khởi tạo máy tính điểm rủi ro theo quý
        
        Args:
            credit_risk_model: Mô hình rủi ro tín dụng đã được huấn luyện
            liquidity_risk_model: Mô hình rủi ro thanh khoản đã được huấn luyện
            anomaly_model: Mô hình phát hiện bất thường đã được huấn luyện
            financial_calculator: Máy tính tỷ số tài chính
        """
        self.credit_risk_model = credit_risk_model
        self.liquidity_risk_model = liquidity_risk_model
        self.anomaly_model = anomaly_model
        self.financial_calculator = financial_calculator
        
        logger.info("QuarterlyScoreCalculator initialized")
    
    def calculate_quarterly_scores(self, time_series_data: pd.DataFrame, 
                                   bank_name: str) -> Dict:
        """
        Tính toán điểm rủi ro cho từng quý độc lập
        
        Args:
            time_series_data (pd.DataFrame): Dữ liệu chuỗi thời gian (20 quý)
                - Phải có cột 'period' để xác định quý
                - Phải có tất cả các cột tài chính cần thiết
            bank_name (str): Tên của ngân hàng
            
        Returns:
            Dict: Chứa:
                - quarterly_scores (DataFrame): Điểm số từng quý
                - quarterly_summary (DataFrame): Tóm tắt thống kê
                - bank_name (str): Tên ngân hàng
                - num_quarters (int): Số lượng quý được tính
                - periods (List): Danh sách các kỳ hạn
        """
        logger.info(f"Calculating quarterly scores for {bank_name}")
        
        if time_series_data.empty:
            logger.warning(f"Empty time series data for {bank_name}")
            return self._empty_result(bank_name)
        
        # Get unique periods (should be 20 for quarterly data)
        periods = sorted(time_series_data['period'].unique())
        num_quarters = len(periods)
        
        logger.info(f"Processing {num_quarters} quarters for {bank_name}: {periods}")
        
        quarterly_scores = []
        
        # Calculate score for each quarter independently
        for period in periods:
            quarter_data = time_series_data[time_series_data['period'] == period].copy()
            
            if quarter_data.empty:
                logger.warning(f"No data for period {period}")
                continue
            
            # Get single row for this quarter
            row = quarter_data.iloc[0]
            
            # Calculate quarterly score components
            quarter_score = {
                'period': period,
                'bank_name': bank_name
            }
            
            # Credit Risk Score for this quarter
            try:
                credit_score = self._calculate_credit_score_for_quarter(row)
                quarter_score['credit_score'] = credit_score
            except Exception as e:
                logger.warning(f"Error calculating credit score for {bank_name} Q{period}: {e}")
                quarter_score['credit_score'] = np.nan
            
            # Liquidity Risk Score for this quarter
            try:
                liquidity_score = self._calculate_liquidity_score_for_quarter(row)
                quarter_score['liquidity_score'] = liquidity_score
            except Exception as e:
                logger.warning(f"Error calculating liquidity score for {bank_name} Q{period}: {e}")
                quarter_score['liquidity_score'] = np.nan
            
            # Anomaly Score for this quarter
            try:
                anomaly_score = self._calculate_anomaly_score_for_quarter(row)
                quarter_score['anomaly_score'] = anomaly_score
            except Exception as e:
                logger.warning(f"Error calculating anomaly score for {bank_name} Q{period}: {e}")
                quarter_score['anomaly_score'] = np.nan
            
            # Calculate composite quarterly score
            try:
                quarterly_overall = self._calculate_composite_score(
                    credit_score=quarter_score.get('credit_score'),
                    liquidity_score=quarter_score.get('liquidity_score'),
                    anomaly_score=quarter_score.get('anomaly_score')
                )
                quarter_score['quarterly_risk_score'] = quarterly_overall
                quarter_score['risk_level'] = self._classify_risk_level(quarterly_overall)
            except Exception as e:
                logger.warning(f"Error calculating composite score for {bank_name} Q{period}: {e}")
                quarter_score['quarterly_risk_score'] = np.nan
                quarter_score['risk_level'] = 'UNKNOWN'
            
            quarterly_scores.append(quarter_score)
        
        # Convert to DataFrame
        quarterly_df = pd.DataFrame(quarterly_scores)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(quarterly_df, bank_name)
        
        return {
            'quarterly_scores': quarterly_df,
            'quarterly_summary': summary,
            'bank_name': bank_name,
            'num_quarters': len(quarterly_df),
            'periods': periods
        }
    
    def _calculate_credit_score_for_quarter(self, row: pd.Series) -> float:
        """
        Tính điểm rủi ro tín dụng cho một quý riêng biệt
        Dựa trên các chỉ số tín dụng chính: NPL, CAR, ROA
        
        Args:
            row (pd.Series): Dòng dữ liệu cho quý này
            
        Returns:
            float: Điểm rủi ro tín dụng (0-100)
        """
        # Extract key credit metrics
        npl_ratio = row.get('npl_ratio', np.nan)  # 0-100% scale
        car = row.get('capital_adequacy_ratio', np.nan)  # 0-100% scale
        roa = row.get('return_on_assets', np.nan)  # -100% to 100% scale
        
        score = 50.0  # Base neutral score
        
        # NPL penalty: Higher NPL = Higher risk
        if pd.notna(npl_ratio):
            if npl_ratio < 1:
                score -= 10
            elif npl_ratio < 3:
                score -= 5
            elif npl_ratio >= 5:
                score += 15
        
        # CAR reward: Higher CAR = Lower risk
        if pd.notna(car):
            if car >= 10.5:  # Regulatory minimum
                score -= 10
            elif car >= 8:
                score -= 5
            elif car < 5:
                score += 20
        
        # ROA indicator: Profitability reflects credit quality
        if pd.notna(roa):
            if roa < -2:
                score += 15
            elif roa < 0:
                score += 8
            elif roa > 2:
                score -= 8
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    def _calculate_liquidity_score_for_quarter(self, row: pd.Series) -> float:
        """
        Tính điểm rủi ro thanh khoản cho một quý riêng biệt
        Dựa trên các chỉ số thanh khoản chính: LCR, LTD, NSFR
        
        Args:
            row (pd.Series): Dòng dữ liệu cho quý này
            
        Returns:
            float: Điểm rủi ro thanh khoản (0-100)
        """
        score = 50.0  # Base neutral score
        
        # Liquidity Coverage Ratio (LCR)
        lcr = row.get('liquidity_coverage_ratio', np.nan)
        if pd.notna(lcr):
            if lcr >= 1.5:
                score -= 15
            elif lcr >= 1.0:
                score -= 8
            elif lcr < 0.5:
                score += 20
        
        # Loan-to-Deposit ratio
        ltd = row.get('loan_to_deposit_ratio', np.nan)
        if pd.notna(ltd):
            if ltd < 0.7:
                score -= 10
            elif ltd > 1.0:
                score += 15
            elif ltd > 0.95:
                score += 8
        
        # Net Stable Funding Ratio
        nsfr = row.get('nsfr', np.nan)
        if pd.notna(nsfr):
            if nsfr >= 1.2:
                score -= 10
            elif nsfr < 1.0:
                score += 15
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    def _calculate_anomaly_score_for_quarter(self, row: pd.Series) -> float:
        """
        Tính điểm bất thường cho một quý riêng biệt
        Dựa trên các chỉ số bất thường: OBS, derivatives exposure, concentration risk
        
        Args:
            row (pd.Series): Dòng dữ liệu cho quý này
            
        Returns:
            float: Điểm bất thường (0-100)
        """
        score = 30.0  # Base lower score (low anomaly by default)
        
        # OBS to Assets ratio
        obs_ratio = row.get('obs_to_assets_ratio', np.nan)
        if pd.notna(obs_ratio):
            if obs_ratio > 0.5:
                score += 20
            elif obs_ratio > 0.3:
                score += 10
        
        # Derivatives exposure
        deriv_ratio = row.get('derivatives_to_assets_ratio', np.nan)
        if pd.notna(deriv_ratio):
            if deriv_ratio > 0.5:
                score += 15
            elif deriv_ratio > 0.2:
                score += 8
        
        # Concentration risk
        sector_hhi = row.get('sector_concentration_hhi', np.nan)
        if pd.notna(sector_hhi):
            if sector_hhi > 0.4:
                score += 15
            elif sector_hhi > 0.3:
                score += 8
        
        # Geographic concentration
        geo_concentration = row.get('geographic_concentration', np.nan)
        if pd.notna(geo_concentration):
            if geo_concentration > 0.8:
                score += 10
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    def _calculate_composite_score(self, credit_score: float, 
                                   liquidity_score: float, 
                                   anomaly_score: float) -> float:
        """
        Tính điểm tổng hợp cho quý từ các thành phần
        Sử dụng trọng số: Credit 40%, Liquidity 35%, Anomaly 25%
        
        Args:
            credit_score (float): Điểm tín dụng (0-100)
            liquidity_score (float): Điểm thanh khoản (0-100)
            anomaly_score (float): Điểm bất thường (0-100)
            
        Returns:
            float: Điểm tổng hợp (0-100)
        """
        scores = []
        weights = []
        
        if pd.notna(credit_score):
            scores.append(credit_score)
            weights.append(0.40)
        
        if pd.notna(liquidity_score):
            scores.append(liquidity_score)
            weights.append(0.35)
        
        if pd.notna(anomaly_score):
            scores.append(anomaly_score)
            weights.append(0.25)
        
        if not scores:
            return 50.0
        
        total_weight = sum(weights)
        composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return min(100.0, max(0.0, composite))
    
    def _classify_risk_level(self, score: float) -> str:
        """
        Phân loại mức rủi ro dựa trên điểm
        
        Args:
            score (float): Điểm rủi ro (0-100)
            
        Returns:
            str: Mức rủi ro (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL)
        """
        if score < 20:
            return 'MINIMAL'
        elif score < 40:
            return 'LOW'
        elif score < 60:
            return 'MEDIUM'
        elif score < 80:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _generate_summary_statistics(self, quarterly_df: pd.DataFrame, 
                                     bank_name: str) -> pd.DataFrame:
        """
        Tạo thống kê tóm tắt cho các điểm quý
        
        Args:
            quarterly_df (pd.DataFrame): DataFrame chứa điểm quý
            bank_name (str): Tên ngân hàng
            
        Returns:
            pd.DataFrame: Thống kê tóm tắt
        """
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        # Overall quarterly score statistics
        score_col = 'quarterly_risk_score'
        if score_col in quarterly_df.columns:
            scores = quarterly_df[score_col].dropna()
            
            summary_data['Metric'].extend([
                'Average Quarterly Score',
                'Min Quarterly Score',
                'Max Quarterly Score',
                'Std Dev Quarterly Score',
                'Latest Quarterly Score',
                'Trend (Latest - Oldest)'
            ])
            
            latest_score = scores.iloc[-1] if len(scores) > 0 else np.nan
            oldest_score = scores.iloc[0] if len(scores) > 0 else np.nan
            trend = latest_score - oldest_score if pd.notna(latest_score) and pd.notna(oldest_score) else np.nan
            
            summary_data['Value'].extend([
                f"{scores.mean():.2f}" if len(scores) > 0 else "N/A",
                f"{scores.min():.2f}" if len(scores) > 0 else "N/A",
                f"{scores.max():.2f}" if len(scores) > 0 else "N/A",
                f"{scores.std():.2f}" if len(scores) > 1 else "N/A",
                f"{latest_score:.2f}" if pd.notna(latest_score) else "N/A",
                f"{trend:.2f}" if pd.notna(trend) else "N/A"
            ])
        
        # Risk level distribution
        if 'risk_level' in quarterly_df.columns:
            risk_counts = quarterly_df['risk_level'].value_counts()
            for level, count in risk_counts.items():
                summary_data['Metric'].append(f"Quarters with {level} Risk")
                summary_data['Value'].append(str(count))
        
        return pd.DataFrame(summary_data)
    
    def _empty_result(self, bank_name: str) -> Dict:
        """
        Tạo kết quả trống khi không có dữ liệu
        
        Args:
            bank_name (str): Tên ngân hàng
            
        Returns:
            Dict: Kết quả trống với cấu trúc đúng
        """
        return {
            'quarterly_scores': pd.DataFrame(),
            'quarterly_summary': pd.DataFrame(),
            'bank_name': bank_name,
            'num_quarters': 0,
            'periods': []
        }


def calculate_all_quarterly_scores(time_series_data: pd.DataFrame,
                                    credit_model=None,
                                    liquidity_model=None,
                                    anomaly_model=None,
                                    financial_calc=None) -> Dict:
    """
    Tính toán điểm rủi ro theo quý cho tất cả các ngân hàng trong dữ liệu
    
    Args:
        time_series_data (pd.DataFrame): Dữ liệu chuỗi thời gian cho tất cả ngân hàng
        credit_model: Mô hình rủi ro tín dụng
        liquidity_model: Mô hình rủi ro thanh khoản
        anomaly_model: Mô hình phát hiện bất thường
        financial_calc: Máy tính tỷ số tài chính
        
    Returns:
        Dict: Chứa quarterly scores cho tất cả ngân hàng
    """
    logger.info("Calculating quarterly scores for all banks")
    
    calculator = QuarterlyScoreCalculator(
        credit_risk_model=credit_model,
        liquidity_risk_model=liquidity_model,
        anomaly_model=anomaly_model,
        financial_calculator=financial_calc
    )
    
    all_quarterly_scores = []
    all_summaries = []
    banks_processed = []
    
    # Get unique banks
    banks = time_series_data['bank_id'].unique()
    
    for bank_id in banks:
        logger.info(f"Processing bank: {bank_id}")
        
        bank_data = time_series_data[time_series_data['bank_id'] == bank_id].copy()
        
        result = calculator.calculate_quarterly_scores(bank_data, bank_id)
        
        if not result['quarterly_scores'].empty:
            all_quarterly_scores.append(result['quarterly_scores'])
            all_summaries.append(result['quarterly_summary'])
            banks_processed.append(bank_id)
            logger.info(f"Successfully calculated quarterly scores for {bank_id}: {result['num_quarters']} quarters")
        else:
            logger.warning(f"No quarterly scores for {bank_id}")
    
    # Combine all quarterly scores into single DataFrame
    if all_quarterly_scores:
        combined_scores = pd.concat(all_quarterly_scores, ignore_index=True)
    else:
        combined_scores = pd.DataFrame()
    
    return {
        'quarterly_scores': combined_scores,
        'banks_processed': banks_processed,
        'num_banks': len(banks_processed),
        'total_quarters_calculated': len(combined_scores)
    }
