# -*- coding: utf-8 -*-
"""
Mô-đun Điều chỉnh Vĩ mô
Bối cảnh hóa điểm rủi ro của từng ngân hàng so với các chuẩn mực ngành.
Chiếm đến rủi ro hệ thống vs. không hệ thống.

Khái niệm chính:
- RỦI RO TUYỆT ĐỐI: Chỉ số của ngân hàng so với ngưỡng cố định (ví dụ: NPL > 5%)
- RỦI RO TƯƠNG ĐỐI: Chỉ số của ngân hàng so với trung bình ngành (ví dụ: NPL vs. trung bình ngành)
- RỦI RO HỆ THỐNG: Khi toàn bộ ngành đang gặp căng thẳng (ví dụ: suy thoái, lãi suất tăng)
- ĐIỀU CHỈNH: Điểm rủi ro được sửa đổi dựa trên độ lệch so với trung bình ngành

Ví dụ:
  - NPL của ngân hàng = 4%, Trung bình ngành = 1% → Rủi ro tương đối cao (3% trên trung bình)
  - NPL của ngân hàng = 4%, Trung bình ngành = 3% → Rủi ro tương đối vừa phải (1% trên trung bình)
  - Cả hai đều không vượt qua ngưỡng tuyệt đối, nhưng bối cảnh tương đối khác
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging


class MacroAdjustmentCalculator:
    """
    Tính toán các điều chỉnh cấp vĩ mô cho điểm rủi ro dựa trên bối cảnh ngành.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_industry_benchmarks(self, 
                                     all_banks_data: pd.DataFrame,
                                     metrics: List[str],
                                     period: Optional[str] = None,
                                     bank_id_column: str = 'bank_id',
                                     time_column: str = 'period') -> Dict[str, Dict]:
        """
        Calculate industry benchmarks (mean, std, percentiles) for metrics.
        
        Args:
            all_banks_data: DataFrame with all banks' data
            metrics: List of metric columns to benchmark
            period: Optional specific period (if None, uses all periods)
            bank_id_column: Column name for bank IDs
            time_column: Column name for time periods
            
        Returns:
            Dictionary {metric_name: {mean, std, p25, p50, p75, etc.}}
        """
        if all_banks_data.empty:
            self.logger.warning("Empty dataframe provided for benchmarking")
            return {}
        
        # Filter by period if specified
        data = all_banks_data
        if period is not None:
            data = all_banks_data[all_banks_data[time_column] == period]
            if data.empty:
                self.logger.warning(f"No data found for period {period}")
                return {}
        
        benchmarks = {}
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            values = data[metric].dropna()
            if len(values) > 0:
                benchmarks[metric] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'p25': float(values.quantile(0.25)),
                    'p50': float(values.quantile(0.50)),
                    'p75': float(values.quantile(0.75)),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                }
        
        return benchmarks
    
    def calculate_relative_deviation(self,
                                    bank_value: float,
                                    industry_mean: float,
                                    industry_std: float) -> float:
        """
        Calculate deviation as z-score (# of std devs from mean).
        
        Args:
            bank_value: Bank's metric value
            industry_mean: Industry average
            industry_std: Industry standard deviation
            
        Returns:
            Z-score (negative = better than average, positive = worse)
        """
        if industry_std == 0:
            return 0.0
        return (bank_value - industry_mean) / industry_std
    
    def calculate_percentile_rank(self,
                                 bank_value: float,
                                 all_values: np.ndarray) -> float:
        """
        Calculate bank's percentile rank within the industry.
        Higher percentile = higher metric value = worse for most risk metrics.
        
        Args:
            bank_value: Bank's metric value
            all_values: Array of all industry values
            
        Returns:
            Percentile (0-100), e.g., 75 means bank is in top 25% worst
        """
        return float(np.mean(all_values <= bank_value) * 100)
    
    def adjust_risk_score(self,
                         absolute_score: float,
                         relative_z_score: float,
                         systemic_stress_level: float = 0.0,
                         adjustment_strength: float = 0.3) -> Dict:
        """
        Adjust risk score based on relative industry context.
        
        Args:
            absolute_score: Original risk score (0-100 typically)
            relative_z_score: Deviation from industry average (z-score)
            systemic_stress_level: Industry stress (0=normal, 1=severe crisis)
            adjustment_strength: How much to weight relative context (0-1)
                                0 = absolute only, 1 = relative heavily weighted
            
        Returns:
            Dictionary with:
              - adjusted_score: Modified risk score
              - adjustment_delta: Change from absolute score
              - adjustment_reason: Human-readable explanation
              - adjustment_confidence: How confident in adjustment (0-1)
        """
        
        # Clamp inputs
        absolute_score = max(0, min(100, absolute_score))
        relative_z_score = max(-5, min(5, relative_z_score))
        systemic_stress_level = max(0, min(1, systemic_stress_level))
        adjustment_strength = max(0, min(1, adjustment_strength))
        
        # Base adjustment: z-score contribution
        # Positive z-score (worse than average) → positive adjustment
        z_score_adjustment = relative_z_score * 10 * adjustment_strength
        
        # Systemic stress mitigation: reduce adjustment if industry-wide stress
        # (everyone's suffering, so individual bank isn't necessarily riskier)
        stress_mitigation = systemic_stress_level * z_score_adjustment * 0.5
        
        # Total adjustment
        total_adjustment = z_score_adjustment - stress_mitigation
        
        # Adjusted score (with bounds)
        adjusted_score = absolute_score + total_adjustment
        adjusted_score = max(0, min(100, adjusted_score))
        
        # Explanation
        if abs(relative_z_score) < 0.5:
            context = "close to industry average"
        elif relative_z_score > 0:
            context = f"worse than {abs(relative_z_score):.1f}σ below average"
        else:
            context = f"better than {abs(relative_z_score):.1f}σ above average"
        
        explanation = (
            f"Absolute score: {absolute_score:.1f}. "
            f"Bank is {context}. "
        )
        
        if systemic_stress_level > 0:
            explanation += f"Industry stress level {systemic_stress_level:.1%} reduces relative concern. "
        
        explanation += f"Adjusted score: {adjusted_score:.1f}"
        
        # Confidence: higher when deviation is clear and stress is low
        confidence = 1.0 - (systemic_stress_level * 0.3)  # Stress reduces confidence
        if abs(relative_z_score) < 0.5:
            confidence *= 0.6  # Low confidence when near average
        
        return {
            'adjusted_score': float(adjusted_score),
            'adjustment_delta': float(total_adjustment),
            'absolute_score': float(absolute_score),
            'relative_z_score': float(relative_z_score),
            'systemic_stress_level': float(systemic_stress_level),
            'adjustment_reason': explanation,
            'adjustment_confidence': float(confidence)
        }
    
    def adjust_multiple_indicators(self,
                                  indicator_scores: Dict[str, float],
                                  benchmarks: Dict[str, Dict],
                                  bank_values: Dict[str, float],
                                  systemic_stress_level: float = 0.0) -> Dict[str, Dict]:
        """
        Adjust multiple risk indicators at once.
        
        Args:
            indicator_scores: {indicator_name: absolute_score}
            benchmarks: {metric_name: {mean, std, ...}}
            bank_values: {metric_name: bank_value}
            systemic_stress_level: Industry stress (0-1)
            
        Returns:
            Dictionary of adjusted indicators with explanations
        """
        adjusted = {}
        
        for indicator, absolute_score in indicator_scores.items():
            if indicator not in benchmarks or indicator not in bank_values:
                # If we can't adjust, keep original score
                adjusted[indicator] = {
                    'absolute_score': float(absolute_score),
                    'adjusted_score': float(absolute_score),
                    'adjustment_delta': 0.0,
                    'relative_z_score': None,
                    'systemic_stress_level': systemic_stress_level,
                    'adjustment_reason': "No benchmark data available",
                    'adjustment_confidence': 0.0
                }
                continue
            
            benchmark = benchmarks[indicator]
            bank_value = bank_values[indicator]
            
            # Calculate z-score
            z_score = self.calculate_relative_deviation(
                bank_value,
                benchmark['mean'],
                benchmark['std']
            )
            
            # Adjust
            adjustment = self.adjust_risk_score(
                absolute_score,
                z_score,
                systemic_stress_level,
                adjustment_strength=0.35
            )
            
            adjusted[indicator] = adjustment
        
        return adjusted
    
    def estimate_systemic_stress(self,
                                all_banks_data: pd.DataFrame,
                                stress_metrics: List[str],
                                thresholds: Dict[str, Tuple[str, float]],
                                period: Optional[str] = None) -> float:
        """
        Estimate industry-wide stress level based on % of banks in distress.
        
        Args:
            all_banks_data: DataFrame with all banks
            stress_metrics: Metrics indicating distress (e.g., ['npl_ratio', 'liquidity_ratio'])
            thresholds: {metric: (operator, threshold)} e.g., {'npl_ratio': ('>', 0.05)}
            period: Optional specific period
            
        Returns:
            Stress level (0=normal, 1=severe) based on % banks failing thresholds
        """
        if all_banks_data.empty:
            return 0.0
        
        data = all_banks_data
        if period is not None:
            data = all_banks_data[all_banks_data.get('period') == period]
            if data.empty:
                return 0.0
        
        # Count banks failing thresholds
        failures = 0
        total = 0
        
        for metric, (op, threshold) in thresholds.items():
            if metric not in data.columns:
                continue
            
            values = data[metric].dropna()
            if len(values) == 0:
                continue
            
            total += len(values)
            if op == '>':
                failures += (values > threshold).sum()
            elif op == '<':
                failures += (values < threshold).sum()
            elif op == '>=':
                failures += (values >= threshold).sum()
            elif op == '<=':
                failures += (values <= threshold).sum()
        
        if total == 0:
            return 0.0
        
        # Stress level = % of banks in distress
        # 0-10% = no stress, 50%+ = high stress
        stress_ratio = failures / total
        
        # Map to 0-1 scale with sigmoid curve
        # At 50% failure rate, stress = 0.5; at 80%, stress = 0.9
        stress_level = 2 / (1 + np.exp(-10 * (stress_ratio - 0.5)))  # Sigmoid
        stress_level = max(0, min(1, stress_level))
        
        return float(stress_level)
    
    def generate_adjustment_report(self,
                                  bank_id: str,
                                  adjusted_indicators: Dict[str, Dict],
                                  systemic_stress_level: float) -> str:
        """
        Generate human-readable report of adjustments.
        
        Args:
            bank_id: Bank identifier
            adjusted_indicators: Output from adjust_multiple_indicators
            systemic_stress_level: Industry stress (0-1)
            
        Returns:
            Formatted text report
        """
        report = f"\n{'='*70}\n"
        report += f"MACRO-ADJUSTED RISK REPORT - {bank_id}\n"
        report += f"{'='*70}\n\n"
        
        report += f"INDUSTRY CONTEXT:\n"
        report += f"  Systemic Stress Level: {systemic_stress_level:.1%}\n"
        if systemic_stress_level < 0.2:
            report += f"  Status: Normal market conditions\n"
        elif systemic_stress_level < 0.5:
            report += f"  Status: Moderate industry stress\n"
        else:
            report += f"  Status: HIGH industry stress - adjustments applied\n"
        report += "\n"
        
        report += f"INDICATOR ADJUSTMENTS:\n"
        report += f"{'-'*70}\n"
        
        for indicator, adjustment in adjusted_indicators.items():
            report += f"\n{indicator.upper()}:\n"
            report += f"  Absolute Score: {adjustment['absolute_score']:.1f}\n"
            report += f"  Adjusted Score: {adjustment['adjusted_score']:.1f}\n"
            
            if adjustment['adjustment_delta'] != 0:
                delta_str = f"{'↑' if adjustment['adjustment_delta'] > 0 else '↓'} {abs(adjustment['adjustment_delta']):.1f}"
                report += f"  Adjustment: {delta_str}\n"
            
            if adjustment['relative_z_score'] is not None:
                report += f"  Relative Position: {adjustment['relative_z_score']:.2f}σ from industry avg\n"
            
            report += f"  Confidence: {adjustment['adjustment_confidence']:.1%}\n"
            report += f"  Reason: {adjustment['adjustment_reason']}\n"
        
        report += f"\n{'='*70}\n"
        
        return report


def apply_macro_adjustments_to_bank_audit(
    audit_report: Dict,
    all_banks_data: pd.DataFrame,
    indicators_to_adjust: List[str] = None
) -> Dict:
    """
    Convenience function: apply macro adjustments to an existing audit report.
    
    Args:
        audit_report: Bank audit report dict
        all_banks_data: DataFrame with all banks for benchmarking
        indicators_to_adjust: List of indicators to adjust (if None, adjust all)
        
    Returns:
        Audit report with added 'macro_adjustments' section
    """
    if indicators_to_adjust is None:
        indicators_to_adjust = ['credit_risk', 'liquidity_risk', 'anomaly_risk']
    
    calc = MacroAdjustmentCalculator()
    
    # Calculate benchmarks
    stress_metrics = ['npl_ratio', 'liquidity_ratio']
    thresholds = {
        'npl_ratio': ('>', 0.05),
        'liquidity_ratio': ('<', 0.3)
    }
    
    benchmarks = calc.calculate_industry_benchmarks(
        all_banks_data,
        stress_metrics
    )
    
    # Estimate systemic stress
    systemic_stress = calc.estimate_systemic_stress(
        all_banks_data,
        stress_metrics,
        thresholds
    )
    
    # Extract bank's metric values
    bank_data = audit_report.get('bank_data', {})
    
    # Build score adjustments
    score_adjustments = {}
    for indicator in indicators_to_adjust:
        if indicator in audit_report:
            score = audit_report[indicator].get('score', 50)
            score_adjustments[indicator] = score
    
    # Apply adjustments
    adjusted = calc.adjust_multiple_indicators(
        score_adjustments,
        benchmarks,
        bank_data,
        systemic_stress
    )
    
    # Add to report
    audit_report['macro_adjustments'] = {
        'systemic_stress_level': systemic_stress,
        'adjusted_indicators': adjusted,
        'benchmarks': benchmarks
    }
    
    return audit_report
