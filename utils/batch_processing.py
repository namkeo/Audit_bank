# -*- coding: utf-8 -*-
"""
Mô-đun Xử lý Hàng loạt
Các hoạt động hàng loạt hiệu quả trên nhiều ngân hàng sử dụng NumPy/Pandas được vector hóa
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict


# Tải lười các hàm tiện ích để tránh lỗi nhập khi các đường dẫn mô-đun không được đặt
_FinancialRatioCalculator = None

def _get_financial_ratio_calculator():
    """Tải lười FinancialRatioCalculator khi sử dụng lần đầu tiên."""
    global _FinancialRatioCalculator
    if _FinancialRatioCalculator is None:
        import importlib.util
        try:
            spec = importlib.util.spec_from_file_location("utility_functions", "4_utility_functions.py")
            util_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(util_module)
            _FinancialRatioCalculator = util_module.FinancialRatioCalculator
        except Exception:
            # Fallback: try standard import
            from utility_functions import FinancialRatioCalculator as FRC
            _FinancialRatioCalculator = FRC
    return _FinancialRatioCalculator


class BatchProcessor:
    """
    Handles efficient batch processing of multiple banks' data
    Replaces bank-by-bank loops with vectorized Pandas/NumPy operations
    """

    @staticmethod
    def prepare_all_banks_ratios(
        all_banks_data: pd.DataFrame,
        time_column: str = 'period',
        bank_id_column: str = 'bank_id'
    ) -> pd.DataFrame:
        """
        Calculate ratios for ALL banks at once (vectorized).
        Avoids per-bank iteration; processes entire DataFrame in one go.

        Args:
            all_banks_data: DataFrame with all banks' data
            time_column: Column name for time periods
            bank_id_column: Column name for bank IDs

        Returns:
            DataFrame with ratios indexed by (bank_id, period)
        """
        # Get the calculator class (lazy-loaded)
        FinancialRatioCalculator = _get_financial_ratio_calculator()

        if all_banks_data.empty:
            return pd.DataFrame()

        # Ensure sorted order for groupby efficiency
        all_banks_data = all_banks_data.sort_values([bank_id_column, time_column])

        # Vectorized: apply ratio calculation per row (much faster than bank-by-bank)
        def calc_ratio_row(row):
            try:
                return pd.Series(FinancialRatioCalculator.calculate_ratios(row, data_format='series'))
            except Exception:
                return pd.Series()

        ratios_df = all_banks_data.apply(calc_ratio_row, axis=1)
        ratios_df[bank_id_column] = all_banks_data[bank_id_column].values
        ratios_df[time_column] = all_banks_data[time_column].values

        return ratios_df.reset_index(drop=True)

    @staticmethod
    def aggregate_ratios(
        ratios_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        bank_id_column: str = 'bank_id',
        time_column: str = 'period'
    ) -> pd.DataFrame:
        """
        Aggregate ratios to (bank_id, period) level in one vectorized operation.

        Args:
            ratios_df: DataFrame with per-row ratios
            metrics: List of metric columns to aggregate (if None, use all numeric)
            bank_id_column: Bank ID column name
            time_column: Period column name

        Returns:
            Aggregated DataFrame indexed by (bank_id, period)
        """
        if metrics is None:
            metrics = ratios_df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove ID/period columns from metrics if present
        metrics = [m for m in metrics if m not in (bank_id_column, time_column)]

        # Vectorized groupby + agg (one operation for all banks + periods)
        agg_dict = {m: 'mean' for m in metrics}
        agg_result = (
            ratios_df
            .groupby([bank_id_column, time_column])
            .agg(agg_dict)
            .reset_index()
        )

        return agg_result

    @staticmethod
    def identify_high_risk_banks(
        ratios_df: pd.DataFrame,
        thresholds: Dict[str, Tuple[str, float]],
        bank_id_column: str = 'bank_id'
    ) -> Dict[str, List[str]]:
        """
        Identify high-risk banks using vectorized boolean operations.
        Fast: O(n) instead of O(n_banks × operations_per_bank).

        Args:
            ratios_df: DataFrame with ratios
            thresholds: Dict of {metric: ('>', threshold)} or {'metric': ('<', threshold)}
            bank_id_column: Bank ID column

        Returns:
            Dict of {bank_id: [list of violated metrics]}
        """
        violations = defaultdict(list)

        for metric, (operator, threshold) in thresholds.items():
            if metric not in ratios_df.columns:
                continue

            # Vectorized: apply condition to entire column
            if operator == '>':
                mask = ratios_df[metric] > threshold
            elif operator == '<':
                mask = ratios_df[metric] < threshold
            elif operator == '>=':
                mask = ratios_df[metric] >= threshold
            elif operator == '<=':
                mask = ratios_df[metric] <= threshold
            else:
                continue

            # Get banks where violation occurred
            violated_banks = ratios_df.loc[mask, bank_id_column].unique()
            for bank_id in violated_banks:
                violations[bank_id].append(metric)

        return dict(violations)

    @staticmethod
    def compute_bank_level_statistics(
        all_banks_data: pd.DataFrame,
        metrics: List[str],
        bank_id_column: str = 'bank_id',
        statistics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute statistics (mean, std, min, max) for all banks at once.
        Vectorized groupby + agg instead of per-bank loops.

        Args:
            all_banks_data: DataFrame with all banks' data
            metrics: List of metric columns
            bank_id_column: Bank ID column
            statistics: List of statistics to compute (default: ['mean', 'std', 'min', 'max'])

        Returns:
            DataFrame with bank-level statistics
        """
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max']

        # Filter metrics that exist
        metrics = [m for m in metrics if m in all_banks_data.columns]

        # Vectorized groupby + agg (single operation for all banks)
        agg_dict = {m: statistics for m in metrics}
        stats_df = all_banks_data.groupby(bank_id_column).agg(agg_dict)

        # Flatten multi-level columns
        stats_df.columns = [f"{metric}_{stat}" for metric, stat in stats_df.columns]
        return stats_df.reset_index()

    @staticmethod
    def parallel_filter_by_period(
        all_banks_data: pd.DataFrame,
        period: str,
        time_column: str = 'period'
    ) -> pd.DataFrame:
        """
        Filter all banks' data for a specific period in one operation.
        Vectorized boolean indexing (O(n) instead of looping).

        Args:
            all_banks_data: DataFrame with all banks
            period: Period to filter
            time_column: Column name for periods

        Returns:
            Filtered DataFrame for the period
        """
        return all_banks_data[all_banks_data[time_column] == period].copy()

    @staticmethod
    def batch_ratio_comparison(
        all_banks_data: pd.DataFrame,
        metric: str,
        bank_id_column: str = 'bank_id',
        time_column: str = 'period'
    ) -> pd.DataFrame:
        """
        Compare a metric across all banks for all periods (pivoted for easy comparison).

        Args:
            all_banks_data: DataFrame with data
            metric: Metric to compare
            bank_id_column: Bank ID column
            time_column: Period column

        Returns:
            Pivoted DataFrame: rows=banks, cols=periods, values=metric
        """
        if metric not in all_banks_data.columns:
            return pd.DataFrame()

        # Vectorized pivot (one operation instead of per-bank loops)
        pivot_df = all_banks_data.pivot_table(
            index=bank_id_column,
            columns=time_column,
            values=metric,
            aggfunc='mean'
        )

        return pivot_df

    @staticmethod
    def compute_growth_rates(
        all_banks_data: pd.DataFrame,
        metric: str,
        bank_id_column: str = 'bank_id',
        time_column: str = 'period'
    ) -> pd.DataFrame:
        """
        Compute period-over-period growth rates for all banks at once.

        Args:
            all_banks_data: DataFrame with data
            metric: Metric to compute growth for
            bank_id_column: Bank ID column
            time_column: Period column

        Returns:
            DataFrame with bank IDs and growth rates
        """
        if metric not in all_banks_data.columns:
            return pd.DataFrame()

        # Sort by bank and period for correct groupby
        sorted_data = all_banks_data.sort_values([bank_id_column, time_column])

        # Vectorized: compute growth within each bank group
        growth_df = sorted_data.groupby(bank_id_column)[metric].pct_change()

        result = pd.DataFrame({
            bank_id_column: sorted_data[bank_id_column].values,
            time_column: sorted_data[time_column].values,
            f'{metric}_growth_rate': growth_df.values
        })

        return result.dropna()

    @staticmethod
    def batch_outlier_detection(
        all_banks_data: pd.DataFrame,
        metrics: List[str],
        z_score_threshold: float = 3.0,
        bank_id_column: str = 'bank_id'
    ) -> Dict[str, List[int]]:
        """
        Detect outlier banks (rows with extreme values) across all metrics at once.
        Vectorized computation instead of per-bank loops.

        Args:
            all_banks_data: DataFrame with data
            metrics: List of metrics to check
            z_score_threshold: Z-score threshold for outlier detection
            bank_id_column: Bank ID column

        Returns:
            Dict of {bank_id: [list of outlier metric indices]}
        """
        outliers = defaultdict(list)
        metrics = [m for m in metrics if m in all_banks_data.columns]

        # Vectorized: compute z-scores for each metric
        for metric in metrics:
            data = all_banks_data[metric].values
            mean = np.nanmean(data)
            std = np.nanstd(data)

            if std == 0:
                continue

            z_scores = np.abs((data - mean) / std)
            outlier_indices = np.where(z_scores > z_score_threshold)[0]

            for idx in outlier_indices:
                bank_id = all_banks_data.iloc[idx][bank_id_column]
                outliers[bank_id].append(metric)

        return dict(outliers)

    @staticmethod
    def rank_banks_by_metric(
        all_banks_data: pd.DataFrame,
        metric: str,
        bank_id_column: str = 'bank_id',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank all banks by a metric in one vectorized operation.

        Args:
            all_banks_data: DataFrame with data
            metric: Metric to rank by
            bank_id_column: Bank ID column
            ascending: If False, highest value gets rank 1

        Returns:
            DataFrame with bank IDs and ranks
        """
        if metric not in all_banks_data.columns:
            return pd.DataFrame()

        result = (
            all_banks_data[[bank_id_column, metric]]
            .drop_duplicates(subset=[bank_id_column])
            .sort_values(metric, ascending=ascending)
            .reset_index(drop=True)
        )

        result['rank'] = range(1, len(result) + 1)
        return result[[bank_id_column, metric, 'rank']]


class BatchAuditRunner:
    """
    High-level wrapper for running audits on multiple banks in batch mode.
    Handles pre/post-processing and result aggregation efficiently.
    """

    def __init__(self, audit_system_class):
        """
        Args:
            audit_system_class: The BankAuditSystem class to use for audits
        """
        self.audit_system_class = audit_system_class
        self.batch_processor = BatchProcessor()

    def run_batch_audits(
        self,
        df: pd.DataFrame,
        bank_ids: Optional[List[str]] = None,
        bank_id_column: str = 'bank_id',
        **audit_kwargs
    ) -> Dict[str, Dict]:
        """
        Run audits for multiple banks efficiently.

        Args:
            df: Full dataset with all banks
            bank_ids: List of bank IDs to audit (if None, audit all)
            bank_id_column: Column name for bank IDs
            **audit_kwargs: Additional kwargs to pass to BankAuditSystem.run_complete_audit

        Returns:
            Dict of {bank_id: audit_report}
        """
        if bank_ids is None:
            bank_ids = df[bank_id_column].unique().tolist()

        all_reports = {}
        audit_period = audit_kwargs.pop('audit_period', '2024')

        for bank_id in bank_ids:
            bank_df = df[df[bank_id_column] == bank_id]

            audit_system = self.audit_system_class(
                bank_name=bank_id,
                audit_period=audit_period
            )

            report = audit_system.run_complete_audit(
                df=bank_df,
                bank_id=bank_id,
                all_banks_data=df,
                **audit_kwargs
            )

            all_reports[bank_id] = report

        return all_reports

    def aggregate_batch_results(
        self,
        all_reports: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Aggregate audit results across all banks into a summary DataFrame.

        Args:
            all_reports: Dict of audit reports from run_batch_audits

        Returns:
            Summary DataFrame with bank-level metrics
        """
        summary_data = []

        for bank_id, report in all_reports.items():
            overall = report.get('overall_risk_score', {})
            summary_data.append({
                'bank_id': bank_id,
                'overall_score': overall.get('overall_score', np.nan),
                'risk_level': overall.get('risk_level', 'UNKNOWN'),
                'credit_risk_score': report.get('credit_risk', {}).get('score', np.nan),
                'liquidity_risk_score': report.get('liquidity_risk', {}).get('score', np.nan),
                'anomaly_count': len(report.get('anomalies_detected', [])),
            })

        return pd.DataFrame(summary_data)
