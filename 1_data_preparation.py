# -*- coding: utf-8 -*-
"""
Mô-đun Chuẩn bị Dữ liệu
Xử lý tải dữ liệu, làm sạch và kỹ thuật tính năng
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

# Nhập các hàm tiện ích đã hợp nhất
import importlib.util

# Hàm trợ giúp để nhập mô-đun có tiền tố số
def _import_module(module_path, *class_names):
    """Nhập mô-đun theo đường dẫn tệp và trả về các lớp được chỉ định"""
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(getattr(module, name) for name in class_names)

# Import logging configuration
try:
    AuditLogger, DataLoadError, DataValidationError, FeaturePreparationError, log_exception = _import_module(
        "6_logging_config.py", 
        "AuditLogger", "DataLoadError", "DataValidationError", "FeaturePreparationError", "log_exception"
    )
except Exception:
    # Fallback to basic logging if logging_config not available
    import logging
    class AuditLogger:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)
    class DataLoadError(Exception):
        pass
    class DataValidationError(Exception):
        pass
    class FeaturePreparationError(Exception):
        pass
    def log_exception(logger, exc, context=None):
        logger.error(f"{type(exc).__name__}: {str(exc)}")

# Import from utility functions
FinancialRatioCalculator, TimeSeriesFeaturePreparation = _import_module(
    "4_utility_functions.py", 
    "FinancialRatioCalculator", 
    "TimeSeriesFeaturePreparation"
)


class DataPreparation:
    """
    Handles all data preparation tasks including:
    - Loading time series data
    - Calculating financial ratios
    - Feature engineering
    - Data validation
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.time_series_data = {}
        self.logger = AuditLogger.get_logger(__name__)
        
    def load_time_series_data(self, df: pd.DataFrame, bank_id: str, 
                              time_column: str = 'period') -> Dict:
        """
        Load and organize time series data for a specific bank
        
        Args:
            df: Input dataframe
            bank_id: Bank identifier
            time_column: Column name for time periods
            
        Returns:
            Dictionary containing organized time series data
            
        Raises:
            DataLoadError: If data loading fails or no data found for bank
            DataValidationError: If required columns are missing
        """
        context = {'bank_id': bank_id, 'method': 'load_time_series_data'}
        
        try:
            self.logger.info(f"Loading time series data for bank: {bank_id}")
            
            # Validate input
            if df is None or df.empty:
                raise DataLoadError("Input dataframe is None or empty", context)
            
            if 'bank_id' not in df.columns:
                raise DataValidationError("Column 'bank_id' not found in dataframe", context)
            
            if time_column not in df.columns:
                raise DataValidationError(f"Time column '{time_column}' not found in dataframe", context)
            
            bank_data = df[df['bank_id'] == bank_id].copy()
            
            if bank_data.empty:
                raise DataLoadError(f"No data found for bank: {bank_id}", context)
            
            self.logger.info(f"Found {len(bank_data)} records for bank {bank_id}")
            
            # Sort by time period
            bank_data = bank_data.sort_values(time_column)
            
            # Store periods and data
            periods = bank_data[time_column].values
            
            # Separate balance sheet and income statement data
            balance_sheet_cols = [col for col in bank_data.columns 
                                if any(x in col.lower() for x in ['asset', 'liability', 'equity', 'capital'])]
            income_statement_cols = [col for col in bank_data.columns 
                                    if any(x in col.lower() for x in ['revenue', 'income', 'expense', 'profit', 'interest'])]
            
            self.time_series_data = {
                'periods': periods,
                'raw_data': bank_data,
                'balance_sheet': bank_data[balance_sheet_cols] if balance_sheet_cols else pd.DataFrame(),
                'income_statement': bank_data[income_statement_cols] if income_statement_cols else pd.DataFrame(),
                'bank_id': bank_id
            }
            
            self.logger.info(f"Successfully loaded time series data with {len(periods)} periods")
            return self.time_series_data
            
        except (DataLoadError, DataValidationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            log_exception(self.logger, e, context)
            raise DataLoadError(f"Unexpected error loading data: {str(e)}", context) from e
    
    def calculate_time_series_ratios(self) -> pd.DataFrame:
        """
        Calculate financial ratios across all time periods
        
        Returns:
            DataFrame containing calculated ratios for each period
            
        Raises:
            DataValidationError: If no time series data loaded
            FeaturePreparationError: If ratio calculation fails
        """
        context = {
            'bank_id': self.time_series_data.get('bank_id', 'unknown'),
            'method': 'calculate_time_series_ratios'
        }
        
        try:
            if not self.time_series_data:
                raise DataValidationError("No time series data loaded. Call load_time_series_data first.", context)
            
            self.logger.info(f"Calculating time series ratios for bank {context['bank_id']}")
            
            raw_data = self.time_series_data['raw_data']
            
            # Vectorized ratio calculation using apply (much faster than iterrows)
            def _calc_ratio_safe(row):
                try:
                    period_ratios = FinancialRatioCalculator.calculate_ratios(row, data_format='series')
                    period_ratios['period'] = row.get('period') if 'period' in row.index else np.nan
                    return period_ratios
                except Exception as e:
                    self.logger.warning(f"Failed to calculate ratios for period {row.get('period', 'unknown')}: {str(e)}")
                    return None
            
            # Use apply with axis=1 (vectorized row-wise operation, much faster than iterrows)
            ratios_list = raw_data.apply(_calc_ratio_safe, axis=1).dropna()
            
            if len(ratios_list) == 0:
                raise FeaturePreparationError("Failed to calculate ratios for any period", context)
            
            ratios_df = pd.DataFrame(ratios_list.tolist()) if not ratios_list.empty else pd.DataFrame()
            self.logger.info(f"Successfully calculated ratios for {len(ratios_df)} periods")
            return ratios_df
            
        except (DataValidationError, FeaturePreparationError):
            raise
        except Exception as e:
            log_exception(self.logger, e, context)
            raise FeaturePreparationError(f"Unexpected error calculating ratios: {str(e)}", context) from e
    
    def calculate_growth_rates(self) -> pd.DataFrame:
        """
        Calculate period-over-period growth rates
        
        Returns:
            DataFrame containing growth rates
            
        Raises:
            DataValidationError: If no time series data loaded
            FeaturePreparationError: If growth calculation fails
        """
        context = {
            'bank_id': self.time_series_data.get('bank_id', 'unknown'),
            'method': 'calculate_growth_rates'
        }
        
        try:
            if not self.time_series_data:
                raise DataValidationError("No time series data loaded.", context)
            
            self.logger.info(f"Calculating growth rates for bank {context['bank_id']}")
            
            raw_data = self.time_series_data['raw_data']
            
            # Calculate percentage change for numeric columns
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            growth_rates = raw_data[numeric_cols].pct_change()
            
            growth_rates.columns = [f'{col}_growth' for col in growth_rates.columns]
            growth_rates['period'] = raw_data['period'].values if 'period' in raw_data.columns else range(len(growth_rates))
            
            self.logger.info(f"Successfully calculated growth rates for {len(numeric_cols)} metrics")
            return growth_rates
            
        except DataValidationError:
            raise
        except Exception as e:
            log_exception(self.logger, e, context)
            raise FeaturePreparationError(f"Failed to calculate growth rates: {str(e)}", context) from e
    
    def analyze_time_series_trends(self) -> Dict:
        """
        Analyze trends in time series data
        
        Returns:
            Dictionary containing trend analysis results
            
        Raises:
            DataValidationError: If no time series data loaded
            FeaturePreparationError: If trend analysis fails
        """
        context = {
            'bank_id': self.time_series_data.get('bank_id', 'unknown'),
            'method': 'analyze_time_series_trends'
        }
        
        try:
            if not self.time_series_data:
                raise DataValidationError("No time series data loaded.", context)
            
            self.logger.info(f"Analyzing time series trends for bank {context['bank_id']}")
            
            ratios_df = self.calculate_time_series_ratios()
            trends = {}
            
            for col in ratios_df.select_dtypes(include=[np.number]).columns:
                if col != 'period':
                    values = ratios_df[col].values
                    
                    # Calculate trend direction
                    if len(values) >= 2:
                        try:
                            trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                            trends[col] = {
                                'slope': trend_slope,
                                'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                                'volatility': np.std(values),
                                'mean': np.mean(values),
                                'latest': values[-1] if len(values) > 0 else 0,
                                'change_from_start': values[-1] - values[0] if len(values) > 0 else 0
                            }
                        except Exception as e:
                            self.logger.warning(f"Failed to calculate trend for column {col}: {str(e)}")
                            # Continue with other columns
            
            self.logger.info(f"Successfully analyzed trends for {len(trends)} metrics")
            return trends
            
        except DataValidationError:
            raise
        except Exception as e:
            log_exception(self.logger, e, context)
            raise FeaturePreparationError(f"Failed to analyze trends: {str(e)}", context) from e
    
    def prepare_time_series_features(self) -> pd.DataFrame:
        """
        Prepare comprehensive feature set for model training
        
        Returns:
            DataFrame with all engineered features
            
        Raises:
            DataValidationError: If no time series data loaded
            FeaturePreparationError: If feature preparation fails
        """
        context = {
            'bank_id': self.time_series_data.get('bank_id', 'unknown'),
            'method': 'prepare_time_series_features'
        }
        
        try:
            if not self.time_series_data:
                raise DataValidationError("No time series data loaded.", context)
            
            self.logger.info(f"Preparing time series features for bank {context['bank_id']}")
            
            # Get base ratios
            ratios_df = self.calculate_time_series_ratios()
            
            # Get growth rates
            growth_df = self.calculate_growth_rates()
            
            # Merge features
            features_df = ratios_df.merge(
                growth_df, 
                on='period', 
                how='left'
            )
            
            # Add rolling statistics
            features_df = self._add_rolling_features(features_df)
            
            # Fill missing values
            features_df = features_df.fillna(0)
            
            self.logger.info(f"Successfully prepared {features_df.shape[1]} features")
            return features_df
            
        except DataValidationError:
            raise
        except Exception as e:
            log_exception(self.logger, e, context)
            raise FeaturePreparationError(f"Failed to prepare features: {str(e)}", context) from e
    
    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6]) -> pd.DataFrame:
        """
        Add rolling window statistics
        
        Args:
            df: Input dataframe
            windows: List of window sizes
            
        Returns:
            DataFrame with added rolling features
        """
        import warnings
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'period']
        
        # Suppress DataFrame fragmentation warnings - acceptable for rolling feature generation
        # which is only called during training/analysis, not in hot loops
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            for window in windows:
                if len(df) >= window:
                    for col in numeric_cols:
                        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        
        return df
    
    def calculate_industry_benchmarks(self, 
                                       all_banks_data: pd.DataFrame,
                                       period: Optional[str] = None) -> Dict:
        """
        Calculate industry benchmarks from peer banks
        
        Args:
            all_banks_data: DataFrame containing data for all banks
            period: Specific period to calculate benchmarks for
            
        Returns:
            Dictionary of benchmark values
            
        Raises:
            DataValidationError: If input data is invalid
            FeaturePreparationError: If benchmark calculation fails
        """
        context = {'method': 'calculate_industry_benchmarks', 'period': period}
        
        try:
            if all_banks_data is None or all_banks_data.empty:
                raise DataValidationError("Input data is None or empty", context)
            
            self.logger.info(f"Calculating industry benchmarks{f' for period {period}' if period else ' for all periods'}")
            
            if period:
                period_data = all_banks_data[all_banks_data['period'] == period]
                if period_data.empty:
                    self.logger.warning(f"No data found for period {period}")
                    return {}
            else:
                period_data = all_banks_data
                
            benchmarks = {}
            
            # Calculate benchmarks for key metrics
            numeric_cols = period_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    values = period_data[col].dropna()
                    if len(values) > 0:
                        benchmarks[col] = {
                            'mean': values.mean(),
                            'median': values.median(),
                            'std': values.std(),
                            'p25': values.quantile(0.25),
                            'p75': values.quantile(0.75),
                            'min': values.min(),
                            'max': values.max()
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to calculate benchmark for column {col}: {str(e)}")
                    # Continue with other columns
            
            self.logger.info(f"Successfully calculated benchmarks for {len(benchmarks)} metrics")
            return benchmarks
            
        except DataValidationError:
            raise
        except Exception as e:
            log_exception(self.logger, e, context)
            raise FeaturePreparationError(f"Failed to calculate benchmarks: {str(e)}", context) from e
    
    def scale_features(self, features_df: pd.DataFrame, 
                      exclude_cols: List[str] = ['period']) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Scale features using StandardScaler
        
        Args:
            features_df: DataFrame containing features
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
            
        Raises:
            DataValidationError: If input dataframe is invalid
            FeaturePreparationError: If scaling fails
        """
        context = {'method': 'scale_features'}
        
        try:
            if features_df is None or features_df.empty:
                raise DataValidationError("Input dataframe is None or empty", context)
            
            self.logger.info(f"Scaling features, excluding columns: {exclude_cols}")
            
            numeric_cols = [col for col in features_df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude_cols]
            
            if not numeric_cols:
                self.logger.warning("No numeric columns found to scale")
                return features_df, self.scaler
            
            scaled_data = features_df.copy()
            
            # Clean data: replace infinity with NaN, then handle NaN values
            features_to_scale = scaled_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
            for col in numeric_cols:
                if features_to_scale[col].isna().any():
                    features_to_scale[col].fillna(features_to_scale[col].median(), inplace=True)
            features_to_scale = features_to_scale.fillna(0)
            
            scaled_data[numeric_cols] = self.scaler.fit_transform(features_to_scale)
            
            self.logger.info(f"Successfully scaled {len(numeric_cols)} features")
            return scaled_data, self.scaler
            
        except DataValidationError:
            raise
        except Exception as e:
            log_exception(self.logger, e, context)
            raise FeaturePreparationError(f"Failed to scale features: {str(e)}", context) from e
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality and identify issues
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
            
        Raises:
            DataValidationError: If input dataframe is invalid
        """
        context = {'method': 'validate_data_quality'}
        
        try:
            if df is None or df.empty:
                raise DataValidationError("Input dataframe is None or empty", context)
            
            self.logger.info(f"Validating data quality for {len(df)} records")
            
            validation_results = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_records': df.duplicated().sum(),
                'data_types': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                'issues': []
            }
            
            # Check for missing critical columns
            critical_cols = ['total_assets', 'total_equity', 'net_income']
            for col in critical_cols:
                if col in df.columns:
                    missing_pct = df[col].isnull().sum() / len(df) * 100
                    if missing_pct > 10:
                        issue_msg = f"High missing rate ({missing_pct:.1f}%) in critical column: {col}"
                        validation_results['issues'].append(issue_msg)
                        self.logger.warning(issue_msg)
            
            # Check for negative values where they shouldn't exist
            positive_cols = ['total_assets', 'total_deposits', 'total_loans']
            for col in positive_cols:
                if col in df.columns:
                    if (df[col] < 0).any():
                        issue_msg = f"Negative values found in {col}"
                        validation_results['issues'].append(issue_msg)
                        self.logger.warning(issue_msg)
            
            self.logger.info(f"Data quality validation complete. Found {len(validation_results['issues'])} issues")
            return validation_results
            
        except DataValidationError:
            raise
        except Exception as e:
            log_exception(self.logger, e, context)
            raise DataValidationError(f"Failed to validate data quality: {str(e)}", context) from e
