# -*- coding: utf-8 -*-
"""
Mô-đun Các Hàm Tiện ích
Hợp nhất các hàm được sử dụng thường xuyên theo nguyên tắc DRY
Cung cấp các hàm chung để tính tỷ lệ, chuẩn bị đặc trưng và huấn luyện mô hình
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from sklearn.preprocessing import StandardScaler
import importlib.util

# Import logging configuration
try:
    def _import_logging():
        spec = importlib.util.spec_from_file_location("logging_config", "6_logging_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (module.AuditLogger, module.FeaturePreparationError, 
                module.ModelTrainingError, module.log_exception)
    
    AuditLogger, FeaturePreparationError, ModelTrainingError, log_exception = _import_logging()
except Exception:
    # Fallback to basic logging
    import logging
    class AuditLogger:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)
    class FeaturePreparationError(Exception):
        pass
    class ModelTrainingError(Exception):
        pass
    def log_exception(logger, exc, context=None):
        logger.error(f"{type(exc).__name__}: {str(exc)}")

logger = AuditLogger.get_logger(__name__)

class FinancialRatioCalculator:
    """
    Centralized financial ratio calculation module
    Eliminates duplication of ratio calculation logic across the codebase
    """
    
    # Class-level logger
    logger = None
    
    @classmethod
    def _get_logger(cls):
        """Get or create logger instance"""
        if cls.logger is None:
            cls.logger = AuditLogger.get_logger(__name__)
        return cls.logger
    
    @staticmethod
    def calculate_ratios(data: Dict[str, Any], data_format: str = 'dict') -> Dict[str, float]:
        """
        Calculate financial ratios from financial data
        
        Supports multiple data formats and handles missing values gracefully
        
        Args:
            data: Dictionary or Series containing financial data
                  Can include both balance sheet and income statement items
            data_format: Format of input data ('dict' for dict-like, 'series' for pd.Series)
            
        Returns:
            Dictionary of calculated financial ratios
            
        Raises:
            FeaturePreparationError: If ratio calculation fails
            
        Examples:
            >>> data = {
            ...     'total_loans': 1000, 'non_performing_loans': 50,
            ...     'total_assets': 2000, 'total_equity': 200,
            ...     'net_income': 100, 'operating_expenses': 50
            ... }
            >>> ratios = FinancialRatioCalculator.calculate_ratios(data)
        """
        logger = FinancialRatioCalculator._get_logger()
        ratios = {}
        
        try:
            # Convert Series to dict if needed
            if data_format == 'series':
                data = data.to_dict() if hasattr(data, 'to_dict') else dict(data)
            
            # Ensure data is a dictionary
            if not isinstance(data, dict):
                data = {}
            
            # ===== CREDIT RISK RATIOS =====
            total_loans = data.get('total_loans', data.get('gross_loans', 1))
            npl = data.get('non_performing_loans', data.get('npl', 0))
            loan_loss_provisions = data.get('loan_loss_provisions', data.get('loan_loss_reserves', 0))
            
            ratios['npl_ratio'] = npl / max(total_loans, 1)
            ratios['provision_coverage_ratio'] = loan_loss_provisions / max(npl, 1) if npl > 0 else 1.0
            
            # ===== CAPITAL ADEQUACY RATIOS =====
            total_assets = data.get('total_assets', 1)
            total_equity = data.get('total_equity', 1)
            tier1_capital = data.get('tier1_capital', total_equity * 0.8)
            tier2_capital = data.get('tier2_capital', total_equity * 0.2)
            risk_weighted_assets = data.get('risk_weighted_assets', total_assets * 0.7)
            
            ratios['capital_adequacy_ratio'] = (tier1_capital + tier2_capital) / max(risk_weighted_assets, 1)
            ratios['equity_to_assets'] = total_equity / max(total_assets, 1)
            
            # ===== LIQUIDITY RATIOS =====
            liquid_assets = data.get('liquid_assets', data.get('high_quality_liquid_assets', 
                                                               data.get('cash_and_equivalents', 0)))
            total_deposits = data.get('total_deposits', data.get('customer_deposits', 1))
            net_cash_outflows = data.get('net_cash_outflows_30d', total_deposits * 0.3)
            short_term_liabilities = data.get('short_term_liabilities', total_deposits)
            
            ratios['liquidity_ratio'] = liquid_assets / max(short_term_liabilities, 1)
            ratios['liquidity_coverage_ratio'] = liquid_assets / max(net_cash_outflows, 1)
            ratios['loan_to_deposit_ratio'] = total_loans / max(total_deposits, 1)
            
            # ===== PROFITABILITY RATIOS =====
            net_income = data.get('net_income', data.get('profit_after_tax', 0))
            total_revenue = data.get('total_revenue', data.get('operating_income', 1))
            interest_income = data.get('interest_income', 0)
            interest_expense = data.get('interest_expense', 0)
            operating_expenses = data.get('operating_expenses', 0)
            
            ratios['return_on_assets'] = net_income / max(total_assets, 1)
            ratios['return_on_equity'] = net_income / max(total_equity, 1)
            ratios['net_interest_margin'] = (interest_income - interest_expense) / max(total_assets, 1)
            ratios['cost_to_income_ratio'] = operating_expenses / max(total_revenue, 1)
            
            # ===== PORTFOLIO DIVERSIFICATION & EXPOSURE =====
            sector_keys = [
                'sector_loans_energy', 'sector_loans_real_estate', 'sector_loans_construction',
                'sector_loans_services', 'sector_loans_agriculture'
            ]
            sector_values = [float(data.get(k, 0) or 0) for k in sector_keys]
            sector_total = sum(sector_values)
            if sector_total > 0 and total_loans > 0:
                shares = [v / max(total_loans, 1) for v in sector_values]
                ratios['sector_concentration_hhi'] = float(sum([s * s for s in shares]))
            else:
                ratios['sector_concentration_hhi'] = np.nan
            
            top1 = float(data.get('top1_borrower_loans', 0) or 0)
            top10 = float(data.get('top10_borrower_loans', 0) or 0)
            top20 = float(data.get('top20_borrower_loans', 0) or 0)
            ratios['top_borrower_concentration'] = top1 / max(total_loans, 1)
            ratios['top10_borrower_concentration'] = top10 / max(total_loans, 1)
            ratios['top20_borrower_concentration'] = top20 / max(total_loans, 1) if top20 > 0 else np.nan
            
            # ===== OFF-BALANCE SHEET RATIOS =====
            obs = float(data.get('obs_exposure_total', data.get('off_balance_exposure', 0)) or 0)
            derivatives_notional = float(data.get('derivatives_notional', 0) or 0)
            unused_credit_lines = float(data.get('unused_credit_lines', 0) or 0)
            guarantees = float(data.get('guarantees_issued', 0) or 0)
            
            ratios['obs_to_loans_ratio'] = obs / max(total_loans, 1)
            ratios['obs_to_assets_ratio'] = obs / max(total_assets, 1)
            ratios['derivatives_to_assets_ratio'] = derivatives_notional / max(total_assets, 1)
            ratios['unused_lines_to_loans_ratio'] = unused_credit_lines / max(total_loans, 1)
            ratios['guarantees_to_loans_ratio'] = guarantees / max(total_loans, 1)
            
            # ===== CONCENTRATION METRICS =====
            # Deposit concentration
            top20_depositors = float(data.get('top20_depositors', 0) or 0)
            top5_depositors = float(data.get('top5_depositors', 0) or 0)
            ratios['top20_depositors_ratio'] = top20_depositors / max(total_deposits, 1)
            ratios['top5_depositors_ratio'] = top5_depositors / max(total_deposits, 1)
            
            # Geographic concentration
            geographic_concentration = float(data.get('geographic_concentration', np.nan) or np.nan)
            ratios['geographic_concentration'] = geographic_concentration
            
            # Composite risk indicators
            ratios['liquidity_concentration_risk'] = float(data.get('liquidity_concentration_risk', np.nan) or np.nan)
            ratios['credit_concentration_risk'] = float(data.get('credit_concentration_risk', np.nan) or np.nan)
            ratios['obs_risk_indicator'] = float(data.get('obs_risk_indicator', np.nan) or np.nan)
            
            rating = str(data.get('external_credit_rating', '') or '').upper()
            rating_map = {
                'AAA': 1, 'AA+': 2, 'AA': 2, 'AA-': 3,
                'A+': 3, 'A': 3, 'A-': 4,
                'BBB+': 4, 'BBB': 4, 'BBB-': 5,
                'BB+': 5, 'BB': 6, 'BB-': 6,
                'B+': 7, 'B': 7, 'B-': 8,
                'CCC+': 8, 'CCC': 9, 'CCC-': 9,
                'CC': 9, 'C': 10, 'D': 10
            }
            ratios['external_rating_score'] = rating_map.get(rating, np.nan)
            
            # ===== EFFICIENCY RATIOS =====
            employees = data.get('number_of_employees', 1)
            ratios['employee_productivity'] = net_income / max(employees, 1)
            ratios['efficiency_ratio'] = operating_expenses / max(total_revenue, 1)
            ratios['asset_utilization'] = total_revenue / max(total_assets, 1)
            
            return ratios
            
        except Exception as e:
            logger.warning(f"Error calculating ratios: {str(e)}")
            # Return empty dict but don't raise - allows partial success
            return ratios
    
    @staticmethod
    def calculate_growth_metrics(current_data: Dict, previous_data: Dict) -> Dict[str, float]:
        """
        Calculate period-over-period growth rates
        
        Args:
            current_data: Current period financial data
            previous_data: Previous period financial data
            
        Returns:
            Dictionary of growth metrics
        """
        logger = FinancialRatioCalculator._get_logger()
        growth_metrics = {}
        
        try:
            key_metrics = ['total_assets', 'total_loans', 'total_deposits', 'net_income', 'total_equity']
            
            for metric in key_metrics:
                current_val = current_data.get(metric, 0)
                previous_val = previous_data.get(metric, 1)
                
                if previous_val > 0:
                    growth_metrics[f'{metric}_growth'] = (current_val - previous_val) / previous_val
            
            return growth_metrics
                    
        except Exception as e:
            logger.warning(f"Error calculating growth metrics: {str(e)}")
            return growth_metrics


class TimeSeriesFeaturePreparation:
    """
    Generic time series feature preparation for model training
    Consolidates duplicate logic in train_credit_risk_models_with_time_series 
    and train_liquidity_risk_models_with_time_series
    """
    
    # Class-level logger
    logger = None
    
    @classmethod
    def _get_logger(cls):
        """Get or create logger instance"""
        if cls.logger is None:
            cls.logger = AuditLogger.get_logger(__name__)
        return cls.logger
    
    @staticmethod
    def prepare_training_features(
        all_banks_data: List[Dict],
        feature_calculator: Callable[[Dict], np.ndarray],
        data_preprocessor: Optional[Callable[[Dict], Dict]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generic function to prepare training features from multiple banks' time series data
        
        This consolidates the repeated loop logic across different risk models
        by accepting a custom feature calculator function
        
        Args:
            all_banks_data: List of bank data dictionaries, each containing:
                - 'bank_name': Name of the bank
                - 'time_series_data': Time series data for that bank
            feature_calculator: Callable function that extracts features from a single bank's data
                               Should accept bank_data dict and return 1D numpy array
            data_preprocessor: Optional callable to preprocess bank data before feature extraction
                              Should accept and return bank_data dict
                              
        Returns:
            Tuple of (feature_matrix, number_of_valid_samples)
            - X_train: 2D numpy array of shape (n_samples, n_features)
            - valid_count: Number of banks successfully processed
            
        Raises:
            ValueError: If insufficient data or invalid inputs provided
            
        Examples:
            >>> def extract_credit_features(bank_data):
            ...     # Extract credit-specific features
            ...     return feature_array
            >>> X_train, count = TimeSeriesFeaturePreparation.prepare_training_features(
            ...     all_banks_data, extract_credit_features
            ... )
        """
        logger = TimeSeriesFeaturePreparation._get_logger()
        X_train = []
        valid_count = 0
        
        try:
            if not all_banks_data:
                raise FeaturePreparationError("No bank data provided", {'method': 'prepare_training_features'})
            
            logger.info(f"Preparing training features for {len(all_banks_data)} banks")
            
            for bank_data in all_banks_data:
                try:
                    # Preprocess if provided
                    if data_preprocessor:
                        bank_data = data_preprocessor(bank_data)
                    
                    # Extract features using provided calculator
                    features = feature_calculator(bank_data)
                    
                    # Validate features
                    if features is not None and len(features) > 0:
                        X_train.append(features.flatten())
                        valid_count += 1
                        
                except Exception as e:
                    bank_name = bank_data.get('bank_name', 'unknown')
                    logger.warning(f"Error processing bank {bank_name}: {str(e)}")
                    continue
            
            if len(X_train) == 0:
                raise FeaturePreparationError("No valid training data extracted from any bank", 
                                            {'method': 'prepare_training_features', 'total_banks': len(all_banks_data)})
            
            logger.info(f"Successfully prepared features for {valid_count}/{len(all_banks_data)} banks")
            return np.array(X_train), valid_count
            
        except FeaturePreparationError:
            raise
        except Exception as e:
            log_exception(logger, e, {'method': 'prepare_training_features'})
            raise FeaturePreparationError(f"Failed to prepare training features: {str(e)}") from e
    
    @staticmethod
    def prepare_bank_features(
        bank_name: str,
        time_series_data: Dict,
        ratio_calculator: Optional[Callable] = None,
        growth_calculator: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Prepare features for a single bank's time series data
        
        Args:
            bank_name: Name of the bank
            time_series_data: Dictionary containing time series information
            ratio_calculator: Optional custom ratio calculator function
            growth_calculator: Optional custom growth calculator function
            
        Returns:
            1D numpy array of features
        """
        features = []
        
        try:
            # Use default ratio calculator if not provided
            if ratio_calculator is None:
                ratio_calculator = FinancialRatioCalculator.calculate_ratios
            
            # Extract features from time series data
            if 'data' in time_series_data:
                for period_data in time_series_data['data']:
                    # Combine balance sheet and income statement
                    combined_data = {
                        **period_data.get('balance_sheet', {}),
                        **period_data.get('income_statement', {})
                    }
                    
                    # Calculate ratios for this period
                    ratios = ratio_calculator(combined_data)
                    features.extend(ratios.values())
            
            return np.array(features) if features else np.array([])
            
        except Exception as e:
            logger.error(f"Error preparing time series features for {bank_name}: {e}")
            return np.array([])


class ModelTrainingPipeline:
    """
    Generic model training pipeline to reduce duplication
    across different risk model training methods
    """
    
    @staticmethod
    def train_models_on_features(
        X_train: np.ndarray,
        model_configs: List[Dict[str, Any]],
        scaler: Optional[StandardScaler] = None
    ) -> Tuple[Dict, StandardScaler]:
        """
        Generic function to train multiple models on prepared features
        
        Args:
            X_train: Feature matrix of shape (n_samples, n_features)
            model_configs: List of model configuration dictionaries, each containing:
                - 'name': Model name (string)
                - 'model_class': Sklearn model class
                - 'params': Dictionary of model parameters
                - 'enabled': Boolean to enable/disable model
            scaler: Optional pre-fitted StandardScaler. If None, creates new one
            
        Returns:
            Tuple of (trained_models_dict, fitted_scaler)
            
        Example:
            >>> model_configs = [
            ...     {
            ...         'name': 'isolation_forest',
            ...         'model_class': IsolationForest,
            ...         'params': {'contamination': 0.1, 'n_estimators': 100},
            ...         'enabled': True
            ...     }
            ... ]
            >>> models, scaler = ModelTrainingPipeline.train_models_on_features(
            ...     X_train, model_configs
            ... )
        """
        models = {}
        
        # Create or use provided scaler
        if scaler is None:
            scaler = StandardScaler()
        
        # Clean data: replace infinity with NaN, then handle NaN values
        X_train_clean = np.where(np.isfinite(X_train), X_train, np.nan)
        X_df = pd.DataFrame(X_train_clean)
        for col in X_df.columns:
            if X_df[col].isna().any():
                X_df[col].fillna(X_df[col].median(), inplace=True)
        X_df = X_df.fillna(0)
        X_train_clean = X_df.values
        
        # Scale features
        try:
            X_scaled = scaler.fit_transform(X_train_clean)
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return models, scaler
        
        # Train each model
        for config in model_configs:
            if not config.get('enabled', True):
                continue
            
            try:
                model_name = config.get('name')
                model_class = config.get('model_class')
                params = config.get('params', {})
                
                # Instantiate and train model
                model = model_class(**params)
                model.fit(X_scaled)
                models[model_name] = model
                
                logger.debug(f"Successfully trained model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error training model {config.get('name')}: {e}")
                continue
        
        return models, scaler
    
    @staticmethod
    def validate_training_data(X_train: np.ndarray, min_samples: int = 3) -> bool:
        """
        Validate that training data is sufficient for model training
        
        Args:
            X_train: Feature matrix
            min_samples: Minimum number of samples required
            
        Returns:
            Boolean indicating whether data is valid
        """
        if X_train is None or len(X_train) == 0:
            logger.error("Error: No training data provided")
            return False
        
        if len(X_train) < min_samples:
            logger.error(f"Error: Insufficient training data ({len(X_train)} < {min_samples})")
            return False
        
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            logger.error("Error: Training data contains NaN or infinite values")
            return False
        
        return True

# ===== STANDALONE HIGH-RISK PERIOD DETECTION FUNCTION =====
def identify_high_risk_periods(
    prepared_data: Dict,
    npl_threshold: float = 0.05,
    liquidity_threshold: float = 0.3,
    capital_threshold: float = 0.08,
    min_risk_indicators: int = 2,
    time_column: str = 'period'
) -> List[Dict]:
    """
    Identify periods with elevated risk using vectorized operations.
    
    This is a standalone utility function that can be used independently or
    called by BankAuditSystem.identify_high_risk_periods() method.
    
    Args:
        prepared_data: Dictionary containing 'features' DataFrame with risk indicators
        npl_threshold: Maximum acceptable NPL ratio (default 0.05 = 5%)
        liquidity_threshold: Minimum acceptable liquidity ratio (default 0.3 = 30%)
        capital_threshold: Minimum acceptable capital adequacy ratio (default 0.08 = 8%)
        min_risk_indicators: Minimum number of risk indicators to flag as high-risk (default 2)
        time_column: Name of time/period column (default 'period')
        
    Returns:
        List of dictionaries containing high-risk periods with indicators
        
    Example:
        >>> prepared_data = {'features': df_with_risk_ratios}
        >>> high_risk = identify_high_risk_periods(prepared_data)
        >>> for period_info in high_risk:
        ...     print(f"Period {period_info['period']}: {period_info['risk_indicators']}")
    """
    if not prepared_data or 'features' not in prepared_data:
        return []
    
    features_df = prepared_data['features']
    
    if features_df.empty or time_column not in features_df.columns:
        return []
    
    # Vectorized risk indicator detection using boolean arrays (much faster than iterrows)
    npl_risk = (features_df.get('npl_ratio', 0) > npl_threshold).fillna(False).astype(int)
    liq_risk = (features_df.get('liquidity_ratio', 1) < liquidity_threshold).fillna(False).astype(int)
    cap_risk = (features_df.get('capital_adequacy_ratio', 1) < capital_threshold).fillna(False).astype(int)
    
    # Total risk count per row (vectorized arithmetic)
    risk_count = npl_risk + liq_risk + cap_risk
    
    # Filter only rows with min_risk_indicators+ risk indicators (vectorized boolean indexing)
    high_risk_mask = risk_count >= min_risk_indicators
    high_risk_indices = np.where(high_risk_mask.values)[0]
    
    # Build result list efficiently
    high_risk_periods = []
    for idx in high_risk_indices:
        indicators = []
        if npl_risk.iloc[idx]:
            indicators.append('High NPL')
        if liq_risk.iloc[idx]:
            indicators.append('Low Liquidity')
        if cap_risk.iloc[idx]:
            indicators.append('Low Capital')
        
        high_risk_periods.append({
            'period': features_df.iloc[idx].get(time_column, idx),
            'risk_indicators': indicators,
            'severity': 'CRITICAL' if risk_count.iloc[idx] >= 3 else 'HIGH' if risk_count.iloc[idx] >= 2 else 'MEDIUM',
            'npl_ratio': float(features_df.iloc[idx].get('npl_ratio', 0)),
            'liquidity_ratio': float(features_df.iloc[idx].get('liquidity_ratio', 0)),
            'capital_adequacy_ratio': float(features_df.iloc[idx].get('capital_adequacy_ratio', 0))
        })
    
    return high_risk_periods