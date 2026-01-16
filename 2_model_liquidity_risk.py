# -*- coding: utf-8 -*-
"""
Mô-đun Đánh giá Rủi ro Thanh khoản
Xử lý mô hình và phân tích rủi ro thanh khoản
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional
import logging
import importlib.util

# Import AuditLogger from 6_logging_config.py
spec_logging = importlib.util.spec_from_file_location("logging_config", "6_logging_config.py")
logging_module = importlib.util.module_from_spec(spec_logging)
spec_logging.loader.exec_module(logging_module)
AuditLogger = logging_module.AuditLogger

logger = AuditLogger.get_logger(__name__)

# Hàm trợ giúp để nhập mô-đun có tiền tố số
def _import_module(module_path, class_name):
    """Nhập mô-đun theo đường dẫn tệp và trả về lớp được chỉ định"""
    try:
        spec = importlib.util.spec_from_file_location("temp_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except Exception as e:
        logger.error(f"Error importing {class_name} from {module_path}: {e}")
        raise

BaseRiskModel = _import_module("2_model_base_risk.py", "BaseRiskModel")


class LiquidityRiskModel(BaseRiskModel):
    """
    Mô hình Đánh giá Rủi ro Thanh khoản
    Đánh giá rủi ro thanh khoản và ổn định tài trợ
    """
    
    def __init__(self):
        super().__init__("Liquidity Risk Model")
        self.liquidity_indicators = {}
        self.stress_test_results = {}
        self.peer_benchmarks = {}  # Store peer comparison data for explanations
        
    def train_models(self, training_data: pd.DataFrame, **kwargs) -> Dict:
        """
        Train liquidity risk models
        
        Args:
            training_data: Historical liquidity data from multiple banks
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training results
        """
        # Prepare liquidity features
        features = self._prepare_liquidity_features(training_data)
        
        if features.empty:
            return {'status': 'failed', 'reason': 'No valid features'}
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Clean data: replace infinity with NaN, then handle NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column median
        for col in features.columns:
            if features[col].isna().any():
                features[col].fillna(features[col].median(), inplace=True)
        
        # Double-check for any remaining NaN values
        features = features.fillna(0)
        
        # Calculate peer benchmarks for regulatory explanations
        self._calculate_peer_benchmarks(features)
        
        # Scale features - use .values to avoid sklearn feature name validation
        X_scaled = self.scaler.fit_transform(features.values)
        
        training_results = {}
        
        # Train regression models for liquidity forecasting
        training_results['regression'] = self._train_forecast_models(X_scaled)
        
        # Train anomaly detection for liquidity stress
        training_results['anomaly'] = self._train_anomaly_models(X_scaled)
        
        # Record training
        self.training_history.append({
            'timestamp': pd.Timestamp.now(),
            'samples': len(features),
            'features': len(self.feature_names),
            'results': training_results
        })
        
        return training_results
    
    def _prepare_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare liquidity-specific features
        
        Args:
            data: Input dataframe
            
        Returns:
            DataFrame with liquidity risk features
        """
        # Collect all columns at once to avoid fragmentation
        cols_to_add = []
        
        # Core Liquidity Ratios
        if 'liquidity_ratio' in data.columns:
            cols_to_add.append('liquidity_ratio')
        
        if 'loan_to_deposit' in data.columns:
            cols_to_add.append('loan_to_deposit')
        
        # Funding Stability
        if 'deposit_stability' in data.columns:
            cols_to_add.append('deposit_stability')
        
        # Asset Liquidity
        if 'liquid_assets_ratio' in data.columns:
            cols_to_add.append('liquid_assets_ratio')
        
        # Cash Flow Indicators
        if 'net_interest_margin' in data.columns:
            cols_to_add.append('net_interest_margin')
        
        # Growth in deposits (sudden withdrawals indicator)
        deposit_growth_cols = [col for col in data.columns 
                              if 'deposit' in col and '_growth' in col]
        cols_to_add.extend(deposit_growth_cols)
        
        # Basel III NSFR (Net Stable Funding Ratio) - structural funding metric
        # Complements short-term LCR with 1-year horizon stability assessment
        if 'nsfr' in data.columns:
            cols_to_add.append('nsfr')
        
        # Include wholesale funding indicators to monitor market funding dependency
        wholesale_cols = ['wholesale_funding_short_term', 'wholesale_funding_stable',
                         'loan_to_total_funding_ratio', 'wholesale_dependency_ratio']
        for col in wholesale_cols:
            if col in data.columns:
                cols_to_add.append(col)
        
        # Create features DataFrame all at once
        features = data[cols_to_add].copy() if cols_to_add else pd.DataFrame()
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def _train_forecast_models(self, X: np.ndarray) -> Dict:
        """
        Train models to forecast liquidity metrics
        
        Args:
            X: Feature matrix
            
        Returns:
            Training results
        """
        results = {}
        
        # Random Forest Regressor for liquidity forecasting
        rf_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Use cross-validation approach
        from sklearn.model_selection import cross_val_score
        try:
            # Create synthetic target (next period liquidity)
            y = X[:, 0] if X.shape[1] > 0 else X.flatten()
            
            rf_regressor.fit(X, y)
            self.models['rf_regressor'] = rf_regressor
            
            cv_scores = cross_val_score(rf_regressor, X, y, cv=min(5, len(X)), random_state=42)
            results['rf_regressor'] = {
                'trained': True,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std()
            }
        except Exception as e:
            results['rf_regressor'] = {'trained': False, 'error': str(e)}
        
        return results
    
    def _train_anomaly_models(self, X: np.ndarray) -> Dict:
        """
        Train anomaly detection for liquidity stress events
        
        Args:
            X: Feature matrix
            
        Returns:
            Training results
        """
        results = {}
        
        # Isolation Forest for detecting liquidity stress
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        iso_forest.fit(X)
        self.models['isolation_forest'] = iso_forest
        results['isolation_forest'] = {'trained': True}
        
        return results
    
    def predict_risk(self, input_data: pd.DataFrame) -> Dict:
        """
        Predict liquidity risk
        
        Args:
            input_data: Current bank data
            
        Returns:
            Dictionary containing liquidity risk assessment
        """
        # Prepare features
        features = self._prepare_liquidity_features(input_data)
        
        if features.empty:
            return {'status': 'failed', 'reason': 'No valid features'}
        
        # Align feature columns with those used during training
        if hasattr(self, 'feature_names') and self.feature_names:
            features = features.reindex(columns=self.feature_names, fill_value=0)

        # Clean data: replace infinity with NaN, then handle NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        for col in features.columns:
            if features[col].isna().any():
                features[col].fillna(features[col].median(), inplace=True)
        features = features.fillna(0)

        # Scale features - use .values to avoid sklearn feature name validation
        X_scaled = self.scaler.transform(features.values)
        
        predictions = {}
        
        # Get anomaly scores
        if 'isolation_forest' in self.models:
            anomaly_scores = self.models['isolation_forest'].decision_function(X_scaled)
            is_anomaly = self.models['isolation_forest'].predict(X_scaled)
            
            predictions['anomaly_detection'] = {
                'anomaly_score': -anomaly_scores.mean(),
                'is_stress_event': (is_anomaly == -1).any(),
                'stress_probability': (is_anomaly == -1).sum() / len(is_anomaly)
            }
        
        # Calculate comprehensive liquidity risk score
        liquidity_risk_score = self._calculate_comprehensive_liquidity_risk(
            input_data, predictions
        )
        
        # Generate regulatory explanation
        explanation = self._generate_liquidity_risk_explanation(
            input_data, predictions, liquidity_risk_score
        )
        
        self.risk_scores['liquidity'] = liquidity_risk_score
        
        return {
            'liquidity_risk_score': liquidity_risk_score,
            'risk_level': self.classify_risk_level(liquidity_risk_score),
            'predictions': predictions,
            'indicators': self.liquidity_indicators,
            'explanation': explanation,  # NEW: Regulatory transparency
            'regulatory_narrative': explanation.get('narrative', '')
        }
    
    def apply_macro_adjustments(self,
                              liquidity_risk_score: float,
                              bank_data: Dict[str, float],
                              all_banks_data: pd.DataFrame,
                              systemic_stress_level: Optional[float] = None) -> Dict:
        """
        Apply macro-adjustments to liquidity risk score based on industry context.
        
        Args:
            liquidity_risk_score: Original liquidity risk score
            bank_data: Bank's key metrics (liquidity_ratio, lcr, etc.)
            all_banks_data: DataFrame with all banks for benchmarking
            systemic_stress_level: Optional pre-calculated stress level (0-1)
            
        Returns:
            Dictionary with adjusted score and explanation
        """
        try:
            from macro_adjustments import MacroAdjustmentCalculator
            
            calc = MacroAdjustmentCalculator()
            
            # Define stress metrics (including structural funding and wholesale dependency)
            stress_metrics = ['liquidity_ratio', 'lcr', 'nsfr', 'wholesale_dependency_ratio']
            thresholds = {
                'liquidity_ratio': ('<', 0.3),
                'lcr': ('<', 1.0),
                'nsfr': ('<', 1.0),  # Basel III minimum: NSFR >= 100%
                'wholesale_dependency_ratio': ('>', 0.50)  # High market funding dependency
            }
            
            # Calculate systemic stress if not provided
            if systemic_stress_level is None:
                systemic_stress_level = calc.estimate_systemic_stress(
                    all_banks_data,
                    stress_metrics,
                    thresholds
                )
            
            # Calculate benchmarks for key liquidity metrics (including wholesale funding)
            liquidity_metrics = ['liquidity_ratio', 'lcr', 'nsfr', 
                               'loan_to_total_funding_ratio', 'wholesale_dependency_ratio']
            benchmarks = calc.calculate_industry_benchmarks(
                all_banks_data,
                liquidity_metrics
            )
            
            # Calculate relative z-scores for key metrics
            z_score = 0
            if 'liquidity_ratio' in benchmarks and 'liquidity_ratio' in bank_data:
                benchmark = benchmarks['liquidity_ratio']
                z_score = calc.calculate_relative_deviation(
                    bank_data['liquidity_ratio'],
                    benchmark['mean'],
                    benchmark['std']
                )
            
            # Apply adjustment
            adjustment = calc.adjust_risk_score(
                liquidity_risk_score,
                z_score,
                systemic_stress_level,
                adjustment_strength=0.35
            )
            
            return {
                'original_score': float(liquidity_risk_score),
                'adjusted_score': adjustment['adjusted_score'],
                'adjustment_delta': adjustment['adjustment_delta'],
                'systemic_stress_level': systemic_stress_level,
                'relative_z_score': z_score,
                'adjustment_reason': adjustment['adjustment_reason'],
                'adjustment_confidence': adjustment['adjustment_confidence'],
                'benchmarks': benchmarks
            }
        
        except ImportError:
            self._init_logger()
            self.logger.warning("macro_adjustments module not available")
            return {
                'original_score': float(liquidity_risk_score),
                'adjusted_score': float(liquidity_risk_score),
                'adjustment_delta': 0.0,
                'adjustment_reason': 'Macro-adjustments not available'
            }
    
    def _init_logger(self):
        """Initialize logger if needed"""
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(self.__class__.__name__)
    
    def _calculate_comprehensive_liquidity_risk(self, data: pd.DataFrame,
                                               predictions: Dict) -> float:
        """
        Calculate comprehensive liquidity risk score
        
        Args:
            data: Input data
            predictions: Model predictions
            
        Returns:
            Liquidity risk score (0-100)
        """
        risk_components = []
        
        # Component 1: Liquidity Ratio (30% weight)
        if 'liquidity_ratio' in data.columns:
            liq_ratio = data['liquidity_ratio'].mean()
            # Lower liquidity = higher risk (invert)
            liq_risk = max(100 - liq_ratio * 200, 0)  # 50% = 0 risk, 0% = 100 risk
            risk_components.append(('liquidity_ratio', liq_risk, 0.3))
        
        # Component 2: Loan to Deposit (25% weight)
        if 'loan_to_deposit' in data.columns:
            ltd = data['loan_to_deposit'].mean()
            # Higher LTD = higher risk
            ltd_risk = min(ltd * 100, 100)  # 100% LTD = 100 risk
            risk_components.append(('loan_to_deposit', ltd_risk, 0.25))
        
        # Component 3: Anomaly Detection (25% weight)
        if 'anomaly_detection' in predictions:
            anomaly_risk = predictions['anomaly_detection']['stress_probability'] * 100
            risk_components.append(('anomaly', anomaly_risk, 0.25))
        
        # Component 4: Deposit Volatility (20% weight)
        deposit_cols = [col for col in data.columns if 'deposit' in col and '_growth' in col]
        if deposit_cols:
            deposit_vol = abs(data[deposit_cols[0]].mean())
            vol_risk = min(deposit_vol * 500, 100)  # High volatility = high risk
            risk_components.append(('deposit_volatility', vol_risk, 0.2))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in risk_components)
        total_weight = sum(weight for _, _, weight in risk_components)
        
        if total_weight > 0:
            return total_score / total_weight
        return 50.0  # Default medium risk
    
    def assess_risk_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Assess specific liquidity risk indicators
        
        Args:
            data: Bank financial data
            
        Returns:
            Dictionary of liquidity risk indicators
        """
        indicators = {}
        
        # Liquidity Coverage Ratio
        if 'liquidity_ratio' in data.columns:
            liq_ratio = data['liquidity_ratio'].iloc[-1] if len(data) > 0 else 0
            try:
                from config.expert_rules_config import load_expert_rules
                rules = load_expert_rules()
                liq_min = rules.get('liquidity', {}).get('liquidity_ratio_min', 0.3)
            except Exception:
                liq_min = 0.3
            indicators['liquidity_ratio'] = {
                'value': liq_ratio,
                'status': 'LOW' if liq_ratio < liq_min else 'ADEQUATE',
                'threshold': liq_min,
                'description': 'Liquidity Coverage Ratio'
            }
        
        # Loan to Deposit Ratio
        if 'loan_to_deposit' in data.columns:
            ltd = data['loan_to_deposit'].iloc[-1] if len(data) > 0 else 0
            try:
                from config.expert_rules_config import load_expert_rules
                rules = load_expert_rules()
                max_ltd = rules.get('liquidity', {}).get('max_loan_to_deposit', 0.85)
            except Exception:
                max_ltd = 0.85
            indicators['loan_to_deposit'] = {
                'value': ltd,
                'status': 'HIGH_RISK' if ltd > max_ltd else 'ACCEPTABLE',
                'threshold': max_ltd,
                'description': 'Loan to Deposit Ratio'
            }
        
        # Funding Concentration
        if 'large_deposits_ratio' in data.columns:
            large_dep = data['large_deposits_ratio'].iloc[-1] if len(data) > 0 else 0
            indicators['funding_concentration'] = {
                'value': large_dep,
                'status': 'HIGH' if large_dep > 0.25 else 'NORMAL',
                'threshold': 0.25,
                'description': 'Concentration of Large Deposits'
            }
        
        # NSFR (Net Stable Funding Ratio) - Basel III Structural Liquidity
        # NSFR measures funding stability over 1-year horizon
        # Required: >= 100% (1.0)
        # Complements short-term LCR with structural funding assessment
        if 'nsfr' in data.columns:
            nsfr = data['nsfr'].iloc[-1] if len(data) > 0 else 0.95
            try:
                from config.expert_rules_config import load_expert_rules
                rules = load_expert_rules()
                # Get NSFR threshold from config (should be >= 100%)
                min_nsfr = rules.get('liquidity', {}).get('min_nsfr', 1.0)
            except Exception:
                min_nsfr = 1.0
            
            indicators['nsfr'] = {
                'value': nsfr,
                'status': 'ADEQUATE' if nsfr >= min_nsfr else 'INADEQUATE',
                'threshold': min_nsfr,
                'description': f'Net Stable Funding Ratio (Basel III, 1-year horizon)',
                'interpretation': (
                    'Measures structural funding stability. '
                    f'Current: {nsfr:.1%} vs. Required: {min_nsfr:.0%}'
                )
            }
        
        # Wholesale funding indicators - monitor market funding dependency
        if 'loan_to_total_funding_ratio' in data.columns:
            loan_to_total_funding = data['loan_to_total_funding_ratio'].iloc[-1] if len(data) > 0 else 0
            try:
                from config.expert_rules_config import load_expert_rules
                rules = load_expert_rules()
                max_ltf = rules.get('liquidity', {}).get('max_loan_to_total_funding', 0.90)
            except Exception:
                max_ltf = 0.90
            
            indicators['loan_to_total_funding'] = {
                'value': loan_to_total_funding,
                'status': 'ADEQUATE' if loan_to_total_funding <= max_ltf else 'EXCESSIVE',
                'threshold': max_ltf,
                'description': f'Loan-to-Total-Funding Ratio (includes wholesale stable funding)',
                'interpretation': (
                    f'Measures comprehensive funding coverage. '
                    f'Current: {loan_to_total_funding:.1%} vs. Max: {max_ltf:.0%}. '
                    f'More conservative than LDR by including stable wholesale funding.'
                )
            }
        
        if 'wholesale_dependency_ratio' in data.columns:
            wholesale_dep = data['wholesale_dependency_ratio'].iloc[-1] if len(data) > 0 else 0
            try:
                from config.expert_rules_config import load_expert_rules
                rules = load_expert_rules()
                max_wholesale_dep = rules.get('liquidity', {}).get('max_wholesale_dependency', 0.50)
            except Exception:
                max_wholesale_dep = 0.50
            
            # Classify dependency level
            if wholesale_dep < 0.30:
                dep_level = 'LOW (retail-funded)'
                status = 'HEALTHY'
            elif wholesale_dep <= 0.50:
                dep_level = 'MODERATE'
                status = 'ACCEPTABLE'
            else:
                dep_level = 'HIGH (market-dependent)'
                status = 'CONCERNING'
            
            indicators['wholesale_dependency'] = {
                'value': wholesale_dep,
                'status': status,
                'threshold': max_wholesale_dep,
                'dependency_level': dep_level,
                'description': f'Wholesale Funding Dependency Ratio',
                'interpretation': (
                    f'Measures reliance on market funding. '
                    f'Current: {wholesale_dep:.1%} ({dep_level}). '
                    f'Higher dependency increases vulnerability to market disruptions.'
                )
            }
        
        self.liquidity_indicators = indicators
        return indicators
    
    def run_liquidity_stress_test(self, data: pd.DataFrame, 
                                  scenarios: Optional[Dict] = None) -> Dict:
        """
        Run liquidity stress tests under various scenarios
        
        Args:
            data: Current bank data
            scenarios: Optional custom stress scenarios
            
        Returns:
            Dictionary of stress test results
        """
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            # Apply scenario shocks
            stressed_data = self._apply_stress_scenario(data, scenario)
            
            # Calculate liquidity under stress
            stress_result = self._calculate_stressed_liquidity(stressed_data)
            
            results[scenario_name] = {
                'scenario': scenario,
                'stressed_liquidity_ratio': stress_result['liquidity_ratio'],
                'liquidity_gap': stress_result['liquidity_gap'],
                'survival_period_days': stress_result['survival_days'],
                'passes_stress_test': stress_result['passes']
            }
        
        self.stress_test_results = results
        return results
    
    def _get_default_stress_scenarios(self) -> Dict:
        """
        Get default liquidity stress scenarios
        
        Returns:
            Dictionary of stress scenarios
        """
        return {
            'mild_stress': {
                'deposit_withdrawal': 0.10,  # 10% deposit withdrawal
                'asset_haircut': 0.05,        # 5% liquid asset devaluation
                'outflow_increase': 0.15,     # 15% higher 30-day outflows
                'description': 'Mild market stress'
            },
            'moderate_stress': {
                'deposit_withdrawal': 0.20,  # 20% deposit withdrawal
                'asset_haircut': 0.15,       # 15% liquid asset devaluation
                'outflow_increase': 0.30,    # 30% higher 30-day outflows
                'description': 'Moderate market stress'
            },
            'severe_stress': {
                'deposit_withdrawal': 0.30,  # 30% deposit withdrawal
                'asset_haircut': 0.25,       # 25% liquid asset devaluation
                'outflow_increase': 0.50,    # 50% higher 30-day outflows
                'description': 'Severe market crisis'
            }
        }
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """
        Apply stress scenario to data
        
        Args:
            data: Original data
            scenario: Stress scenario parameters
            
        Returns:
            Stressed data
        """
        stressed = data.copy()
        
        # Apply deposit withdrawal
        if 'total_deposits' in stressed.columns:
            stressed['total_deposits'] *= (1 - scenario.get('deposit_withdrawal', 0))
        
        # Apply asset haircut to high-quality liquid assets or liquid_assets proxy
        haircut = scenario.get('asset_haircut', 0)
        if 'high_quality_liquid_assets' in stressed.columns:
            stressed['high_quality_liquid_assets'] *= (1 - haircut)
        if 'liquid_assets' in stressed.columns:
            stressed['liquid_assets'] *= (1 - haircut)

        # Increase 30-day net cash outflows
        outflow_increase = scenario.get('outflow_increase', 0)
        if 'net_cash_outflows_30d' in stressed.columns:
            stressed['net_cash_outflows_30d'] *= (1 + outflow_increase)
        
        return stressed
    
    def _calculate_stressed_liquidity(self, stressed_data: pd.DataFrame) -> Dict:
        """
        Calculate liquidity metrics under stress
        
        Args:
            stressed_data: Data after stress scenario
            
        Returns:
            Dictionary of stressed liquidity metrics
        """
        result = {}
        
        if len(stressed_data) > 0:
            # Choose HQLA proxy
            hqla = 0
            if 'high_quality_liquid_assets' in stressed_data.columns:
                hqla = stressed_data['high_quality_liquid_assets'].iloc[-1]
            elif 'liquid_assets' in stressed_data.columns:
                hqla = stressed_data['liquid_assets'].iloc[-1]

            deposits = stressed_data['total_deposits'].iloc[-1] if 'total_deposits' in stressed_data.columns else 0
            outflows_30d = stressed_data['net_cash_outflows_30d'].iloc[-1] if 'net_cash_outflows_30d' in stressed_data.columns else max(deposits * 0.25, 1e-6)

            try:
                from config.expert_rules_config import load_expert_rules
                rules = load_expert_rules()
                coverage_min = rules.get('liquidity', {}).get('liquidity_ratio_min', 0.3)
                survival_min = rules.get('liquidity', {}).get('survival_days_min', 30)
                min_lcr = rules.get('liquidity', {}).get('min_lcr', 1.0)
            except Exception:
                coverage_min = 0.3
                survival_min = 30
                min_lcr = 1.0

            # Liquidity ratio vs deposits
            result['liquidity_ratio'] = hqla / deposits if deposits > 0 else 0
            result['liquidity_gap'] = hqla - deposits * coverage_min

            # LCR (Basel III): HQLA / 30-day net outflows
            result['lcr'] = hqla / outflows_30d if outflows_30d > 0 else 0
            result['lcr_gap'] = result['lcr'] - min_lcr

            # Estimate survival days (simplified)
            daily_outflow = outflows_30d / 30.0
            result['survival_days'] = int(hqla / daily_outflow) if daily_outflow > 0 else 999

            # Pass/fail: must meet coverage, survival, and LCR >= 100%
            result['passes'] = (
                result['liquidity_ratio'] >= coverage_min and
                result['survival_days'] >= survival_min and
                result['lcr'] >= min_lcr
            )
        else:
            result = {
                'liquidity_ratio': 0,
                'liquidity_gap': 0,
                'lcr': 0,
                'lcr_gap': -1,
                'survival_days': 0,
                'passes': False
            }
        
        return result
    
    def _get_default_thresholds(self) -> Dict:
        """
        Get default liquidity risk thresholds
        
        Returns:
            Dictionary of thresholds
        """
        try:
            from config.expert_rules_config import load_expert_rules
            rules = load_expert_rules()
            return {
                'liquidity_ratio': {'min': rules.get('liquidity', {}).get('liquidity_ratio_min', 0.30), 'severity': 'HIGH'},
                'loan_to_deposit': {'max': rules.get('liquidity', {}).get('max_loan_to_deposit', 0.85), 'severity': 'HIGH'},
                'net_stable_funding_ratio': {'min': rules.get('liquidity', {}).get('min_lcr', 1.0), 'severity': 'MEDIUM'}
            }
        except Exception:
            return {
                'liquidity_ratio': {'min': 0.30, 'severity': 'HIGH'},
                'loan_to_deposit': {'max': 0.85, 'severity': 'HIGH'},
                'net_stable_funding_ratio': {'min': 1.0, 'severity': 'MEDIUM'}
            }    
    def _calculate_peer_benchmarks(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate peer benchmarks for each feature to enable regulatory comparisons
        
        Args:
            data: Training dataset with multiple banks
            
        Returns:
            Dictionary of feature benchmarks (mean, std, percentiles)
        """
        benchmarks = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].dropna()
                if len(values) > 0:
                    benchmarks[col] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'p25': float(values.quantile(0.25)),
                        'p75': float(values.quantile(0.75)),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
        
        self.peer_benchmarks = benchmarks
        return benchmarks
    
    def _generate_liquidity_risk_explanation(self,
                                             input_data: pd.DataFrame,
                                             predictions: Dict,
                                             liquidity_risk_score: float) -> Dict:
        """
        Generate regulatory-compliant explanation for liquidity risk assessment
        
        Args:
            input_data: Input bank data
            predictions: Predictions from ML models
            liquidity_risk_score: Overall liquidity risk score
            
        Returns:
            Detailed explanation with contributing factors
        """
        explanation = {
            'key_factors': [],
            'narrative': '',
            'stress_indicators': {},
            'metric_deviations': {}
        }
        
        if len(input_data) == 0:
            return explanation
        
        record = input_data.iloc[0]
        
        # Analyze stress test results
        if self.stress_test_results:
            explanation['stress_indicators'] = {
                'stress_test_conducted': True,
                'baseline_passed': self.stress_test_results.get('baseline', {}).get('passes', False),
                'stress_passed': self.stress_test_results.get('stress', {}).get('passes', False),
                'severe_passed': self.stress_test_results.get('severe', {}).get('passes', False)
            }
        
        # Analyze anomaly detection
        if 'anomaly_detection' in predictions:
            anom = predictions['anomaly_detection']
            if anom.get('is_stress_event', False):
                explanation['stress_indicators']['ml_stress_detected'] = True
                explanation['stress_indicators']['stress_probability'] = anom.get('stress_probability', 0)
        
        # Identify key liquidity metrics and compare to benchmarks
        liquidity_metrics = ['liquidity_ratio', 'loan_to_deposit', 'lcr', 'nsfr',
                            'cash_to_assets', 'liquid_assets_to_deposits']
        
        significant_deviations = []
        
        for metric in liquidity_metrics:
            if metric in record.index and metric in self.peer_benchmarks:
                value = record[metric]
                benchmark = self.peer_benchmarks[metric]
                
                if pd.isna(value):
                    continue
                
                # Calculate deviation
                if benchmark['std'] > 0:
                    z_score = (value - benchmark['mean']) / benchmark['std']
                    pct_deviation = ((value - benchmark['mean']) / abs(benchmark['mean']) * 100) if benchmark['mean'] != 0 else 0
                    
                    # Flag problematic metrics based on domain knowledge
                    is_problematic = False
                    reason = ""
                    
                    if metric in ['liquidity_ratio', 'lcr', 'nsfr'] and value < benchmark['p25']:
                        is_problematic = True
                        reason = f"{metric.upper()} below peer 25th percentile"
                    elif metric == 'loan_to_deposit' and value > benchmark['p75']:
                        is_problematic = True
                        reason = "Loan-to-deposit ratio above peer 75th percentile"
                    elif abs(z_score) > 2:
                        is_problematic = True
                        reason = f"Extreme deviation (z-score: {z_score:.2f})"
                    
                    if is_problematic:
                        significant_deviations.append({
                            'metric': metric,
                            'value': float(value),
                            'peer_mean': benchmark['mean'],
                            'peer_median': benchmark['median'],
                            'z_score': float(z_score),
                            'pct_deviation': float(pct_deviation),
                            'reason': reason
                        })
        
        # Sort by absolute z-score
        significant_deviations.sort(key=lambda x: abs(x['z_score']), reverse=True)
        explanation['key_factors'] = significant_deviations[:5]
        explanation['metric_deviations'] = {d['metric']: d for d in significant_deviations}
        
        # Generate narrative
        narrative_parts = []
        
        risk_level_text = "CRITICAL" if liquidity_risk_score >= 0.8 else "HIGH" if liquidity_risk_score >= 0.6 else "MODERATE" if liquidity_risk_score >= 0.4 else "LOW"
        narrative_parts.append(f"Liquidity Risk Assessment: {risk_level_text} (score: {liquidity_risk_score:.2f}).")
        
        # Stress test results
        if self.stress_test_results:
            failed_scenarios = []
            if not self.stress_test_results.get('baseline', {}).get('passes', True):
                failed_scenarios.append("baseline")
            if not self.stress_test_results.get('stress', {}).get('passes', True):
                failed_scenarios.append("stress")
            if not self.stress_test_results.get('severe', {}).get('passes', True):
                failed_scenarios.append("severe")
            
            if failed_scenarios:
                narrative_parts.append(f"Failed stress scenarios: {', '.join(failed_scenarios)}.")
                
                # Add specific details from severe scenario
                if 'severe' in failed_scenarios and 'severe' in self.stress_test_results:
                    severe = self.stress_test_results['severe']
                    if 'lcr' in severe and severe['lcr'] < 1.0:
                        narrative_parts.append(
                            f"Under severe stress, LCR drops to {severe['lcr']:.2f} "
                            f"(below regulatory minimum of 1.0)."
                        )
                    if 'survival_days' in severe and severe['survival_days'] < 30:
                        narrative_parts.append(
                            f"Estimated survival: {severe['survival_days']} days under severe stress "
                            f"(below 30-day minimum)."
                        )
        
        # Anomaly detection insights
        if 'anomaly_detection' in predictions:
            anom = predictions['anomaly_detection']
            if anom.get('is_stress_event', False):
                narrative_parts.append(
                    f"ML models detected liquidity stress patterns "
                    f"(stress probability: {anom.get('stress_probability', 0):.1%})."
                )
        
        # Metric deviations
        if significant_deviations:
            top_3 = significant_deviations[:3]
            concern_desc = []
            
            for dev in top_3:
                metric_display = dev['metric'].replace('_', ' ').upper()
                direction = "below" if dev['pct_deviation'] < 0 else "above"
                
                concern_desc.append(
                    f"{metric_display}: {dev['value']:.4f} vs peer average {dev['peer_mean']:.4f} "
                    f"({direction} by {abs(dev['pct_deviation']):.1f}% - {dev['reason']})"
                )
            
            narrative_parts.append(
                f"Key concerns: {'; '.join(concern_desc)}."
            )
        
        # Add indicator-based insights
        if self.liquidity_indicators:
            if 'liquidity_ratio' in self.liquidity_indicators:
                liq_ratio = self.liquidity_indicators['liquidity_ratio']
                if liq_ratio < 0.30:
                    narrative_parts.append(f"Liquidity ratio ({liq_ratio:.2%}) below 30% prudential minimum.")
            
            if 'lcr' in self.liquidity_indicators:
                lcr = self.liquidity_indicators['lcr']
                if lcr < 1.0:
                    narrative_parts.append(f"LCR ({lcr:.2f}) below Basel III requirement of 100%.")
        
        if not narrative_parts:
            narrative_parts.append("Liquidity risk assessment completed based on model analysis.")
        
        explanation['narrative'] = ' '.join(narrative_parts)
        
        return explanation