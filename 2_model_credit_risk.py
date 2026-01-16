# -*- coding: utf-8 -*-
"""
Mô-đun Đánh giá Rủi ro Tín dụng
Xử lý mô hình hóa và dự đoán rủi ro tín dụng
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Dict, List, Optional, Tuple
import xgboost as xgb

import importlib.util
import logging

# Suppress sklearn robust covariance warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Determinant has increased')
warnings.filterwarnings('ignore', category=UserWarning, message='The covariance matrix associated to your dataset is not full rank')

# Hàm trợ giúp để nhập mô-đun có tiền tố số
def _import_module(module_path, class_name):
    """Nhập mô-đun theo đường dẫn tệp và trả về lớp được chỉ định"""
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

BaseRiskModel = _import_module("2_model_base_risk.py", "BaseRiskModel")
require_trained = _import_module("2_model_base_risk.py", "require_trained")


class CreditRiskModel(BaseRiskModel):
    """
    Mô hình Đánh giá Rủi ro Tín dụng
    Đánh giá rủi ro tín dụng sử dụng nhiều mô hình ML và chỉ báo tài chính
    
    VÒNG ĐỜI:
        1. Khởi tạo: model = CreditRiskModel()
        2. Huấn luyện: model.train_models(training_data)  [BẮT BUỘC]
        3. Dự đoán: results = model.predict_risk(test_data)
        
    PHƯƠNG THỨC FACTORY (Khuyến nghị):
        model = CreditRiskModel.create_and_train(training_data)
        results = model.predict_risk(test_data)
    """
    
    def __init__(self):
        super().__init__("Credit Risk Model")
        self.credit_indicators = {}
        self.npl_prediction = None
        self.peer_benchmarks = {}  # Store peer comparison data for explanations
        try:
            from config.expert_rules_config import load_expert_rules
            self.expert_rules = load_expert_rules()
        except Exception:
            self.expert_rules = {}
        
    def train_models(self, training_data: pd.DataFrame, 
                    labels: Optional[pd.Series] = None,
                    **kwargs) -> Dict:
        """
        Đào tạo các mô hình rủi ro tín dụng trên dữ liệu lịch sử
        
        Args:
            training_data (pd.DataFrame): Dữ liệu huấn luyện chứa các chỉ báo tài chính
            labels (Optional[pd.Series]): Nhãn mục tiêu (0: tốt, 1: xấu) - tùy chọn
            **kwargs: Các tham số bổ sung cho các mô hình
            
        Returns:
            dict: Từ điển chứa kết quả huấn luyện bao gồm mô hình, chính xác và thông tin trạng thái
            
        Raises:
            ValueError: Nếu dữ liệu huấn luyện rỗng hoặc không có đủ đặc trưng
            RuntimeError: Nếu quá trình huấn luyện thất bại
        """
        # Prepare features
        features = self._prepare_credit_features(training_data)
        
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
        
        # If labels provided, train supervised models
        if labels is not None:
            training_results['supervised'] = self._train_supervised_models(
                X_scaled, labels
            )
        
        # Train unsupervised models for anomaly detection
        training_results['unsupervised'] = self._train_unsupervised_models(
            X_scaled
        )
        
        # Mark as trained
        self._mark_as_trained()
        
        # Record training history
        self.training_history.append({
            'timestamp': pd.Timestamp.now(),
            'samples': len(features),
            'features': len(self.feature_names),
            'results': training_results
        })
        
        return training_results
    
    def _prepare_credit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare credit-specific features
        
        Args:
            data: Input dataframe
            
        Returns:
            DataFrame with credit risk features
        """
        # Collect all columns at once to avoid fragmentation
        cols_to_add = []
        
        # Asset Quality Features
        if 'npl_ratio' in data.columns:
            cols_to_add.append('npl_ratio')
        
        if 'loan_loss_coverage' in data.columns:
            cols_to_add.append('loan_loss_coverage')
        
        # Capital Adequacy Features
        if 'capital_adequacy_ratio' in data.columns:
            cols_to_add.append('capital_adequacy_ratio')
        
        if 'equity_to_assets' in data.columns:
            cols_to_add.append('equity_to_assets')
        
        # Profitability Features (impacts ability to absorb losses)
        if 'roa' in data.columns:
            cols_to_add.append('roa')
        
        if 'roe' in data.columns:
            cols_to_add.append('roe')
        
        # Loan Portfolio Features
        if 'loan_to_deposit' in data.columns:
            cols_to_add.append('loan_to_deposit')
        
        # Growth Features (rapid growth can indicate higher risk)
        growth_cols = [col for col in data.columns if '_growth' in col and 'loan' in col]
        cols_to_add.extend(growth_cols)
        
        # Create features DataFrame all at once
        features = data[cols_to_add].copy() if cols_to_add else pd.DataFrame()
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def _train_supervised_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train supervised credit risk models
        
        Args:
            X: Feature matrix
            y: Labels (0: good, 1: bad credit)
            
        Returns:
            Dictionary of training results
        """
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        results['random_forest'] = {
            'train_score': rf_model.score(X_train, y_train),
            'test_score': rf_model.score(X_test, y_test),
            'cv_score': cross_val_score(rf_model, X, y, cv=5, random_state=42).mean()
        }
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        results['gradient_boosting'] = {
            'train_score': gb_model.score(X_train, y_train),
            'test_score': gb_model.score(X_test, y_test),
            'cv_score': cross_val_score(gb_model, X, y, cv=5, random_state=42).mean()
        }
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        results['xgboost'] = {
            'train_score': xgb_model.score(X_train, y_train),
            'test_score': xgb_model.score(X_test, y_test)
        }
        
        return results
    
    def _train_unsupervised_models(self, X: np.ndarray) -> Dict:
        """
        Train unsupervised models for credit risk anomaly detection
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of training results
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import EllipticEnvelope
        
        results = {}
        
        # Isolation Forest for outlier detection
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        iso_forest.fit(X)
        self.models['isolation_forest'] = iso_forest
        results['isolation_forest'] = {'trained': True}
        
        # Elliptic Envelope
        try:
            elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
            elliptic.fit(X)
            self.models['elliptic_envelope'] = elliptic
            results['elliptic_envelope'] = {'trained': True}
        except:
            results['elliptic_envelope'] = {'trained': False, 'error': 'Singular matrix'}
        
        return results
    
    @require_trained
    def predict_risk(self, input_data: pd.DataFrame) -> Dict:
        """
        Predict credit risk for given bank data
        
        Args:
            input_data: Current bank financial data
            
        Returns:
            Dictionary containing risk predictions and scores
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        # Prepare features
        features = self._prepare_credit_features(input_data)
        
        if features.empty:
            return {'status': 'failed', 'reason': 'No valid features'}
        
        # Align feature columns with those used during training
        if hasattr(self, 'feature_names') and self.feature_names:
            # Reindex to training feature names, fill missing with 0 and drop extras
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
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # For classification models
                    proba = model.predict_proba(X_scaled)
                    predictions[model_name] = {
                        'risk_probability': proba[:, 1].mean() if proba.shape[1] > 1 else 0,
                        'prediction': model.predict(X_scaled)[0]
                    }
                elif hasattr(model, 'decision_function'):
                    # For anomaly detection models
                    scores = model.decision_function(X_scaled)
                    predictions[model_name] = {
                        'anomaly_score': -scores.mean(),  # Negative for anomaly
                        'is_anomaly': (scores < 0).any()
                    }
            except Exception as e:
                predictions[model_name] = {'error': str(e)}
        
        # Calculate overall credit risk score
        credit_risk_score = self._calculate_comprehensive_credit_risk(
            input_data, predictions
        )
        
        # Generate regulatory explanation
        explanation = self._generate_credit_risk_explanation(
            input_data, predictions, credit_risk_score
        )
        
        self.risk_scores['credit'] = credit_risk_score
        
        return {
            'credit_risk_score': credit_risk_score,
            'risk_level': self.classify_risk_level(credit_risk_score),
            'model_predictions': predictions,
            'indicators': self.credit_indicators,
            'explanation': explanation,  # NEW: Regulatory transparency
            'regulatory_narrative': explanation.get('narrative', '')
        }
    
    def apply_macro_adjustments(self, 
                              credit_risk_score: float,
                              bank_data: Dict[str, float],
                              all_banks_data: pd.DataFrame,
                              systemic_stress_level: Optional[float] = None) -> Dict:
        """
        Apply macro-adjustments to credit risk score based on industry context.
        
        Args:
            credit_risk_score: Original credit risk score
            bank_data: Bank's key metrics (npl_ratio, capital_adequacy_ratio, etc.)
            all_banks_data: DataFrame with all banks for benchmarking
            systemic_stress_level: Optional pre-calculated stress level (0-1)
            
        Returns:
            Dictionary with adjusted score and explanation
        """
        try:
            from macro_adjustments import MacroAdjustmentCalculator
            
            calc = MacroAdjustmentCalculator()
            
            # Define stress metrics
            stress_metrics = ['npl_ratio', 'loan_to_deposit_ratio']
            thresholds = {
                'npl_ratio': ('>', 0.05),
                'loan_to_deposit_ratio': ('>', 0.95)
            }
            
            # Calculate systemic stress if not provided
            if systemic_stress_level is None:
                systemic_stress_level = calc.estimate_systemic_stress(
                    all_banks_data,
                    stress_metrics,
                    thresholds
                )
            
            # Calculate benchmarks for key credit metrics
            credit_metrics = ['npl_ratio', 'capital_adequacy_ratio', 'loan_to_deposit_ratio']
            benchmarks = calc.calculate_industry_benchmarks(
                all_banks_data,
                credit_metrics
            )
            
            # Calculate relative z-scores for key metrics
            z_score = 0
            if 'npl_ratio' in benchmarks and 'npl_ratio' in bank_data:
                benchmark = benchmarks['npl_ratio']
                z_score = calc.calculate_relative_deviation(
                    bank_data['npl_ratio'],
                    benchmark['mean'],
                    benchmark['std']
                )
            
            # Apply adjustment
            adjustment = calc.adjust_risk_score(
                credit_risk_score,
                z_score,
                systemic_stress_level,
                adjustment_strength=0.4
            )
            
            return {
                'original_score': float(credit_risk_score),
                'adjusted_score': adjustment['adjusted_score'],
                'adjustment_delta': adjustment['adjustment_delta'],
                'systemic_stress_level': systemic_stress_level,
                'relative_z_score': z_score,
                'adjustment_reason': adjustment['adjustment_reason'],
                'adjustment_confidence': adjustment['adjustment_confidence'],
                'benchmarks': benchmarks
            }
        
        except ImportError:
            self.logger.warning("macro_adjustments module not available")
            return {
                'original_score': float(credit_risk_score),
                'adjusted_score': float(credit_risk_score),
                'adjustment_delta': 0.0,
                'adjustment_reason': 'Macro-adjustments not available'
            }
    
    
    def _calculate_comprehensive_credit_risk(self, data: pd.DataFrame, 
                                           predictions: Dict) -> float:
        """
        Calculate comprehensive credit risk score
        
        Args:
            data: Input data
            predictions: Model predictions
            
        Returns:
            Credit risk score (0-100)
        """
        risk_components = []
        
        # Component 1: ML Model predictions (40% weight)
        ml_score = 0
        ml_count = 0
        for model_name, pred in predictions.items():
            if 'risk_probability' in pred:
                ml_score += pred['risk_probability'] * 100
                ml_count += 1
            elif 'anomaly_score' in pred:
                ml_score += min(pred['anomaly_score'] * 10, 100)
                ml_count += 1
        
        if ml_count > 0:
            risk_components.append(('ml_models', ml_score / ml_count, 0.4))
        
        # Component 2: NPL Ratio (30% weight)
        if 'npl_ratio' in data.columns:
            npl_ratio = data['npl_ratio'].mean()
            # Convert to risk score (higher NPL = higher risk)
            npl_risk = min(npl_ratio * 500, 100)  # 20% NPL = 100 risk
            risk_components.append(('npl', npl_risk, 0.3))
        
        # Component 3: Capital Adequacy (20% weight)
        if 'capital_adequacy_ratio' in data.columns:
            car = data['capital_adequacy_ratio'].mean()
            # Lower CAR = higher risk (invert)
            car_risk = max(100 - car * 10, 0)  # 10% CAR = 0 risk
            risk_components.append(('capital', car_risk, 0.2))
        
        # Component 4: Loan Growth (10% weight)
        growth_cols = [col for col in data.columns if 'loan' in col and '_growth' in col]
        if growth_cols:
            loan_growth = data[growth_cols[0]].mean()
            # Extreme growth = higher risk
            growth_risk = min(abs(loan_growth) * 100, 100)
            risk_components.append(('growth', growth_risk, 0.1))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in risk_components)
        total_weight = sum(weight for _, _, weight in risk_components)
        
        if total_weight > 0:
            return total_score / total_weight
        return 50.0  # Default medium risk if no components
    
    @require_trained
    def assess_risk_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Assess specific credit risk indicators
        
        Args:
            data: Bank financial data
            
        Returns:
            Dictionary of credit risk indicators
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        indicators = {}
        
        # Asset Quality Indicators
        if 'npl_ratio' in data.columns:
            npl_ratio = data['npl_ratio'].iloc[-1] if len(data) > 0 else 0
            max_npl = self.expert_rules.get('asset_quality', {}).get('max_npl_ratio', 0.05)
            indicators['npl_ratio'] = {
                'value': npl_ratio,
                'status': 'HIGH_RISK' if npl_ratio > max_npl else 'ACCEPTABLE',
                'threshold': max_npl,
                'description': 'Non-Performing Loan Ratio'
            }
        
        # Provision coverage ratio (Reserves/NPL)
        min_cov = self.expert_rules.get('asset_quality', {}).get('min_coverage_ratio', 0.7)
        if 'provision_coverage_ratio' in data.columns:
            coverage = data['provision_coverage_ratio'].iloc[-1] if len(data) > 0 else 0
            indicators['provision_coverage_ratio'] = {
                'value': coverage,
                'status': 'LOW' if coverage < min_cov else 'ADEQUATE',
                'threshold': min_cov,
                'description': 'Provision Coverage Ratio (Reserves/NPL)'
            }
        elif 'loan_loss_coverage' in data.columns:
            coverage = data['loan_loss_coverage'].iloc[-1] if len(data) > 0 else 0
            indicators['loan_loss_coverage'] = {
                'value': coverage,
                'status': 'LOW' if coverage < min_cov else 'ADEQUATE',
                'threshold': min_cov,
                'description': 'Loan Loss Reserve Coverage'
            }
        
        # Capital Adequacy
        if 'capital_adequacy_ratio' in data.columns:
            car = data['capital_adequacy_ratio'].iloc[-1] if len(data) > 0 else 0
            min_car = self.expert_rules.get('capital_adequacy', {}).get('min_car', 0.08)
            indicators['capital_adequacy_ratio'] = {
                'value': car,
                'status': 'LOW' if car < min_car else 'ADEQUATE',
                'threshold': min_car,
                'description': 'Capital Adequacy Ratio (Basel)'
            }
        
        # Concentration Risk
        if 'loan_to_deposit' in data.columns:
            ltd = data['loan_to_deposit'].iloc[-1] if len(data) > 0 else 0
            max_ltd = self.expert_rules.get('liquidity', {}).get('max_loan_to_deposit', 0.9)
            indicators['loan_to_deposit'] = {
                'value': ltd,
                'status': 'HIGH' if ltd > max_ltd else 'NORMAL',
                'threshold': max_ltd,
                'description': 'Loan to Deposit Ratio'
            }
        
        # Portfolio diversification & exposures
        if 'sector_concentration_hhi' in data.columns:
            hhi = data['sector_concentration_hhi'].iloc[-1] if len(data) > 0 else np.nan
            max_hhi = self.expert_rules.get('concentration', {}).get('max_sector_hhi', 0.20)
            indicators['sector_concentration_hhi'] = {
                'value': hhi,
                'status': 'HIGH' if (pd.notna(hhi) and hhi > max_hhi) else 'NORMAL',
                'threshold': max_hhi,
                'description': 'Sector Concentration (HHI)'
            }
        if 'top_borrower_concentration' in data.columns:
            tbr = data['top_borrower_concentration'].iloc[-1] if len(data) > 0 else np.nan
            max_tbr = self.expert_rules.get('concentration', {}).get('max_top_borrower_ratio', 0.10)
            indicators['top_borrower_concentration'] = {
                'value': tbr,
                'status': 'HIGH' if (pd.notna(tbr) and tbr > max_tbr) else 'NORMAL',
                'threshold': max_tbr,
                'description': 'Largest Borrower Exposure / Loans'
            }
        if 'top10_borrower_concentration' in data.columns:
            t10 = data['top10_borrower_concentration'].iloc[-1] if len(data) > 0 else np.nan
            max_t10 = self.expert_rules.get('concentration', {}).get('max_top10_borrower_ratio', 0.35)
            indicators['top10_borrower_concentration'] = {
                'value': t10,
                'status': 'HIGH' if (pd.notna(t10) and t10 > max_t10) else 'NORMAL',
                'threshold': max_t10,
                'description': 'Top 10 Borrowers Exposure / Loans'
            }
        if 'obs_to_loans_ratio' in data.columns:
            obs = data['obs_to_loans_ratio'].iloc[-1] if len(data) > 0 else np.nan
            max_obs = self.expert_rules.get('off_balance', {}).get('max_obs_to_loans_ratio', 0.25)
            indicators['obs_to_loans_ratio'] = {
                'value': obs,
                'status': 'HIGH' if (pd.notna(obs) and obs > max_obs) else 'NORMAL',
                'threshold': max_obs,
                'description': 'Off-Balance Sheet Exposure / Loans'
            }
        if 'external_rating_score' in data.columns:
            score = data['external_rating_score'].iloc[-1] if len(data) > 0 else np.nan
            min_notch = self.expert_rules.get('external', {}).get('min_rating_notch', 5)
            indicators['external_credit_rating'] = {
                'value': score,
                'status': 'LOW' if (pd.notna(score) and score <= min_notch) else 'ELEVATED',
                'threshold': min_notch,
                'description': 'External Credit Rating (lower is better)'
            }
        
        self.credit_indicators = indicators
        return indicators
    
    def predict_npl_trend(self, time_series_data: pd.DataFrame) -> Dict:
        """
        Predict future NPL trends
        
        Args:
            time_series_data: Historical NPL data
            
        Returns:
            Dictionary containing NPL predictions
        """
        if 'npl_ratio' not in time_series_data.columns:
            return {'status': 'failed', 'reason': 'No NPL data available'}
        
        npl_values = time_series_data['npl_ratio'].values
        
        if len(npl_values) < 3:
            return {'status': 'failed', 'reason': 'Insufficient historical data'}
        
        # Simple linear trend prediction
        x = np.arange(len(npl_values))
        coeffs = np.polyfit(x, npl_values, 1)
        
        # Predict next 3 periods
        future_x = np.arange(len(npl_values), len(npl_values) + 3)
        predictions = np.polyval(coeffs, future_x)
        
        trend = 'increasing' if coeffs[0] > 0 else 'decreasing'
        
        self.npl_prediction = {
            'current_npl': npl_values[-1],
            'predicted_npl': predictions.tolist(),
            'trend': trend,
            'trend_slope': coeffs[0],
            'risk_level': 'HIGH' if trend == 'increasing' and npl_values[-1] > 0.05 else 'MODERATE'
        }
        
        return self.npl_prediction
    
    def _get_default_thresholds(self) -> Dict:
        """
        Get default credit risk thresholds
        
        Returns:
            Dictionary of thresholds
        """
        return {
            'npl_ratio': {'max': 0.05, 'severity': 'HIGH'},
            'capital_adequacy_ratio': {'min': 0.08, 'severity': 'HIGH'},
            'loan_to_deposit': {'max': 0.95, 'severity': 'MEDIUM'},
            'loan_loss_coverage': {'min': 0.70, 'severity': 'MEDIUM'}
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
    
    def _generate_credit_risk_explanation(self, 
                                          input_data: pd.DataFrame,
                                          model_predictions: Dict,
                                          credit_risk_score: float) -> Dict:
        """
        Generate regulatory-compliant explanation for credit risk assessment
        
        Args:
            input_data: Input bank data
            model_predictions: Predictions from ML models
            credit_risk_score: Overall credit risk score
            
        Returns:
            Detailed explanation with contributing factors
        """
        explanation = {
            'key_factors': [],
            'narrative': '',
            'model_insights': {},
            'metric_deviations': {}
        }
        
        if len(input_data) == 0:
            return explanation
        
        record = input_data.iloc[0]
        
        # Analyze model predictions
        high_risk_models = []
        for model_name, pred in model_predictions.items():
            if 'risk_probability' in pred and pred['risk_probability'] > 0.6:
                high_risk_models.append(model_name)
                explanation['model_insights'][model_name] = {
                    'probability': pred['risk_probability'],
                    'verdict': 'HIGH RISK'
                }
            elif 'is_anomaly' in pred and pred['is_anomaly']:
                high_risk_models.append(model_name)
                explanation['model_insights'][model_name] = {
                    'verdict': 'ANOMALY DETECTED'
                }
        
        # Identify key credit metrics and compare to benchmarks
        credit_metrics = ['npl_ratio', 'capital_adequacy_ratio', 'loan_to_deposit', 
                         'loan_loss_coverage', 'roa', 'roe', 'liquidity_ratio']
        
        significant_deviations = []
        
        for metric in credit_metrics:
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
                    
                    if metric == 'npl_ratio' and value > benchmark['p75']:
                        is_problematic = True
                        reason = "NPL ratio above peer 75th percentile"
                    elif metric == 'capital_adequacy_ratio' and value < benchmark['p25']:
                        is_problematic = True
                        reason = "Capital adequacy below peer 25th percentile"
                    elif metric == 'loan_loss_coverage' and value < benchmark['p25']:
                        is_problematic = True
                        reason = "Loan loss coverage below peer 25th percentile"
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
        
        risk_level_text = "CRITICAL" if credit_risk_score >= 0.8 else "HIGH" if credit_risk_score >= 0.6 else "MODERATE" if credit_risk_score >= 0.4 else "LOW"
        narrative_parts.append(f"Credit Risk Assessment: {risk_level_text} (score: {credit_risk_score:.2f}).")
        
        if high_risk_models:
            model_names = ', '.join(high_risk_models)
            narrative_parts.append(f"Models flagging elevated risk: {model_names}.")
        
        if significant_deviations:
            top_3 = significant_deviations[:3]
            concern_desc = []
            
            for dev in top_3:
                metric_display = dev['metric'].replace('_', ' ').title()
                direction = "above" if dev['pct_deviation'] > 0 else "below"
                
                concern_desc.append(
                    f"{metric_display}: {dev['value']:.4f} vs peer average {dev['peer_mean']:.4f} "
                    f"({direction} by {abs(dev['pct_deviation']):.1f}% - {dev['reason']})"
                )
            
            narrative_parts.append(
                f"Key concerns: {'; '.join(concern_desc)}."
            )
        
        # Add indicator-based insights
        if self.credit_indicators:
            if 'npl_ratio' in self.credit_indicators:
                npl = self.credit_indicators['npl_ratio']
                if npl > 0.05:
                    narrative_parts.append(f"NPL ratio ({npl:.2%}) exceeds 5% regulatory threshold.")
            
            if 'capital_adequacy_ratio' in self.credit_indicators:
                car = self.credit_indicators['capital_adequacy_ratio']
                if car < 0.08:
                    narrative_parts.append(f"Capital adequacy ratio ({car:.2%}) below 8% Basel minimum.")
        
        if not narrative_parts:
            narrative_parts.append("Credit risk assessment completed based on ensemble model analysis.")
        
        explanation['narrative'] = ' '.join(narrative_parts)
        
        return explanation
    
    def _initialize_logger(self):
        """Initialize logger"""
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(self.__class__.__name__)
