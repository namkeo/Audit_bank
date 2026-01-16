# -*- coding: utf-8 -*-
"""
Base Risk Model Class
Provides common functionality for all risk models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import importlib.util
from enum import Enum
from functools import wraps

# Import logging configuration
try:
    def _import_logging():
        spec = importlib.util.spec_from_file_location("logging_config", "6_logging_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (module.AuditLogger, module.ModelTrainingError, 
                module.RiskAssessmentError, module.ModelNotTrainedError, module.log_exception)
    
    AuditLogger, ModelTrainingError, RiskAssessmentError, ModelNotTrainedError, log_exception = _import_logging()
except Exception:
    # Fallback to basic logging
    import logging
    class AuditLogger:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)
    class ModelTrainingError(Exception):
        pass
    class RiskAssessmentError(Exception):
        pass
    class ModelNotTrainedError(Exception):
        pass
    def log_exception(logger, exc, context=None):
        logger.error(f"{type(exc).__name__}: {str(exc)}")


class ModelState(Enum):
    """Represents the state of a model in its lifecycle"""
    UNINITIALIZED = "uninitialized"  # Just created, no training
    TRAINED = "trained"              # Models trained, ready for predictions
    READY = "ready"                  # Fully validated and ready


def require_trained(method):
    """Decorator to ensure model is trained before calling method"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_trained():
            raise ModelNotTrainedError(
                f"{self.model_name} must be trained before calling {method.__name__}(). "
                f"Call train_models() first.",
                context={'model': self.model_name, 'method': method.__name__, 'state': self._state.value}
            )
        return method(self, *args, **kwargs)
    return wrapper


class BaseRiskModel(ABC):
    """
    Abstract base class for all risk models
    Provides common functionality and enforces interface
    
    LIFECYCLE:
        1. Initialize: model = ModelClass()
        2. Train: model.train_models(training_data)  [REQUIRED before predictions]
        3. Predict: results = model.predict_risk(test_data)
        
    ALTERNATIVE (Factory Method):
        model = ModelClass.create_and_train(training_data)
        results = model.predict_risk(test_data)
    
    STATE MANAGEMENT:
        - UNINITIALIZED: Just created, no models trained
        - TRAINED: Models trained, ready for predictions
        - READY: Fully validated and production-ready
    """
    
    def __init__(self, model_name: str):
        """
        Initialize base risk model
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.models = {}
        self.scaler = StandardScaler()
        self.risk_scores = {}
        self.logger = AuditLogger.get_logger(__name__)
        self.risk_indicators = {}
        self.training_history = []
        
        # State management
        self._state = ModelState.UNINITIALIZED
        self.feature_names = None
        
        self.logger.info(f"Initialized {model_name} in state: {self._state.value}")
    
    def is_trained(self) -> bool:
        """
        Check if model has been trained
        
        Returns:
            True if model is trained and ready for predictions
        """
        return (len(self.models) > 0 and 
                self.feature_names is not None and 
                self._state in [ModelState.TRAINED, ModelState.READY])
    
    def get_state(self) -> ModelState:
        """Get current model state"""
        return self._state
    
    def _mark_as_trained(self) -> None:
        """Mark model as trained (call after successful training)"""
        self._state = ModelState.TRAINED
        self.logger.info(f"{self.model_name} state changed to: {self._state.value}")
    
    def validate_for_prediction(self) -> None:
        """
        Validate that model is ready for predictions
        
        Raises:
            ModelNotTrainedError: If model is not properly trained
        """
        if not self.is_trained():
            raise ModelNotTrainedError(
                f"{self.model_name} cannot make predictions - model not trained",
                context={
                    'model': self.model_name,
                    'state': self._state.value,
                    'has_models': len(self.models) > 0,
                    'has_features': self.feature_names is not None
                }
            )
        
    @classmethod
    def create_and_train(cls, training_data: pd.DataFrame, **kwargs):
        """
        Factory method: Create and train model in one call
        
        Args:
            training_data: Data to train on
            **kwargs: Additional parameters for train_models()
            
        Returns:
            Trained model instance ready for predictions
            
        Example:
            >>> model = CreditRiskModel.create_and_train(training_df)
            >>> predictions = model.predict_risk(test_df)
        """
        instance = cls()
        instance.train_models(training_data, **kwargs)
        return instance
        
    @abstractmethod
    def train_models(self, training_data: pd.DataFrame, **kwargs) -> Dict:
        """
        Train risk assessment models
        Must be implemented by subclasses
        
        Args:
            training_data: Training dataset
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training results
        """
        pass
    
    @abstractmethod
    def predict_risk(self, input_data: pd.DataFrame) -> Dict:
        """
        Predict risk for given data
        Must be implemented by subclasses
        
        Args:
            input_data: Data to assess
            
        Returns:
            Dictionary containing risk predictions
        """
        pass
    
    @abstractmethod
    def assess_risk_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Assess specific risk indicators
        Must be implemented by subclasses
        
        Args:
            data: Data to assess
            
        Returns:
            Dictionary of risk indicators
        """
        pass
    
    def calculate_risk_score(self, indicators: Dict, weights: Optional[Dict] = None) -> float:
        """
        Calculate overall risk score from indicators
        
        Args:
            indicators: Dictionary of risk indicators
            weights: Optional weights for each indicator
            
        Returns:
            Weighted risk score (0-100)
        """
        if not indicators:
            return 0.0
            
        if weights is None:
            # Equal weights if not provided
            weights = {key: 1.0 / len(indicators) for key in indicators.keys()}
        
        total_score = 0.0
        total_weight = 0.0
        
        for indicator, value in indicators.items():
            weight = weights.get(indicator, 0.0)
            
            # Normalize value to 0-100 scale if needed
            normalized_value = self._normalize_indicator(indicator, value)
            
            total_score += normalized_value * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def _normalize_indicator(self, indicator_name: str, value: float) -> float:
        """
        Normalize indicator value to 0-100 scale
        
        Args:
            indicator_name: Name of the indicator
            value: Raw indicator value
            
        Returns:
            Normalized value (0-100)
        """
        # Default normalization - can be overridden by subclasses
        if isinstance(value, (int, float)):
            # Clip to reasonable range
            return np.clip(value * 100, 0, 100)
        return 0.0
    
    def classify_risk_level(self, risk_score: float) -> str:
        """
        Classify risk level based on score
        
        Args:
            risk_score: Numerical risk score (0-100)
            
        Returns:
            Risk level classification
        """
        if risk_score >= 75:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
    
    def detect_early_warning_signals(self, data: pd.DataFrame, 
                                    thresholds: Optional[Dict] = None) -> List[Dict]:
        """
        Detect early warning signals in the data
        
        Args:
            data: Input data to analyze
            thresholds: Optional custom thresholds
            
        Returns:
            List of detected warning signals
        """
        warnings = []
        
        if thresholds is None:
            thresholds = self._get_default_thresholds()
        
        # Check each metric against thresholds
        for metric, threshold in thresholds.items():
            if metric in data.columns:
                violations = self._check_threshold_violations(
                    data[metric], 
                    threshold
                )
                if violations:
                    warnings.extend(violations)
        
        return warnings
    
    def _get_default_thresholds(self) -> Dict:
        """
        Get default threshold values
        Can be overridden by subclasses
        
        Returns:
            Dictionary of default thresholds
        """
        return {}
    
    def _check_threshold_violations(self, values: pd.Series, 
                                   threshold: Dict) -> List[Dict]:
        """
        Check for threshold violations
        
        Args:
            values: Series of values to check
            threshold: Threshold configuration
            
        Returns:
            List of violations
        """
        violations = []
        
        if 'min' in threshold:
            below_min = values < threshold['min']
            if below_min.any():
                violations.append({
                    'type': 'below_minimum',
                    'metric': values.name,
                    'threshold': threshold['min'],
                    'count': below_min.sum(),
                    'severity': threshold.get('severity', 'MEDIUM')
                })
        
        if 'max' in threshold:
            above_max = values > threshold['max']
            if above_max.any():
                violations.append({
                    'type': 'above_maximum',
                    'metric': values.name,
                    'threshold': threshold['max'],
                    'count': above_max.sum(),
                    'severity': threshold.get('severity', 'MEDIUM')
                })
        
        return violations
    
    def validate_model_inputs(self, data: pd.DataFrame, 
                            required_columns: List[str]) -> bool:
        """
        Validate that required columns exist in data
        
        Args:
            data: Input dataframe
            required_columns: List of required column names
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns for {self.model_name}: {missing_cols}"
            )
        
        # Check for sufficient data
        if len(data) == 0:
            raise ValueError(f"No data provided for {self.model_name}")
        
        return True
    
    def get_feature_importance(self, model_type: str = 'default') -> Optional[Dict]:
        """
        Get feature importance from trained models
        
        Args:
            model_type: Type of model to get importance from
            
        Returns:
            Dictionary of feature importances or None
        """
        if model_type not in self.models:
            return None
        
        model = self.models[model_type]
        
        # Try to get feature importance (works for tree-based models)
        if hasattr(model, 'feature_importances_'):
            return dict(zip(
                self.feature_names if hasattr(self, 'feature_names') else range(len(model.feature_importances_)),
                model.feature_importances_
            ))
        
        # For linear models, use coefficients
        if hasattr(model, 'coef_'):
            return dict(zip(
                self.feature_names if hasattr(self, 'feature_names') else range(len(model.coef_)),
                np.abs(model.coef_)
            ))
        
        return None
    
    def save_model(self, filepath: str, model_type: str = 'all'):
        """
        Save trained models to disk
        
        Args:
            filepath: Path to save models
            model_type: Which model to save ('all' or specific type)
        """
        import joblib
        
        if model_type == 'all':
            joblib.dump(self.models, filepath)
        elif model_type in self.models:
            joblib.dump(self.models[model_type], filepath)
        else:
            raise ValueError(f"Model type '{model_type}' not found")
    
    def load_model(self, filepath: str, model_type: str = 'all'):
        """
        Load trained models from disk
        
        Args:
            filepath: Path to load models from
            model_type: Which model to load
        """
        import joblib
        
        loaded = joblib.load(filepath)
        
        if model_type == 'all':
            self.models = loaded
        else:
            self.models[model_type] = loaded
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the model
        
        Returns:
            Dictionary containing model statistics
        """
        return {
            'model_name': self.model_name,
            'models_trained': list(self.models.keys()),
            'risk_scores_count': len(self.risk_scores),
            'indicators_tracked': len(self.risk_indicators),
            'training_iterations': len(self.training_history)
        }
    
    def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report
        
        Returns:
            Dictionary containing risk assessment report
        """
        return {
            'model_name': self.model_name,
            'risk_scores': self.risk_scores,
            'risk_indicators': self.risk_indicators,
            'summary': self.get_summary_statistics(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
