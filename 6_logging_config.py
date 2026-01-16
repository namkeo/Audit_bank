# -*- coding: utf-8 -*-
"""
Mô-đun Cấu hình Ghi nhật ký
Cung cấp ghi nhật ký có cấu trúc cho hệ thống kiểm toán ngân hàng
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings


class AuditLogger:
    """
    Cấu hình ghi nhật ký tập trung cho hệ thống kiểm toán
    Cung cấp ghi nhật ký có cấu trúc với xử lý lỗi nhận biết bối cảnh
    """
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def setup_logging(cls, 
                     log_level: str = "INFO",
                     log_file: Optional[str] = None,
                     console_output: bool = True):
        """
        Configure logging for the entire application
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
            console_output: Whether to output to console
        """
        if cls._configured:
            return
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not create log file {log_file}: {e}")
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.setup_logging()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]


class AuditException(Exception):
    """Base exception for audit system errors"""
    def __init__(self, message: str, context: Optional[dict] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class DataLoadError(AuditException):
    """Raised when data loading fails"""
    pass


class DataValidationError(AuditException):
    """Raised when data validation fails"""
    pass


class ModelTrainingError(AuditException):
    """Raised when model training fails"""
    pass


class FeaturePreparationError(AuditException):
    """Raised when feature preparation fails"""
    pass


class RiskAssessmentError(AuditException):
    """Raised when risk assessment fails"""
    pass


class ModelNotTrainedError(AuditException):
    """Raised when attempting to use an untrained model"""
    pass


def log_exception(logger: logging.Logger, 
                 exception: Exception, 
                 context: Optional[dict] = None,
                 level: str = "ERROR"):
    """
    Log an exception with context information
    
    Args:
        logger: Logger instance
        exception: The exception to log
        context: Optional context dictionary (e.g., bank_id, method_name)
        level: Log level (ERROR, WARNING, CRITICAL)
    """
    context_str = ""
    if context:
        context_str = " - " + ", ".join(f"{k}={v}" for k, v in context.items())
    
    log_method = getattr(logger, level.lower())
    log_method(f"{type(exception).__name__}: {str(exception)}{context_str}", 
               exc_info=level == "ERROR")


def log_and_raise(logger: logging.Logger,
                 exception_class: type,
                 message: str,
                 context: Optional[dict] = None):
    """
    Log an error and raise an exception
    
    Args:
        logger: Logger instance
        exception_class: Exception class to raise
        message: Error message
        context: Optional context dictionary
        
    Raises:
        exception_class: The specified exception
    """
    context_str = ""
    if context:
        context_str = " - " + ", ".join(f"{k}={v}" for k, v in context.items())
    
    logger.error(f"{message}{context_str}")
    raise exception_class(message, context)


def deprecated(reason: Optional[str] = None,
               alternative: Optional[str] = None,
               removal_version: Optional[str] = None):
    """
    Decorator to mark functions/methods as deprecated.
    Emits a `DeprecationWarning` and logs a warning via AuditLogger.

    Args:
        reason: Optional message explaining why it is deprecated
        alternative: Suggested alternative API to use
        removal_version: Version when this will be removed
    """
    def decorator(func):
        logger = AuditLogger.get_logger(func.__module__)

        def _build_message(name: str) -> str:
            parts = [f"{name} is deprecated."]
            if reason:
                parts.append(reason)
            if alternative:
                parts.append(f"Use {alternative} instead.")
            if removal_version:
                parts.append(f"Scheduled for removal in version {removal_version}.")
            return " ".join(parts)

        def wrapper(*args, **kwargs):
            msg = _build_message(func.__name__)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            logger.warning(msg)
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__dict__.update(func.__dict__)
        return wrapper

    return decorator


# Initialize default logging on module import
AuditLogger.setup_logging(
    log_level="INFO",
    log_file=f"audit_system_{datetime.now().strftime('%Y%m%d')}.log",
    console_output=True
)
