# logging_config.py - Update the decorator
import functools
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class AccessibilityLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure root logger
        self.logger = logging.getLogger('accessibility_system')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler (rotating, 10MB max, 5 backups)
        file_handler = RotatingFileHandler(
            'logs/accessibility_system.log',
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self, name=None):
        if name:
            return logging.getLogger(f'accessibility_system.{name}')
        return self.logger

# Global logger instance
logger = AccessibilityLogger().get_logger()

def log_exceptions(func):
    """Decorator to log exceptions with unique endpoint names"""
    @functools.wraps(func)  # This preserves the original function name
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper