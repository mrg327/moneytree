"""
Centralized logging configuration for MoneyTree application.

Provides consistent logging setup across all modules with appropriate formatters,
handlers, and log levels for development and production use.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    detailed_format: bool = False
) -> None:
    """
    Configure logging for the MoneyTree application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        console_output: Whether to output logs to console
        detailed_format: Whether to use detailed log format with timestamps
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    root_logger.setLevel(numeric_level)
    
    # Create formatters
    if detailed_format:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        debug_formatter = logging.Formatter('%(levelname)s - %(name)s - %(funcName)s - %(message)s')
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(debug_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_module_logger(module_name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger for a specific module with optional custom level.
    
    Args:
        module_name: Name of the module
        level: Optional custom logging level for this module
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)
    
    if level:
        numeric_level = getattr(logging, level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
    
    return logger


# Predefined logger configurations for different components
class LoggerConfig:
    """Predefined logging configurations for different application components."""
    
    WIKI_CRAWLER = "moneytree.wiki.crawler"
    LLM_GENERATOR = "moneytree.llm.generator"
    TTS_GENERATOR = "moneytree.tts.generator"
    VIDEO_PROCESSOR = "moneytree.video.processor"
    DOWNLOAD_MANAGER = "moneytree.download.manager"
    
    @classmethod
    def get_all_loggers(cls) -> dict:
        """Get all predefined loggers."""
        return {
            'wiki_crawler': logging.getLogger(cls.WIKI_CRAWLER),
            'llm_generator': logging.getLogger(cls.LLM_GENERATOR),
            'tts_generator': logging.getLogger(cls.TTS_GENERATOR),
            'video_processor': logging.getLogger(cls.VIDEO_PROCESSOR),
            'download_manager': logging.getLogger(cls.DOWNLOAD_MANAGER),
        }


# Performance timing decorator
def log_execution_time(logger: logging.Logger):
    """
    Decorator to log execution time of functions.
    
    Args:
        logger: Logger instance to use for timing logs
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
                raise
                
        return wrapper
    return decorator


# Context manager for operation logging
class LoggedOperation:
    """Context manager for logging operations with timing."""
    
    def __init__(self, logger: logging.Logger, operation_name: str, log_level: str = "INFO"):
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = getattr(logging, log_level.upper())
        self.start_time = None
    
    def __enter__(self):
        self.start_time = __import__('time').time()
        self.logger.log(self.log_level, f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = __import__('time').time() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.log_level, f"Completed {self.operation_name} in {execution_time:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {execution_time:.2f}s: {exc_val}")
        
        return False  # Don't suppress exceptions