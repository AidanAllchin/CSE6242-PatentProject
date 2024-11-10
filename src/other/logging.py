from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

class LogLevel(str, Enum):
    """
    Enum for log levels with corresponding logging module levels.
    Values are the same as the logging module levels.
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """
        Convert our enum to logging module levels.
        """
        return getattr(logging, self.value)

@dataclass
class LogConfig:
    """
    Configuration for the logging system.
    """
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[Path] = None
    console_output: bool = True
    log_format: str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

class PatentLogger:
    """
    Switching to something more robust than print statements (I love you `log()`, sorry).
    
    Features:
    - Configurable log levels
    - File and console output
    - Proper timestamp formatting
    - Module-level logging
    - Exception tracking
    
    Example:
        >>> logger = PatentLogger.get_logger("data_processing")
        >>> logger.info("Processing patent data...")
        >>> try:
        ...     raise ValueError("Invalid patent ID")
        ... except Exception as e:
        ...     logger.exception("Failed to process patent")
    """
    
    _loggers: dict[str, logging.Logger] = {}
    _initialized: bool = False
    _config: LogConfig = LogConfig()
    
    @classmethod
    def initialize(cls, config: LogConfig = None) -> None:
        """
        Initialize the logging system with the given configuration.

        Args:
            config: The configuration to use for the logging system
        """
        if cls._initialized:
            return
            
        cls._config = config or LogConfig()
        
        logging.basicConfig(
            level=cls._config.log_level.to_logging_level(),
            format=cls._config.log_format,
            datefmt=cls._config.date_format,
            handlers=cls._get_handlers()
        )
        
        cls._initialized = True
    
    @classmethod
    def _get_handlers(cls) -> list[logging.Handler]:
        """
        Create and configure log handlers based on configuration.

        Returns:
            A list of configured log handlers (console and/or file)
        """
        handlers = []
        
        if cls._config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(cls._config.log_format, datefmt=cls._config.date_format)
            )
            handlers.append(console_handler)
        
        if cls._config.log_file:
            file_handler = logging.FileHandler(cls._config.log_file)
            file_handler.setFormatter(
                logging.Formatter(cls._config.log_format, datefmt=cls._config.date_format)
            )
            handlers.append(file_handler)
        
        return handlers
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for the given name (per file/module).
        
        Args:
            name: The name of the logger, typically the module name
            
        Returns:
            A configured logger instance
        """
        if not cls._initialized:
            cls.initialize()
            
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(cls._config.log_level.to_logging_level())
            cls._loggers[name] = logger
            
        return cls._loggers[name]