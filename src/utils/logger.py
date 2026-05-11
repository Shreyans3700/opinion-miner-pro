"""Logging configuration for the Review Intelligence System project.

This module sets up a custom logger with both file and console handlers.
Logs are stored in the 'logs' directory with rotation to manage file sizes.
"""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Create logs directory at the project root, not the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file name
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate logs

# Formatter
formatter = logging.Formatter(
    "[ %(asctime)s ] %(levelname)s %(name)s:%(lineno)d - %(message)s"
)

# File Handler (with rotation)
file_handler = RotatingFileHandler(
    LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=3
)
file_handler.setFormatter(formatter)

# Console Handler (for terminal output)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add handlers only once (avoid duplicate logs)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Test logging
logger.info("Logger initialized successfully")


def get_logger(name: str):
    """Get a child logger with the same handlers as the main logger."""
    _logging = __import__("logging")
    child = logger.manager.getLogger(name)
    if not child.handlers:
        child.setLevel(_logging.INFO)
        child.addHandler(file_handler)
        child.addHandler(console_handler)
    return child


# Export logger
logging = logger
