"""Logging configuration for the Bank Churn Modelling project.

This module sets up a custom logger with both file and console handlers.
Logs are stored in the 'logs' directory with rotation to manage file sizes.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

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

# Export logger
logging = logger