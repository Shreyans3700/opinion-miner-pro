"""Custom exception handling for the Bank Churn Modelling project.

This module defines custom exception classes and utilities for detailed error
reporting with file and line number information.
"""

import sys
from src.utils.logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Extract detailed exception information including file and line number.

    Args:
        error (Exception): The original exception object.
        error_detail (sys): System module used to access traceback information.

    Returns:
        str: Formatted error message with filename and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown File"
        line_number = "Unknown Line"

    return (
        f"Error occurred in script [{file_name}] at line [{line_number}]: {str(error)}"
    )


class CustomException(Exception):
    def __init__(self, error_msg: Exception, error_detail: sys):
        """Initialize the custom exception with detailed error context.

        Args:
            error_msg (Exception): The original exception being wrapped.
            error_detail (sys): System module used to access traceback details.

        Returns:
            None
        """
        self.error_msg = error_message_detail(error_msg, error_detail)

        # Log the error immediately
        logging.error(self.error_msg)

        # Optional: store original exception
        self.original_exception = error_msg

        super().__init__(self.error_msg)

    def __str__(self):
        """Return the formatted error message for this exception."""
        return self.error_msg