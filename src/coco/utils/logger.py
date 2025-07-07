import logging
import sys
from typing import Optional

def get_logger(name: str, log_level: int = logging.INFO, file_path: Optional[str] = None) -> logging.Logger:
    """
    Sets up a logger with optional file output and consistent formatting.

    Args:
        name (str): Logger name.
        log_level (int): Logging level.
        file_path (str, optional): If provided, logs will also be saved to this file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent double logging

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if file_path is not None:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger