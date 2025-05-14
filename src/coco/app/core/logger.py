import sys
import logging

def get_logger(name: str) -> logging.Logger:
    """
    Gets a logger with the given name and configures it to output logs at INFO level
    to stdout.

    If the logger has not been configured yet, this function will add a StreamHandler
    with a formatter that outputs logs in the format:

        [%(asctime)s] [%(levelname)s] [%(name)s] %(message)s

    The logger is configured to output logs at the INFO level.

    :param name: The name of the logger.
    :return: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
