from __future__ import annotations
import traceback
from contextlib import ContextDecorator
from functools import wraps
import logging

class log_exceptions(ContextDecorator):
    """
    Logs exceptions and re-raises them.
    
    Can be used as a decorator or context manager:

    Usage as decorator:
        @log_exceptions(logger, "Error in function")
        def my_func(...):
            ...

    Usage as context manager:
        with log_exceptions(logger, "Error doing something"):
            ...
    """
    def __init__(self, logger: logging.Logger, msg: str|None = None, level=logging.ERROR):
        self.logger = logger
        self.msg = msg
        self.level = level

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            message = self.msg or "Exception occurred"
            self.logger.log(self.level, f"{message}: {exc_value}")
            self.logger.debug("traceback:\n%s", "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            return False  # Re-raise exception
        return True

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper