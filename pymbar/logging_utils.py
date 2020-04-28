"""
Utilities for our custom logging features
"""
from contextlib import contextmanager
import logging


class ForceEmitLogger(logging.Logger):
    def log(self, level, msg, *args, **kwargs):
        """
        Do what ``logging.Logger`` does, but offer to override all
        level thresholds if ``force_emit=True`` is passed to the
        ``.log()`` call.
        """
        if not isinstance(level, int):
            if logging.raiseExceptions:
                raise TypeError("level must be an integer")
            else:
                return
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)
        elif kwargs.pop("force_emit", None):
            with self.temporary_level(level):
                self._log(level, msg, args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """
        Do what ``logging.Logger`` does, but offer to override all
        level thresholds if ``force_emit=True`` is passed to the
        ``.log()`` call.
        """
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, msg, args, **kwargs)
        elif kwargs.pop("force_emit", None):
            with self.temporary_level(logging.DEBUG):
                self._log(logging.DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Do what ``logging.Logger`` does, but offer to override all
        level thresholds if ``force_emit=True`` is passed to the
        ``.log()`` call.
        """
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)
        elif kwargs.pop("force_emit", None):
            with self.temporary_level(logging.INFO):
                self._log(logging.INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Do what ``logging.Logger`` does, but offer to override all
        level thresholds if ``force_emit=True`` is passed to the
        ``.log()`` call.
        """
        if self.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, msg, args, **kwargs)
        elif kwargs.pop("force_emit", None):
            with self.temporary_level(logging.INFO):
                self._log(logging.INFO, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Do what ``logging.Logger`` does, but offer to override all
        level thresholds if ``force_emit=True`` is passed to the
        ``.log()`` call.
        """
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args, **kwargs)
        elif kwargs.pop("force_emit", None):
            with self.temporary_level(logging.ERROR):
                self._log(logging.ERROR, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Do what ``logging.Logger`` does, but offer to override all
        level thresholds if ``force_emit=True`` is passed to the
        ``.log()`` call.
        """
        if self.isEnabledFor(logging.CRITICAL):
            self._log(logging.CRITICAL, msg, args, **kwargs)
        elif kwargs.pop("force_emit", None):
            with self.temporary_level(logging.CRITICAL):
                self._log(logging.CRITICAL, msg, args, **kwargs)

    @contextmanager
    def temporary_level(self, level=logging.INFO):
        """
        Context manager to change the logging level for a block.
        """
        logger_level = self.getEffectiveLevel()
        handler_levels = [h.level for h in self.handlers]
        self.setLevel(level)
        try:
            yield
        finally:
            self.setLevel(logger_level)
            for handler, handler_level in zip(self.handlers, handler_levels):
                handler.level = handler_level


class PerLevelFormatter(logging.Formatter):
    """
    Adapted from https://stackoverflow.com/a/14859558
    """

    FORMATS = {
        logging.ERROR: "ERROR! %(message)s",
        logging.WARNING: "WARNING: %(message)s",
        logging.INFO: "%(message)s",
        logging.DEBUG: "Debug: %(message)s",
    }

    def __init__(self, fmt="%(levelno)d: %(message)s", datefmt=None, style="%", **kwargs):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, **kwargs)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt
        self._style._fmt = self.FORMATS.get(record.levelno, self._style._fmt)
        # Call the original formatter class to do the grunt work
        result = super().format(record)
        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


def setup_logging(level=logging.WARNING):
    """
    Basic configuration for the project, with a custom formatter
    """
    logging.setLoggerClass(ForceEmitLogger)
    logger = logging.getLogger(__name__.split(".", 1)[0])
    handler = logging.StreamHandler()
    formatter = PerLevelFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    configure_logging_level(level)


def configure_logging_level(level):
    logger = logging.getLogger(__name__.split(".", 1)[0])
    logger.setLevel(level)
