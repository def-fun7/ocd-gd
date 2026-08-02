"""
Centralized logging configuration for the old_gd package.

Provides a single named package logger (`old_gd`) that every module pulls
a child logger from, plus a convenience `setup_logging()` function for
users who want to actually see the output.

Library code should never force its own handlers/format onto whoever
imports it, so a `NullHandler` is attached by default — importing old_gd
produces no console output and no "No handlers could be found" warnings
(standard practice, see
https://docs.python.org/3/howto/logging.html#library-config). If the user
wants to see the log messages, they call `setup_logging()` themselves.
"""

import logging

_PACKAGE_LOGGER_NAME = "ocd_gd"

_package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
_package_logger.addHandler(logging.NullHandler())


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger of the package logger for `module_name`.

    Parameters
    ----------
    module_name : str
        Typically just pass `__name__` from the calling module.

    Returns
    -------
    logging.Logger
        A logger named `old_gd.<module_name>`.
    """
    return logging.getLogger(_PACKAGE_LOGGER_NAME).getChild(module_name.split(".")[-1])


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Attach a simple console handler to the package logger.

    Call this once from your own script/notebook (not from within the
    library itself) if you want old_gd's log messages to actually print
    somewhere:

        from ocd_gd._logging_config import setup_logging
        setup_logging()

    Parameters
    ----------
    level : int, default logging.INFO
        Minimum severity level to emit.
    fmt : str
        Log message format string.
    datefmt : str
        Timestamp format string.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    _package_logger.handlers = [handler]
    _package_logger.setLevel(level)
    _package_logger.propagate = False
