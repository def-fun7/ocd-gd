"""
Centralized logging configuration for the ocd_gd package.

Provides a single named package logger (`ocd_gd`) that every module pulls
a child logger from, plus a convenience `setup_logging()` function for
users who want to actually see the output.

Library code should never force its own handlers/format onto whoever
imports it, so a `NullHandler` is attached by default — importing ocd_gd
produces no console output and no "No handlers could be found" warnings
(standard practice, see
https://docs.python.org/3/howto/logging.html#library-config). If the user
wants to see the log messages, they call `setup_logging()` themselves.
"""

import logging

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

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

        from ocd_gd._terminal_config import setup_logging
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
    root_logger = logging.getLogger()

    # Clear existing handlers to prevent duplicated logs
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Ensure package loggers inherit root configuration
    _package_logger.setLevel(level)
    _package_logger.propagate = True


def print_banner(title: str, subtitle: str = "") -> None:
    """Print a top-level banner/panel for the example script."""
    if HAS_RICH:
        content = (
            f"[bold cyan]ocd-gd[/bold cyan] — [yellow]{subtitle}[/yellow]"
            if subtitle
            else f"[bold cyan]{title}[/bold cyan]"
        )
        console.print(Panel.fit(content, border_style="cyan"))
    else:
        log = get_logger(__name__)
        full_title = f"{title} - {subtitle}" if subtitle else title
        log.info("==================================================")
        log.info("  %s", full_title)
        log.info("==================================================")


def print_kv_table(
    title: str,
    data: dict[str, str | int | float],
    header_style: str = "bold magenta",
) -> None:
    """Render a two-column key-value mapping as a table (or formatted log list)."""
    if HAS_RICH:
        table = Table(title=title, show_header=True, header_style=header_style)
        table.add_column("Property / Metric", style="dim")
        table.add_column("Value", justify="right")
        for key, val in data.items():
            table.add_row(str(key), str(val))
        console.print(table)
    else:
        log = get_logger(__name__)

        log.info("┌── %s", title)
        for key, val in data.items():
            log.info("│ %-22s : %s", key, val)
        log.info("└──")


def print_dataframe_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    header_style: str = "bold green",
) -> None:
    """Render a general multi-column table (useful for sweeps/grid results)."""
    if HAS_RICH:
        table = Table(title=title, show_header=True, header_style=header_style)
        for h in headers:
            table.add_column(h, justify="right" if h != headers[0] else "left")
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        console.print(table)
    else:
        log = get_logger(__name__)

        log.info("=== %s ===", title)
        log.info(" | ".join(headers))
        log.info("-" * 40)
        for row in rows:
            log.info(" | ".join(str(cell) for cell in row))
