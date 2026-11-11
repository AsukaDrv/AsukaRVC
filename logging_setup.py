"""Central logging configuration for AsukaRVC.

Every module that wants to log should do:

    from logging_setup import get_logger
    logger = get_logger(__name__)

The root handler is configured once by setup_logging(), which app.py
calls at startup. Library noise (httpx, numba, ...) is silenced by
configure_logging_libs().
"""

import logging
import os
import sys

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m" # bold red
}
_COLOR_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Console formatter that colors the level name when stdout is a TTY."""

    def format(self, record):
        formatted = super().format(record)
        if sys.stdout.isatty():
            color = _LEVEL_COLORS.get(record.levelname)
            if color:
                level_token = f"[{record.levelname}]"
                formatted = formatted.replace(level_token, f"{color}{level_token}{_COLOR_RESET}", 1)
        return formatted


_configured = False


def setup_logging(level=logging.INFO):
    """Configure the root logger once. Safe to call multiple times."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _configured = True


def get_logger(name):
    """Return a namespaced logger, after making sure logging is set up."""
    if not _configured:
        setup_logging(level=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG" and logging.DEBUG or logging.INFO)
    return logging.getLogger(name)


def configure_logging_libs(debug=False):
    """Silence noisy third-party libraries used by the WebUI."""
    modules = [
        "numba",
        "httpx",
        "markdown_it",
        "fairseq",
        "faiss",
    ]
    try:
        for module in modules:
            logging.getLogger(module).setLevel(logging.WARNING)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" if not debug else "1"
    except Exception:
        pass
