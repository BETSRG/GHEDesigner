from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "ghedesigner.gui"
LOGGER = logging.getLogger(LOGGER_NAME)


def default_log_path() -> Path:
    """Return the default per-user GUI diagnostic log path."""
    return Path.home() / ".ghedesigner" / "gui.log"


def configure_gui_logging(log_file: Path, debug: bool = False) -> Path:
    """Configure a rotating diagnostic log shared by the GUI service and browser client."""
    resolved = log_file.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        resolved,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s pid=%(process)d thread=%(threadName)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    for existing in LOGGER.handlers:
        existing.close()
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    return resolved
