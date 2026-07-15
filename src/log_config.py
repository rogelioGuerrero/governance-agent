"""Configuracion centralizada de logging para governance-agent."""

import logging
import sys

_logger = None


def get_logger(name: str = "governance") -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("governance")
        _logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        )
        _logger.addHandler(handler)
    return _logger.getChild(name) if name != "governance" else _logger
