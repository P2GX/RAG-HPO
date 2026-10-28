"""Helpers for dependencies that ship as optional extras."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or explain which ``rag-hpo`` extra provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"'{module}' is required for this feature but is not installed. "
            f'Install it with: pip install "rag-hpo[{extra}]"'
        ) from exc
