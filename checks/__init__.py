"""The verification suite, one module per topic. check.py runs it."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable


def checks() -> dict[str, Callable]:
    """Every check across the package, by the label the runner prints."""
    found = {}
    for found_module in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"checks.{found_module.name}")
        for attr, fn in vars(module).items():
            if attr.startswith("check_") and callable(fn):
                assert attr[6:] not in found, f"{attr} is defined twice"
                found[attr[6:]] = fn
    return dict(sorted(found.items()))
