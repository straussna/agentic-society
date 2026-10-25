"""Where this repository's untracked data lives: <DATA_ROOT>/agentic-society.

The one place that reads DATA_ROOT. Records, mirrors, the interaction store,
experiment records, archives and generated branch manifests all live under it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = "agentic-society"


def data_root() -> Path:
    """This repository's data folder: agentic-society under the DATA_ROOT environment
    variable, or under C:\\data on Windows and ~/data elsewhere when it is unset."""
    base = os.environ.get("DATA_ROOT") or (
        r"C:\data" if sys.platform == "win32" else Path.home() / "data"
    )
    return Path(base) / REPO
