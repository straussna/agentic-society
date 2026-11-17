"""Copy the documentation into the data folder: docs/ to <DATA_ROOT>/agentic-society/docs/.

Every file under docs/ is copied, keeping its relative path and writing over any
existing copy, except Markdown agent guides (CLAUDE.md, AGENTS.md). Files already in
the destination that docs/ does not hold are left alone.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import datapath

DOCS = Path(__file__).resolve().parent / "docs"
AGENT_GUIDES = {"CLAUDE.md", "AGENTS.md"}


def copy_docs(source: Path = DOCS, dest: Path | None = None) -> list[Path]:
    """Copy every documentation file under source into dest, the data folder's docs/
    when dest is None. Returns the destination path of each file written."""
    dest = datapath.data_root() / "docs" if dest is None else dest
    written = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in AGENT_GUIDES:
            continue
        target = dest / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest",
        type=Path,
        help="destination directory (default: the data folder's docs/)",
    )
    args = parser.parse_args(argv)
    written = copy_docs(dest=args.dest)
    dest = args.dest or datapath.data_root() / "docs"
    print(f"copied {len(written)} files to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
