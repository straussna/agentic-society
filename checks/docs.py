"""copy_docs.py: the documentation copied into the data folder."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import copy_docs


def check_copy_docs_writes_every_doc_over_its_copy_and_skips_agent_guides():
    """Every file under docs/ lands under the data folder's docs/ at its relative path,
    over any older copy; CLAUDE.md and AGENTS.md stay behind, and files only the
    destination holds are kept."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "repo" / "docs"
        (source / "img").mkdir(parents=True)
        (source / "design.md").write_text("design", encoding="utf-8")
        (source / "img" / "map.svg").write_text("<svg/>", encoding="utf-8")
        (source / "CLAUDE.md").write_text("guide", encoding="utf-8")
        (source / "img" / "AGENTS.md").write_text("guide", encoding="utf-8")
        data = root / "data"
        dest = data / "agentic-society" / "docs"
        dest.mkdir(parents=True)
        (dest / "design.md").write_text("stale", encoding="utf-8")
        (dest / "kept.txt").write_text("kept", encoding="utf-8")

        saved = os.environ.get("DATA_ROOT")
        os.environ["DATA_ROOT"] = str(data)
        try:
            written = copy_docs.copy_docs(source)
        finally:
            if saved is None:
                del os.environ["DATA_ROOT"]
            else:
                os.environ["DATA_ROOT"] = saved

        assert written == [dest / "design.md", dest / "img" / "map.svg"], written
        assert (dest / "design.md").read_text(encoding="utf-8") == "design"
        assert (dest / "img" / "map.svg").read_text(encoding="utf-8") == "<svg/>"
        assert (dest / "kept.txt").read_text(encoding="utf-8") == "kept"
        assert not (dest / "CLAUDE.md").exists()
        assert not (dest / "img" / "AGENTS.md").exists()


def check_copy_docs_copies_the_repo_docs_to_a_named_destination():
    """The command line's --dest copies the repository's own docs/ there."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "docs"
        assert copy_docs.main(["--dest", str(dest)]) == 0
        expected = sorted(
            p.relative_to(copy_docs.DOCS)
            for p in copy_docs.DOCS.rglob("*")
            if p.is_file() and p.name not in copy_docs.AGENT_GUIDES
        )
        got = sorted(p.relative_to(dest) for p in dest.rglob("*") if p.is_file())
        assert got == expected and expected, got
