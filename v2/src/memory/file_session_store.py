"""File-backed session store."""

from __future__ import annotations

from pathlib import Path


class FileSessionStore:
    """Stores session messages as JSON files."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._next_id = 1
        self._root.mkdir(parents=True, exist_ok=True)

    def create(self) -> str:
        session_id = f"s{self._next_id}"
        self._next_id += 1
        (self._root / f"{session_id}.json").write_text("[]", encoding="utf-8")
        return session_id
