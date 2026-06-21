"""File-backed session store."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.contracts import Message
from src.memory.session_id import new_session_id


class FileSessionStore:
    """Stores session messages as JSON files."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def create(self) -> str:
        session_id = new_session_id()
        (self._root / f"{session_id}.json").write_text("[]", encoding="utf-8")
        return session_id

    def load(self, session_id: str) -> list[Message]:
        raw = (self._root / f"{session_id}.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        return [
            Message(role=item["role"], content=item["content"])
            for item in data
        ]

    def save(self, session_id: str, messages: list[Message]) -> None:
        data = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        (self._root / f"{session_id}.json").write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
