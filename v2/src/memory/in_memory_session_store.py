"""In-memory session message store."""

from __future__ import annotations

from src.core.contracts import Message


class InMemorySessionStore:
    """Stores session messages in the current Python process."""

    def __init__(self) -> None:
        self._next_id = 1
        self._messages_by_session_id: dict[str, list[Message]] = {}

    def create(self) -> str:
        session_id = f"s{self._next_id}"
        self._next_id += 1
        self._messages_by_session_id[session_id] = []
        return session_id

    def load(self, session_id: str) -> list[Message]:
        return list(self._messages_by_session_id[session_id])

    def save(self, session_id: str, messages: list[Message]) -> None:
        self._messages_by_session_id[session_id] = list(messages)
