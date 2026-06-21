"""Memory contracts."""

from __future__ import annotations

from typing import Protocol

from src.core.contracts import Message


class SessionStore(Protocol):
    """Stores messages by session id."""

    def create(self) -> str:
        ...

    def load(self, session_id: str) -> list[Message]:
        ...

    def save(self, session_id: str, messages: list[Message]) -> None:
        ...

    def exists(self, session_id: str) -> bool:
        ...
