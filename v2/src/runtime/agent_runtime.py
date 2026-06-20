"""Minimal Agent runtime."""

from __future__ import annotations

from src.core.contracts import AgentRequest, AgentResult, Message
from src.memory.contracts import SessionStore
from src.model.contracts import Model


class AgentRuntime:
    """Runs one agent request against a model."""

    def __init__(self, model: Model, session_store: SessionStore) -> None:
        self._model = model
        self._session_store = session_store

    def run(self, request: AgentRequest) -> AgentResult:
        session_id = request.session_id or self._session_store.create()
        messages = self._session_store.load(session_id)
        messages.append(Message(role="user", content=request.input))
        output = self._model.complete(messages)
        messages.append(Message(role="assistant", content=output))
        self._session_store.save(session_id, messages)
        return AgentResult(
            output=output,
            session_id=session_id,
        )
