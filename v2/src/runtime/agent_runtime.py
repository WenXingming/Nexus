"""Minimal Agent runtime."""

from __future__ import annotations

from src.core.contracts import AgentRequest, AgentResult, Message
from src.model.contracts import Model


class AgentRuntime:
    """Runs one agent request against a model."""

    def __init__(self, model: Model) -> None:
        self._model = model

    def run(self, request: AgentRequest) -> AgentResult:
        messages = [Message(role="user", content=request.input)]
        output = self._model.complete(messages)
        return AgentResult(
            output=output,
            session_id=request.session_id or "default",
        )
