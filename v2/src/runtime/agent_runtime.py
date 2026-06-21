"""Minimal Agent runtime."""

from __future__ import annotations

from src.core.contracts import AgentRequest, AgentResult, Message
from src.memory.contracts import SessionStore
from src.model.contracts import Model
from src.runtime.config import AgentConfig


class AgentRuntime:
    """Runs one agent request against a model."""

    def __init__(
        self,
        model: Model,
        session_store: SessionStore,
        config: AgentConfig | None = None,
    ) -> None:
        self._model = model
        self._session_store = session_store
        self._config = config or AgentConfig()

    def has_session(self, session_id: str) -> bool:
        return self._session_store.exists(session_id)

    def run(self, request: AgentRequest) -> AgentResult:
        session_id = request.session_id or self._session_store.create()
        messages = self._session_store.load(session_id)
        messages.append(Message(role="user", content=request.input))
        model_messages = self._build_model_messages(messages)
        output = self._model.complete(model_messages)
        messages.append(Message(role="assistant", content=output))
        self._session_store.save(session_id, messages)
        return AgentResult(
            output=output,
            session_id=session_id,
        )

    def _build_model_messages(self, messages: list[Message]) -> list[Message]:
        if self._config.system_prompt is None:
            return messages
        return [
            Message(role="system", content=self._config.system_prompt),
            *messages,
        ]
