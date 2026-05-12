"""Session 模块单元测试。"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core_contracts.model_config import ModelConfig
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.session_contracts import (
    SessionSnapshot,
    SessionState,
)
from src.session import SessionGateway, create_gateway
from src.session.session_state import SessionStateRuntime
from src.session.session_store import SessionStore


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(api_key="sk-test", base_url="https://api.example.com/v1", model_name="gpt-4o-mini")


@pytest.fixture
def snapshot(model_config: ModelConfig) -> SessionSnapshot:
    return SessionSnapshot(
        session_id="session-001",
        model_config=model_config,
        messages=(
            Message(role="system", content="You are a coding agent."),
            Message(role="user", content="hello"),
        ),
        transcript=(
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ),
        last_response="hi",
        turns=1,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        metadata={"source": "unit-test", "retry": 0},
    )


class TestSessionGateway:
    """验证 SessionGateway 的门面行为。"""

    def test_save_delegates_to_store(self, snapshot: SessionSnapshot) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        session_store.save.return_value = Path("C:/tmp/sessions/session-001.json")
        session_state = MagicMock()
        gateway = SessionGateway(session_store=session_store, session_state=session_state)

        session_id, session_path = gateway.save(snapshot)

        session_store.save.assert_called_once_with(snapshot)
        assert session_id == "session-001"
        assert session_path.endswith("session-001.json")

    def test_load_delegates_to_store(self, snapshot: SessionSnapshot) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        session_store.load.return_value = snapshot
        session_state = MagicMock()
        gateway = SessionGateway(session_store=session_store, session_state=session_state)

        result = gateway.load("session-001")

        session_store.load.assert_called_once_with("session-001")
        assert result == snapshot

    def test_create_state_delegates_to_runtime(self) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        state = SessionState(messages=[Message(role="user", content="hello")])
        session_state = MagicMock()
        session_state.build_new.return_value = state
        gateway = SessionGateway(session_store=session_store, session_state=session_state)

        result = gateway.create_state("hello")

        session_state.build_new.assert_called_once_with("hello")
        assert result is state

    def test_create_empty_state_delegates_to_runtime(self) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        state = SessionState(session_id="empty-session")
        session_state = MagicMock()
        session_state.build_empty.return_value = state
        gateway = SessionGateway(session_store=session_store, session_state=session_state)

        result = gateway.create_empty_state()

        session_state.build_empty.assert_called_once_with()
        assert result is state

    def test_resume_state_delegates_to_runtime(self) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        state = SessionState(session_id="test-123", messages=[Message(role="user", content="hello")])
        session_state = MagicMock()
        session_state.build_from_persisted.return_value = state
        gateway = SessionGateway(session_store=session_store, session_state=session_state)
        messages = (Message(role="user", content="hello"),)

        result = gateway.resume_state("test-123", messages)

        session_state.build_from_persisted.assert_called_once_with("test-123", messages, ())
        assert result is state

    def test_wraps_unexpected_store_errors(self, snapshot: SessionSnapshot) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        session_store.save.side_effect = RuntimeError("disk broken")
        gateway = SessionGateway(session_store=session_store, session_state=MagicMock())

        with pytest.raises(RuntimeError, match="disk broken"):
            gateway.save(snapshot)

    def test_append_user_delegates_to_runtime(self) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        session_state = MagicMock()
        gateway = SessionGateway(session_store=session_store, session_state=session_state)
        state = SessionState()

        gateway.append_user(state, "hello")

        session_state.append_user.assert_called_once_with(state, "hello")

    def test_append_assistant_delegates_to_runtime(self) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        session_state = MagicMock()
        gateway = SessionGateway(session_store=session_store, session_state=session_state)
        state = SessionState()

        gateway.append_assistant(state, "hi")

        session_state.append_assistant.assert_called_once_with(state, "hi")

    def test_save_state_converts_and_saves(self, model_config: ModelConfig, snapshot: SessionSnapshot) -> None:
        session_store = MagicMock()
        session_store.directory = Path("C:/tmp/sessions")
        session_store.save.return_value = Path("C:/tmp/sessions/session-001.json")
        session_state = MagicMock()
        session_state.to_snapshot.return_value = snapshot
        gateway = SessionGateway(session_store=session_store, session_state=session_state)
        state = SessionState(session_id="session-001")

        session_id, session_path = gateway.save_state(state, model_config, last_response="hi")

        session_state.to_snapshot.assert_called_once_with(
            state=state,
            session_id="session-001",
            model_config=model_config,
            usage=None,
            last_response="hi",
            metadata=None,
        )
        session_store.save.assert_called_once_with(snapshot)
        assert session_id == "session-001"
        assert session_path.endswith("session-001.json")


class TestSessionStateRuntime:
    """验证 SessionStateRuntime 的状态构建行为。"""

    def test_build_new_creates_user_message(self) -> None:
        runtime = SessionStateRuntime()

        state = runtime.build_new("  hello  ")

        assert state.messages == [Message(role="user", content="hello")]
        assert state.transcript_entries == [Message(role="user", content="hello")]
        assert runtime.get_turn_count(state) == 1

    def test_build_new_rejects_blank_prompt(self) -> None:
        runtime = SessionStateRuntime()

        with pytest.raises(ValueError, match="prompt"):
            runtime.build_new("   ")

    def test_build_empty_creates_no_messages(self) -> None:
        runtime = SessionStateRuntime()

        state = runtime.build_empty()

        assert state.session_id
        assert state.messages == []
        assert state.transcript_entries == []

    def test_build_from_persisted_copies_messages(self) -> None:
        runtime = SessionStateRuntime()
        persisted = (Message(role="user", content="hello"), Message(role="assistant", content="hi"))

        state = runtime.build_from_persisted("test-123", messages=persisted, transcript=())

        assert state.session_id == "test-123"
        assert state.messages == list(persisted)
        assert state.transcript_entries == list(persisted)
        assert state.messages is not persisted

    def test_build_from_persisted_rejects_non_tuple(self) -> None:
        runtime = SessionStateRuntime()

        with pytest.raises(ValueError, match="messages"):
            runtime.build_from_persisted("test-123", messages=[Message(role="user", content="hello")], transcript=())  # type: ignore[arg-type]


class TestSessionStore:
    """验证 SessionStore 的持久化边界。"""

    def test_save_and_load_roundtrip(self, tmp_path: Path, snapshot: SessionSnapshot) -> None:
        store = SessionStore(directory=tmp_path)

        saved_path = store.save(snapshot)
        loaded = store.load("session-001")

        assert saved_path == tmp_path / "session-001.json"
        assert loaded.model_config.api_key == ""
        assert loaded.model_config.model_name == snapshot.model_config.model_name
        assert loaded.messages == snapshot.messages
        assert loaded.transcript == snapshot.transcript
        assert loaded.metadata == snapshot.metadata

    def test_load_missing_snapshot_raises(self, tmp_path: Path) -> None:
        store = SessionStore(directory=tmp_path)

        with pytest.raises(FileNotFoundError):
            store.load("missing")

    def test_rejects_path_traversal_session_id(self, tmp_path: Path, model_config: ModelConfig) -> None:
        store = SessionStore(directory=tmp_path)
        bad_snapshot = SessionSnapshot(
            session_id="../escape",
            model_config=model_config,
            messages=(Message(role="user", content="hello"),),
        )

        with pytest.raises(ValueError, match="session_id"):
            store.save(bad_snapshot)

    def test_load_corrupted_json_raises(self, tmp_path: Path) -> None:
        store = SessionStore(directory=tmp_path)
        target = tmp_path / "session-001.json"
        target.write_text("{broken", encoding="utf-8")

        with pytest.raises(RuntimeError, match="JSON"):
            store.load("session-001")


class TestFactoriesAndSnapshot:
    """验证工厂函数与快照对象行为。"""

    def test_create_gateway_assembles_default_dependencies(self, tmp_path: Path) -> None:
        gateway = create_gateway(session_store_directory=tmp_path)

        assert isinstance(gateway, SessionGateway)
        assert gateway.directory == tmp_path.resolve()

    def test_state_to_snapshot_preserves_turn_count(self, model_config: ModelConfig) -> None:
        runtime = SessionStateRuntime()
        state = runtime.build_new("hello")
        runtime.append_assistant(state, "hi")
        runtime.append_user(state, "continue")

        snapshot = runtime.to_snapshot(
            state=state,
            session_id="session-002",
            model_config=model_config,
            usage=TokenUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
            last_response="done",
            metadata={"tag": "smoke"},
        )

        restored = SessionSnapshot.from_dict(snapshot.to_dict())

        assert snapshot.turns == 2
        assert restored.model_config.api_key == ""
        assert restored.model_config.model_name == snapshot.model_config.model_name
        assert restored.messages == snapshot.messages
        assert restored.transcript == snapshot.transcript

    def test_snapshot_does_not_serialize_api_key(self, model_config: ModelConfig) -> None:
        snapshot = SessionSnapshot(
            session_id="session-003",
            model_config=model_config,
            messages=(Message(role="user", content="hello"),),
        )

        payload = snapshot.to_dict()

        assert "api_key" not in payload["model_config"]

    def test_snapshot_roundtrip_preserves_tool_messages(self, model_config: ModelConfig) -> None:
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "list_dir", "arguments": '{"path": "."}'},
            }
        ]
        snapshot = SessionSnapshot(
            session_id="session-004",
            model_config=model_config,
            messages=(
                Message(role="assistant", content=None, tool_calls=tool_calls),
                Message(role="tool", content="result", tool_call_id="call-1", name="list_dir"),
            ),
        )

        restored = SessionSnapshot.from_dict(snapshot.to_dict())

        assert restored.messages == snapshot.messages
