from src.core.contracts import AgentRequest, Message
from src.memory.in_memory_session_store import InMemorySessionStore
from src.model.fake_client import FakeClient
from src.runtime.agent_runtime import AgentRuntime


class RecordingSessionStore:
    def __init__(self) -> None:
        self.create_call_count = 0
        self.messages_by_session_id: dict[str, list[Message]] = {}
        self.saved_session_id: str | None = None
        self.saved_messages: list[Message] | None = None

    def create(self) -> str:
        self.create_call_count += 1
        session_id = f"s{self.create_call_count}"
        self.messages_by_session_id[session_id] = []
        return session_id

    def load(self, session_id: str) -> list[Message]:
        return list(self.messages_by_session_id[session_id])

    def save(self, session_id: str, messages: list[Message]) -> None:
        self.saved_session_id = session_id
        self.saved_messages = list(messages)
        self.messages_by_session_id[session_id] = list(messages)


class RecordingModel:
    def __init__(self) -> None:
        self.messages: list[Message] | None = None

    def complete(self, messages: list[Message]) -> str:
        self.messages = list(messages)
        return "ok"


def test_run_turns_input_into_user_message() -> None:
    model = RecordingModel()
    runtime = AgentRuntime(model=model, session_store=RecordingSessionStore())

    runtime.run(AgentRequest(input="hi"))

    assert model.messages == [Message(role="user", content="hi")]


def test_run_returns_model_output() -> None:
    runtime = AgentRuntime(model=FakeClient(), session_store=RecordingSessionStore())

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"


def test_run_creates_session_id_when_missing() -> None:
    runtime = AgentRuntime(model=FakeClient(), session_store=RecordingSessionStore())

    result = runtime.run(AgentRequest(input="hi"))

    assert result.session_id == "s1"


def test_run_uses_request_session_id() -> None:
    store = RecordingSessionStore()
    store.messages_by_session_id["s1"] = []
    runtime = AgentRuntime(model=FakeClient(), session_store=store)

    result = runtime.run(AgentRequest(input="hi", session_id="s1"))

    assert result.session_id == "s1"
    assert store.create_call_count == 0


def test_run_sends_history_plus_current_user_message_to_model() -> None:
    model = RecordingModel()
    store = RecordingSessionStore()
    store.messages_by_session_id["s1"] = [
        Message(role="user", content="first"),
        Message(role="assistant", content="ok"),
    ]
    runtime = AgentRuntime(model=model, session_store=store)

    runtime.run(AgentRequest(input="second", session_id="s1"))

    assert model.messages == [
        Message(role="user", content="first"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="second"),
    ]


def test_run_saves_user_and_assistant_messages() -> None:
    store = RecordingSessionStore()
    runtime = AgentRuntime(model=FakeClient(), session_store=store)

    result = runtime.run(AgentRequest(input="hi"))

    assert store.saved_session_id == result.session_id
    assert store.saved_messages == [
        Message(role="user", content="hi"),
        Message(role="assistant", content="Echo: hi"),
    ]


def test_run_saves_history_with_new_turn() -> None:
    store = RecordingSessionStore()
    store.messages_by_session_id["s1"] = [
        Message(role="user", content="first"),
        Message(role="assistant", content="Echo: first"),
    ]
    runtime = AgentRuntime(model=FakeClient(), session_store=store)

    runtime.run(AgentRequest(input="second", session_id="s1"))

    assert store.saved_messages == [
        Message(role="user", content="first"),
        Message(role="assistant", content="Echo: first"),
        Message(role="user", content="second"),
        Message(role="assistant", content="Echo: second"),
    ]


def test_run_with_in_memory_store_keeps_history_across_turns() -> None:
    store = InMemorySessionStore()
    runtime = AgentRuntime(model=FakeClient(), session_store=store)

    first = runtime.run(AgentRequest(input="first"))
    runtime.run(AgentRequest(input="second", session_id=first.session_id))

    assert store.load(first.session_id) == [
        Message(role="user", content="first"),
        Message(role="assistant", content="Echo: first"),
        Message(role="user", content="second"),
        Message(role="assistant", content="Echo: second"),
    ]
