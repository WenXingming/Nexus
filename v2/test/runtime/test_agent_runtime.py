from src.core.contracts import AgentRequest, Message
from src.model.fake_model import FakeModel
from src.runtime.agent_runtime import AgentRuntime


class RecordingModel:
    def __init__(self) -> None:
        self.messages: list[Message] | None = None

    def complete(self, messages: list[Message]) -> str:
        self.messages = messages
        return "ok"


def test_run_turns_input_into_user_message() -> None:
    model = RecordingModel()
    runtime = AgentRuntime(model=model)

    runtime.run(AgentRequest(input="hi"))

    assert model.messages == [Message(role="user", content="hi")]


def test_run_returns_model_output() -> None:
    runtime = AgentRuntime(model=FakeModel())

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"


def test_run_uses_default_session_id() -> None:
    runtime = AgentRuntime(model=FakeModel())

    result = runtime.run(AgentRequest(input="hi"))

    assert result.session_id == "default"


def test_run_uses_request_session_id() -> None:
    runtime = AgentRuntime(model=FakeModel())

    result = runtime.run(AgentRequest(input="hi", session_id="s1"))

    assert result.session_id == "s1"
