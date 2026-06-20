from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest


def test_create_runtime_returns_working_runtime() -> None:
    runtime = create_runtime()

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"
    assert result.session_id == "s1"
