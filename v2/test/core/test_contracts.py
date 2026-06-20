from dataclasses import FrozenInstanceError

import pytest

from src.core.contracts import AgentRequest, AgentResult, Message


def test_message_can_be_created() -> None:
    message = Message(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"


def test_message_is_immutable() -> None:
    message = Message(role="user", content="hello")

    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]


def test_agent_request_defaults_session_id_to_none() -> None:
    request = AgentRequest(input="hi")

    assert request.input == "hi"
    assert request.session_id is None


def test_agent_request_can_include_session_id() -> None:
    request = AgentRequest(input="hi", session_id="s1")

    assert request.input == "hi"
    assert request.session_id == "s1"


def test_agent_request_is_immutable() -> None:
    request = AgentRequest(input="hi")

    with pytest.raises(FrozenInstanceError):
        request.input = "changed"  # type: ignore[misc]


def test_agent_result_can_be_created() -> None:
    result = AgentResult(output="hello", session_id="s1")

    assert result.output == "hello"
    assert result.session_id == "s1"


def test_agent_result_is_immutable() -> None:
    result = AgentResult(output="hello", session_id="s1")

    with pytest.raises(FrozenInstanceError):
        result.output = "changed"  # type: ignore[misc]
