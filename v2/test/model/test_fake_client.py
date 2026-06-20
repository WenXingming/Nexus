from src.core.contracts import Message
from src.model.fake_client import FakeClient


def test_echoes_single_user_message() -> None:
    client = FakeClient()

    result = client.complete([Message(role="user", content="hi")])

    assert result == "Echo: hi"


def test_echoes_latest_user_message() -> None:
    client = FakeClient()
    messages = [
        Message(role="user", content="first"),
        Message(role="assistant", content="Echo: first"),
        Message(role="user", content="second"),
    ]

    result = client.complete(messages)

    assert result == "Echo: second"


def test_returns_empty_echo_without_user_message() -> None:
    client = FakeClient()

    result = client.complete([Message(role="assistant", content="hello")])

    assert result == "Echo:"
