from dataclasses import FrozenInstanceError

import pytest

from src.contracts import Message


def test_message_can_be_created() -> None:
    message = Message(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"


def test_message_is_immutable() -> None:
    message = Message(role="user", content="hello")

    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]
