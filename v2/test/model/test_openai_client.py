from src.core.contracts import Message
from src.model.config import ModelConfig
from src.model.openai_client import OpenAIClient


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content: str | None = "answer") -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeSdkClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = FakeChat(completions)


def test_complete_sends_openai_compatible_messages() -> None:
    completions = FakeCompletions()
    client = OpenAIClient(
        config=ModelConfig(provider="openai", model="test-model"),
        client=FakeSdkClient(completions),
    )

    client.complete([
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hi"),
    ])

    assert completions.calls == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
        }
    ]


def test_complete_returns_response_text() -> None:
    client = OpenAIClient(
        config=ModelConfig(provider="openai", model="test-model"),
        client=FakeSdkClient(FakeCompletions(content="hello")),
    )

    result = client.complete([Message(role="user", content="hi")])

    assert result == "hello"


def test_complete_returns_empty_string_when_response_content_is_none() -> None:
    client = OpenAIClient(
        config=ModelConfig(provider="openai", model="test-model"),
        client=FakeSdkClient(FakeCompletions(content=None)),
    )

    result = client.complete([Message(role="user", content="hi")])

    assert result == ""
