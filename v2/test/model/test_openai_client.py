from src.core.contracts import Message
from src.model.config import ModelConfig
from src.model.openai_client import OpenAIClient


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeStreamChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = FakeDelta(content)


class FakeStreamChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeStreamChoice(content)]


class FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(
        self,
        content: str | None = "answer",
        stream_contents: list[str | None] | None = None,
    ) -> None:
        self.content = content
        self.stream_contents = stream_contents or []
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream") is True:
            return [
                FakeStreamChunk(content)
                for content in self.stream_contents
            ]
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


def test_stream_sends_openai_compatible_stream_request() -> None:
    completions = FakeCompletions(stream_contents=["hello"])
    client = OpenAIClient(
        config=ModelConfig(provider="openai", model="test-model"),
        client=FakeSdkClient(completions),
    )

    list(client.stream([
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hi"),
    ]))

    assert completions.calls == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            "stream": True,
        }
    ]


def test_stream_returns_text_chunks() -> None:
    client = OpenAIClient(
        config=ModelConfig(provider="openai", model="test-model"),
        client=FakeSdkClient(FakeCompletions(stream_contents=["hel", "lo"])),
    )

    chunks = list(client.stream([Message(role="user", content="hi")]))

    assert chunks == ["hel", "lo"]


def test_stream_skips_empty_chunks() -> None:
    client = OpenAIClient(
        config=ModelConfig(provider="openai", model="test-model"),
        client=FakeSdkClient(FakeCompletions(stream_contents=["hel", None, "", "lo"])),
    )

    chunks = list(client.stream([Message(role="user", content="hi")]))

    assert chunks == ["hel", "lo"]
