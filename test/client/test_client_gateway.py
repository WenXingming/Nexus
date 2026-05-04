"""
ClientGateway 及 OpenAIClient 的单元测试。

通过 patch src.client.openai_client.OpenAI 严格隔离 openai SDK，
确保测试仅验证 Gateway 的校验/委托逻辑与 OpenAIClient 的编排、
格式转换及异常翻译。
"""

from unittest.mock import MagicMock, patch

import openai
import pytest

from src.client import ClientGateway, create_gateway
from src.core_contracts.client_contracts import (
    LlmRequest,
    LlmResult,
    LlmStreamChunk,
)
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.model_contracts import Message, TokenUsage


# =============================================================================
# 测试夹具
# =============================================================================


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(api_key="sk-test", base_url="https://api.example.com/v1", model_name="test-model")


@pytest.fixture
def valid_messages() -> list[Message]:
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]


@pytest.fixture
def default_request(valid_messages: list[Message]) -> LlmRequest:
    return LlmRequest(messages=valid_messages)


@pytest.fixture
def explicit_request(valid_messages: list[Message]) -> LlmRequest:
    return LlmRequest(
        model="gpt-4o",
        messages=valid_messages,
        temperature=0.5,
        max_tokens=512,
        stop=["END"],
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=-0.2,
    )


@pytest.fixture
def mock_openai_cls() -> MagicMock:
    """Patch OpenAI 构造函数，使其返回的实例的 chat.completions.create 可被控制。"""
    with patch("src.client.openai_client.OpenAI") as mock_cls:
        yield mock_cls


@pytest.fixture
def gateway(config: ModelConfig, mock_openai_cls: MagicMock) -> ClientGateway:
    return create_gateway(config=config)


@pytest.fixture
def mock_create(gateway: ClientGateway) -> MagicMock:
    """返回 gateway 内部 OpenAIClient 持有的 mock 客户端的 chat.completions.create 方法。"""
    return gateway._openai_client._client.chat.completions.create


# =============================================================================
# __init__ —— 构造与配置
# =============================================================================


class TestInit:
    """验证 Gateway 构造逻辑（经由 create_gateway 工厂函数）。"""

    def test_creates_openai_client_with_correct_args(self, config: ModelConfig, mock_openai_cls: MagicMock) -> None:
        create_gateway(config=config)
        mock_openai_cls.assert_called_once_with(api_key=config.api_key, base_url=config.base_url)

    def test_creates_openai_client_without_base_url(self, mock_openai_cls: MagicMock) -> None:
        config = ModelConfig(api_key="sk-test")
        create_gateway(config=config)
        mock_openai_cls.assert_called_once_with(api_key="sk-test", base_url=None)

    def test_raises_on_empty_api_key(self) -> None:
        config = ModelConfig(api_key="")
        with pytest.raises(ValueError, match="api_key"):
            create_gateway(config=config)

    def test_raises_on_whitespace_api_key(self) -> None:
        config = ModelConfig(api_key="   ")
        with pytest.raises(ValueError, match="api_key"):
            create_gateway(config=config)


# =============================================================================
# chat() —— 正常路径
# =============================================================================


class TestChatSuccess:
    """验证 chat() 在正常输入下正确编排并返回结果。"""

    def _make_completion(
        self, content: str = "Hello, world!", finish_reason: str = "stop",
        model: str = "gpt-4o", prompt_tokens: int = 10,
        completion_tokens: int = 5, total_tokens: int = 15,
    ) -> MagicMock:
        completion = MagicMock()
        completion.model = model
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = content
        completion.choices[0].finish_reason = finish_reason
        completion.usage.prompt_tokens = prompt_tokens
        completion.usage.completion_tokens = completion_tokens
        completion.usage.total_tokens = total_tokens
        return completion

    def test_returns_llm_result(self, gateway: ClientGateway, explicit_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.return_value = self._make_completion()

        result = gateway.chat(explicit_request)

        assert isinstance(result, LlmResult)
        assert result.content == "Hello, world!"
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage == TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    def test_passes_required_params(self, gateway: ClientGateway, explicit_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.return_value = self._make_completion()

        gateway.chat(explicit_request)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 512

    def test_passes_optional_params(self, gateway: ClientGateway, explicit_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.return_value = self._make_completion()

        gateway.chat(explicit_request)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["stop"] == ["END"]
        assert kwargs["top_p"] == 0.9
        assert kwargs["frequency_penalty"] == 0.3
        assert kwargs["presence_penalty"] == -0.2

    def test_falls_back_to_config_defaults(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        """当 LlmRequest 字段为 None 时，应回退至 ModelConfig 的默认值。"""
        mock_create.return_value = self._make_completion(model="test-model")

        gateway.chat(default_request)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == "test-model"          # 回退到 config.model_name
        assert kwargs["temperature"] == 0.7             # 回退到 config.temperature
        assert kwargs["max_tokens"] == 1024             # 回退到 config.max_tokens


# =============================================================================
# chat() —— 异常翻译
# =============================================================================


class TestChatErrorTranslation:
    """验证 chat() 将 openai 异常正确翻译为 Python 内置异常。"""

    def test_authentication_error(self, gateway: ClientGateway, default_request: LlmRequest) -> None:
        original = openai.AuthenticationError(message="bad key", response=MagicMock(), body=None)
        gateway._openai_client._client.chat.completions.create.side_effect = original
        with pytest.raises(PermissionError):
            gateway.chat(default_request)

    def test_rate_limit_error(self, gateway: ClientGateway, default_request: LlmRequest) -> None:
        original = openai.RateLimitError(message="too many", response=MagicMock(), body=None)
        gateway._openai_client._client.chat.completions.create.side_effect = original
        with pytest.raises(RuntimeError):
            gateway.chat(default_request)

    def test_bad_request_error(self, gateway: ClientGateway, default_request: LlmRequest) -> None:
        original = openai.BadRequestError(message="bad param", response=MagicMock(), body=None)
        gateway._openai_client._client.chat.completions.create.side_effect = original
        with pytest.raises(ValueError):
            gateway.chat(default_request)

    def test_timeout_error(self, gateway: ClientGateway, default_request: LlmRequest) -> None:
        original = openai.APITimeoutError(request=MagicMock())
        gateway._openai_client._client.chat.completions.create.side_effect = original
        with pytest.raises(TimeoutError):
            gateway.chat(default_request)

    def test_connection_error(self, gateway: ClientGateway, default_request: LlmRequest) -> None:
        original = openai.APIConnectionError(request=MagicMock())
        gateway._openai_client._client.chat.completions.create.side_effect = original
        with pytest.raises(RuntimeError):
            gateway.chat(default_request)

    def test_generic_api_error(self, gateway: ClientGateway, default_request: LlmRequest) -> None:
        original = openai.APIError(message="internal error", request=MagicMock(), body=None)
        gateway._openai_client._client.chat.completions.create.side_effect = original
        with pytest.raises(RuntimeError):
            gateway.chat(default_request)

    def test_error_chaining_preserves_cause(self, gateway: ClientGateway, default_request: LlmRequest) -> None:
        original = openai.APITimeoutError(request=MagicMock())
        gateway._openai_client._client.chat.completions.create.side_effect = original
        try:
            gateway.chat(default_request)
        except TimeoutError as exc:
            assert exc.__cause__ is original


# =============================================================================
# chat() —— 校验失败
# =============================================================================


class TestChatValidation:
    """验证 chat() 在无效输入下提前拒绝。"""

    def test_empty_model_raises(self, gateway: ClientGateway, valid_messages: list[Message]) -> None:
        request = LlmRequest(model="", messages=valid_messages)
        with pytest.raises(ValueError, match="model"):
            gateway.chat(request)

    def test_whitespace_model_raises(self, gateway: ClientGateway, valid_messages: list[Message]) -> None:
        request = LlmRequest(model="   ", messages=valid_messages)
        with pytest.raises(ValueError, match="model"):
            gateway.chat(request)

    def test_none_model_is_valid(self, gateway: ClientGateway, valid_messages: list[Message], mock_create: MagicMock) -> None:
        """model=None 应通过校验并回退到 config.model_name。"""
        mock_create.return_value = MagicMock()
        mock_create.return_value.choices = [MagicMock()]
        mock_create.return_value.choices[0].message.content = "ok"
        mock_create.return_value.choices[0].finish_reason = "stop"
        mock_create.return_value.model = "test-model"
        mock_create.return_value.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        request = LlmRequest(model=None, messages=valid_messages)
        result = gateway.chat(request)
        assert mock_create.call_args.kwargs["model"] == "test-model"

    def test_empty_messages_raises(self, gateway: ClientGateway) -> None:
        request = LlmRequest(model="gpt-4o", messages=[])
        with pytest.raises(ValueError, match="messages"):
            gateway.chat(request)

    def test_message_with_empty_content_raises(self, gateway: ClientGateway) -> None:
        request = LlmRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="")],
        )
        with pytest.raises(ValueError, match="content"):
            gateway.chat(request)

    def test_message_with_whitespace_content_raises(self, gateway: ClientGateway) -> None:
        request = LlmRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="   ")],
        )
        with pytest.raises(ValueError, match="content"):
            gateway.chat(request)

    @pytest.mark.parametrize("temp", [-0.1, 2.1])
    def test_temperature_out_of_range(self, gateway: ClientGateway, valid_messages: list[Message], temp: float) -> None:
        request = LlmRequest(model="gpt-4o", messages=valid_messages, temperature=temp)
        with pytest.raises(ValueError, match="temperature"):
            gateway.chat(request)

    @pytest.mark.parametrize("temp", [0.0, 1.0, 2.0])
    def test_temperature_boundary_valid(self, gateway: ClientGateway, valid_messages: list[Message], temp: float, mock_create: MagicMock) -> None:
        mock_create.return_value = MagicMock()
        mock_create.return_value.choices = [MagicMock()]
        mock_create.return_value.choices[0].message.content = "ok"
        mock_create.return_value.choices[0].finish_reason = "stop"
        mock_create.return_value.model = "g"
        mock_create.return_value.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        request = LlmRequest(model="gpt-4o", messages=valid_messages, temperature=temp)
        gateway.chat(request)  # 不应抛出

    def test_none_temperature_is_valid(self, gateway: ClientGateway, valid_messages: list[Message], mock_create: MagicMock) -> None:
        """temperature=None 应跳过校验并使用 config 默认值。"""
        mock_create.return_value = MagicMock()
        mock_create.return_value.choices = [MagicMock()]
        mock_create.return_value.choices[0].message.content = "ok"
        mock_create.return_value.choices[0].finish_reason = "stop"
        mock_create.return_value.model = "g"
        mock_create.return_value.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        request = LlmRequest(model="gpt-4o", messages=valid_messages, temperature=None)
        gateway.chat(request)  # 不应抛出

    @pytest.mark.parametrize("tokens", [0, -1])
    def test_max_tokens_not_positive(self, gateway: ClientGateway, valid_messages: list[Message], tokens: int) -> None:
        request = LlmRequest(model="gpt-4o", messages=valid_messages, max_tokens=tokens)
        with pytest.raises(ValueError, match="max_tokens"):
            gateway.chat(request)

    @pytest.mark.parametrize("field,value", [
        ("top_p", -0.1), ("top_p", 1.1),
        ("frequency_penalty", -2.1), ("frequency_penalty", 2.1),
        ("presence_penalty", -2.1), ("presence_penalty", 2.1),
    ])
    def test_optional_float_out_of_range(
        self, gateway: ClientGateway, valid_messages: list[Message], field: str, value: float,
    ) -> None:
        kwargs = {field: value}
        request = LlmRequest(model="gpt-4o", messages=valid_messages, **kwargs)
        with pytest.raises(ValueError, match=field):
            gateway.chat(request)


# =============================================================================
# chat_stream() —— 正常路径
# =============================================================================


class TestChatStreamSuccess:
    """验证 chat_stream() 的正常流式输出路径。"""

    def _make_chunk(self, content: str = "", finish_reason: str | None = None, index: int = 0) -> MagicMock:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].finish_reason = finish_reason
        chunk.choices[0].index = index
        return chunk

    def test_yields_chunks_in_order(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.return_value = [
            self._make_chunk("Hello"),
            self._make_chunk(" world"),
            self._make_chunk("!", finish_reason="stop"),
        ]

        results = list(gateway.chat_stream(default_request))

        assert len(results) == 3
        assert results[0] == LlmStreamChunk(content="Hello", finish_reason=None, index=0)
        assert results[1] == LlmStreamChunk(content=" world", finish_reason=None, index=0)
        assert results[2] == LlmStreamChunk(content="!", finish_reason="stop", index=0)

    def test_forces_stream_true(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.return_value = []

        list(gateway.chat_stream(default_request))

        assert mock_create.call_args.kwargs["stream"] is True

    def test_empty_stream_yields_nothing(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.return_value = []

        results = list(gateway.chat_stream(default_request))

        assert results == []

    def test_stream_uses_config_defaults(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.return_value = []

        list(gateway.chat_stream(default_request))

        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1024


# =============================================================================
# chat_stream() —— 边界与异常
# =============================================================================


class TestChatStreamEdgeCases:
    """验证 chat_stream() 的边界情况。"""

    def _make_chunk(self, content: str = "", finish_reason: str | None = None, index: int = 0) -> MagicMock:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].finish_reason = finish_reason
        chunk.choices[0].index = index
        return chunk

    def test_chunk_with_empty_choices(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        chunk = MagicMock()
        chunk.choices = []
        mock_create.return_value = [chunk]

        results = list(gateway.chat_stream(default_request))

        assert results[0].content == ""
        assert results[0].finish_reason is None
        assert results[0].index == 0

    def test_chunk_with_missing_delta(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        del chunk.choices[0].delta
        mock_create.return_value = [chunk]

        results = list(gateway.chat_stream(default_request))

        assert results[0].content == ""

    def test_stream_error_during_creation(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        mock_create.side_effect = openai.RateLimitError(message="rate limited", response=MagicMock(), body=None)

        with pytest.raises(RuntimeError):
            list(gateway.chat_stream(default_request))


# =============================================================================
# 响应解析 —— 边界情况
# =============================================================================


class TestResponseParsingEdgeCases:
    """验证 _parse_response 及相关方法对异常响应的处理。"""

    def test_empty_choices_raises(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        completion = MagicMock()
        completion.choices = []
        mock_create.return_value = completion

        with pytest.raises(RuntimeError, match="choices"):
            gateway.chat(default_request)

    def test_missing_message_raises(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        del completion.choices[0].message
        mock_create.return_value = completion

        with pytest.raises(RuntimeError, match="message"):
            gateway.chat(default_request)

    def test_none_content_defaults_to_empty(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = None
        completion.choices[0].finish_reason = "stop"
        completion.model = "gpt-4o"
        completion.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        mock_create.return_value = completion

        result = gateway.chat(default_request)
        assert result.content == ""

    def test_missing_usage_defaults_to_zeros(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Hi"
        completion.choices[0].finish_reason = "stop"
        completion.model = "gpt-4o"
        del completion.usage
        mock_create.return_value = completion

        result = gateway.chat(default_request)
        assert result.usage == TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def test_none_finish_reason_defaults_to_unknown(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Hi"
        completion.choices[0].finish_reason = None
        completion.model = "gpt-4o"
        completion.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        mock_create.return_value = completion

        result = gateway.chat(default_request)
        assert result.finish_reason == "unknown"

    def test_missing_model_defaults_to_unknown(self, gateway: ClientGateway, default_request: LlmRequest, mock_create: MagicMock) -> None:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Hi"
        completion.choices[0].finish_reason = "stop"
        del completion.model
        completion.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        mock_create.return_value = completion

        result = gateway.chat(default_request)
        assert result.model == "unknown"


# =============================================================================
# DTO 不可变性
# =============================================================================


class TestDtoImmutability:
    """验证所有 DTO 均为 frozen 数据类。"""

    def test_message_is_frozen(self) -> None:
        msg = Message(role="user", content="Hi")
        with pytest.raises(Exception):
            msg.role = "system"  # type: ignore[misc]

    def test_model_config_is_frozen(self) -> None:
        cfg = ModelConfig(api_key="sk-test")
        with pytest.raises(Exception):
            cfg.api_key = "other"  # type: ignore[misc]

    def test_llm_request_is_frozen(self) -> None:
        req = LlmRequest(messages=[Message(role="user", content="Hi")])
        with pytest.raises(Exception):
            req.model = "other"  # type: ignore[misc]

    def test_llm_result_is_frozen(self) -> None:
        result = LlmResult(
            content="Hi", model="g", finish_reason="stop",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        with pytest.raises(Exception):
            result.content = "other"  # type: ignore[misc]

    def test_llm_stream_chunk_is_frozen(self) -> None:
        chunk = LlmStreamChunk(content="Hi", finish_reason=None, index=0)
        with pytest.raises(Exception):
            chunk.content = "other"  # type: ignore[misc]

    def test_token_usage_is_frozen(self) -> None:
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        with pytest.raises(Exception):
            usage.total_tokens = 10  # type: ignore[misc]


# =============================================================================
# Message Literal 类型安全
# =============================================================================


class TestMessageLiteralRole:
    """验证 Message.role 的 Literal 类型约束。"""

    def test_valid_roles_accepted(self) -> None:
        for role in ("system", "user", "assistant"):
            msg = Message(role=role, content="Hi")  # type: ignore[arg-type]
            assert msg.role == role
