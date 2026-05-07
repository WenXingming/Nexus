"""
LLM 客户端门面网关 —— 模块对外暴露的唯一公有接口。

ClientGateway 仅负责两件事：
1. 对 LlmRequest 进行全字段前置校验（Fast-Fail）
2. 将合法请求委托给注入的 OpenAIClient 执行

所有 OpenAI 协议细节（参数构建、响应解析、异常翻译）均由
OpenAIClient 承担，网关本身不依赖 openai SDK。
"""

from collections.abc import Iterator

from src.client.openai_client import OpenAIClient
from src.core_contracts.client_contracts import (
    LlmRequest,
    LlmResult,
    LlmStreamChunk,
)
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.model_contracts import Message


class ClientGateway:
    """LLM 调用的门面网关，对外提供极简的 chat / chat_stream 接口。

    通过依赖注入接收 OpenAIClient，自身只保留请求校验逻辑。
    推荐使用 ``src.client.create_gateway`` 工厂函数创建实例。
    """

    _TEMPERATURE_MIN: float = 0.0
    _TEMPERATURE_MAX: float = 2.0
    _TOP_P_MIN: float = 0.0
    _TOP_P_MAX: float = 1.0
    _PENALTY_MIN: float = -2.0
    _PENALTY_MAX: float = 2.0

    def __init__(self, openai_client: OpenAIClient) -> None:
        """注入 OpenAIClient 依赖。

        Args:
            openai_client: 已初始化的 OpenAI 协议适配器。
        """
        self._openai_client = openai_client

    # =========================================================================
    # 公有接口
    # =========================================================================

    def chat(self, request: LlmRequest) -> LlmResult:
        """发送一次聊天补全请求并返回完整结果。

        Args:
            request: 标准化的聊天补全请求契约。

        Returns:
            LlmResult: 包含生成文本、用量及完成原因的标准化结果。

        Raises:
            ValueError: 请求参数不合法。
            RuntimeError: 网络连接失败或其他底层客户端异常。
            TimeoutError: 请求超时。
            PermissionError: API 密钥认证失败。
        """
        self._validate(request)
        return self._openai_client.complete(request)

    def chat_stream(self, request: LlmRequest) -> Iterator[LlmStreamChunk]:
        """发送流式聊天补全请求并返回增量块迭代器。

        Args:
            request: 标准化的聊天补全请求契约（stream 会被强制设为 True）。

        Yields:
            LlmStreamChunk: 每次增量的标准化流块。

        Raises:
            ValueError: 请求参数不合法。
            RuntimeError: 网络连接失败或其他底层客户端异常。
            TimeoutError: 请求超时。
            PermissionError: API 密钥认证失败。
        """
        self._validate(request)
        yield from self._openai_client.complete_stream(request)

    # =========================================================================
    # 请求校验（深度优先展开）
    # =========================================================================

    def _validate(self, request: LlmRequest) -> None:
        self._validate_model(request.model)
        self._validate_messages(request.messages)
        if request.temperature is not None:
            self._validate_temperature(request.temperature)
        if request.max_tokens is not None:
            self._validate_max_tokens(request.max_tokens)
        self._validate_optional_float("top_p", request.top_p, self._TOP_P_MIN, self._TOP_P_MAX)
        self._validate_optional_float("frequency_penalty", request.frequency_penalty, self._PENALTY_MIN, self._PENALTY_MAX)
        self._validate_optional_float("presence_penalty", request.presence_penalty, self._PENALTY_MIN, self._PENALTY_MAX)

    def _validate_model(self, model: str | None) -> None:
        if model is not None and not model.strip():
            raise ValueError("LlmRequest.model 不能为空字符串（或设为 None 以使用 ModelConfig 默认值）。")

    def _validate_messages(self, messages: list[Message]) -> None:
        """校验消息列表的合法性。

        对 tool 角色校验 tool_call_id 和 content；
        对携带 tool_calls 的 assistant 消息放行 content 为空的情况；
        其余角色要求 content 为非空字符串。

        Args:
            messages: 待校验的消息列表。

        Raises:
            ValueError: 消息列表为空、tool 角色缺少必要字段、或普通消息 content
                为空时抛出。
        """
        if not messages:
            raise ValueError("LlmRequest.messages 列表不能为空。")
        for idx, msg in enumerate(messages):
            if msg.role == "tool":
                if not msg.tool_call_id:
                    raise ValueError(
                        f"LlmRequest.messages[{idx}] 为 tool 角色时必须提供 tool_call_id。"
                    )
                if not msg.content:
                    raise ValueError(
                        f"LlmRequest.messages[{idx}] 为 tool 角色时 content 不能为空。"
                    )
            elif msg.tool_calls:
                continue
            elif not msg.content or not msg.content.strip():
                raise ValueError(
                    f"LlmRequest.messages[{idx}].content 不能为空字符串。"
                )

    def _validate_temperature(self, temperature: float) -> None:
        if not (self._TEMPERATURE_MIN <= temperature <= self._TEMPERATURE_MAX):
            raise ValueError(
                f"temperature 必须在 [{self._TEMPERATURE_MIN}, {self._TEMPERATURE_MAX}] 范围内，当前值: {temperature}"
            )

    def _validate_max_tokens(self, max_tokens: int) -> None:
        if not (ModelConfig.MIN_MAX_TOKENS <= max_tokens <= ModelConfig.MAX_MAX_TOKENS):
            raise ValueError(
                f"max_tokens 必须在 [{ModelConfig.MIN_MAX_TOKENS}, {ModelConfig.MAX_MAX_TOKENS}] 范围内，当前值: {max_tokens}"
            )

    def _validate_optional_float(self, field_name: str, value: float | None, lo: float, hi: float) -> None:
        if value is not None and not (lo <= value <= hi):
            raise ValueError(f"{field_name} 必须在 [{lo}, {hi}] 范围内，当前值: {value}")
