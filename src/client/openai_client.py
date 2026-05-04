"""
OpenAI 协议适配层。

本模块将 OpenAI SDK 的原生接口封装为本系统的标准契约，承担：
1. 根据 ModelConfig 创建并持有 OpenAI SDK 客户端
2. 消息格式转换与 API 参数构建
3. 响应/流式块解析
4. 将 OpenAI 原生异常翻译为 Python 内置异常

外部调用者应通过 ClientGateway 访问，而非直接使用本类。
"""

from collections.abc import Iterator

import openai
from openai import OpenAI

from src.core_contracts.client_contracts import (
    LlmRequest,
    LlmResult,
    LlmStreamChunk,
)
from src.core_contracts.tools_contracts import JsonDict
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.model_contracts import Message, TokenUsage


class OpenAIClient:
    """OpenAI SDK 适配器，将标准契约转换为 OpenAI API 调用。"""

    def __init__(self, config: ModelConfig) -> None:
        """根据 ModelConfig 初始化并创建底层 OpenAI 客户端。

        Args:
            config: 包含 api_key、base_url 及默认模型参数的配置契约。

        Raises:
            ValueError: 当 api_key 为空字符串时抛出。
        """
        if not config.api_key or not config.api_key.strip():
            raise ValueError("ModelConfig.api_key 不能为空字符串。")
        self._config: ModelConfig = config
        self._client: OpenAI = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    # =========================================================================
    # 公有接口
    # =========================================================================

    def complete(self, request: LlmRequest) -> LlmResult:
        """执行非流式聊天补全并返回完整结果。

        Args:
            request: 已通过校验的标准化请求契约。

        Returns:
            LlmResult: 包含生成文本、用量及完成原因的标准化结果。

        Raises:
            RuntimeError: 网络连接失败或其他底层客户端异常。
            TimeoutError: 请求超时。
            PermissionError: API 密钥认证失败。
        """
        params = self._build_params(request, self._assemble_messages(request.messages))
        try:
            completion = self._client.chat.completions.create(**params)
        except openai.OpenAIError as exc:
            raise self._translate_error(exc) from exc
        return self._parse_response(completion)

    def complete_stream(self, request: LlmRequest) -> Iterator[LlmStreamChunk]:
        """执行流式聊天补全并逐块产出增量结果。

        Args:
            request: 已通过校验的标准化请求契约（stream 会被强制置为 True）。

        Yields:
            LlmStreamChunk: 每次增量的标准化流块。

        Raises:
            RuntimeError: 网络连接失败或其他底层客户端异常。
            TimeoutError: 请求超时。
            PermissionError: API 密钥认证失败。
        """
        params = self._build_params(request, self._assemble_messages(request.messages))
        params["stream"] = True
        try:
            stream = self._client.chat.completions.create(**params)
            yield from self._process_stream(stream)
        except openai.OpenAIError as exc:
            raise self._translate_error(exc) from exc

    # =========================================================================
    # 私有辅助函数（深度优先展开）
    # =========================================================================

    def _build_params(self, request: LlmRequest, messages: list[dict[str, str]]) -> dict:
        """根据 LlmRequest 构建 OpenAI API 调用参数字典。

        Args:
            request: 标准化请求契约，其中 model / temperature / max_tokens 为 None
                时回退到 ModelConfig 默认值。
            messages: 已由 _assemble_messages 转换的消息列表。

        Returns:
            dict: 可直接解包传递给 OpenAI chat.completions.create 的参数字典，
            包含 model、messages、temperature、max_tokens 及可选的 stop、top_p、
            frequency_penalty、presence_penalty、tools。
        """
        params: dict = {
            "model": request.model if request.model is not None else self._config.model_name,
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None else self._config.temperature,
            "max_tokens": request.max_tokens if request.max_tokens is not None else self._config.max_tokens,
        }
        if request.stop:
            params["stop"] = request.stop
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty
        if request.tools:
            params["tools"] = request.tools
        return params

    def _assemble_messages(self, messages: list[Message]) -> list[JsonDict]:
        """将 Message 列表转换为 OpenAI API 调用所需的 JSON 字典列表。

        仅序列化 Message 中非 None 的字段，确保 tool_calls / tool_call_id / name
        仅在存在时才出现在输出字典中。

        Args:
            messages: 本系统的 Message 对象列表。

        Returns:
            list[JsonDict]: OpenAI chat.completions.create 兼容的消息字典列表。
        """
        result: list[JsonDict] = []
        for msg in messages:
            d: JsonDict = {"role": msg.role}
            if msg.content is not None:
                d["content"] = msg.content
            if msg.tool_calls is not None:
                d["tool_calls"] = msg.tool_calls
            if msg.tool_call_id is not None:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name is not None:
                d["name"] = msg.name
            result.append(d)
        return result

    def _parse_response(self, completion: object) -> LlmResult:
        """将 OpenAI completion 对象解析为标准的 LlmResult。

        Args:
            completion: OpenAI SDK 返回的 completion 对象。

        Returns:
            LlmResult: 包含 content、model、finish_reason、usage 及可选 tool_calls
            的标准化结果。

        Raises:
            RuntimeError: choices 列表为空或缺少 message 字段时抛出。
        """
        if not completion.choices:
            raise RuntimeError("LLM 返回的 choices 列表为空。")
        choice = completion.choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise RuntimeError("LLM 返回的 choice 中缺少 message 字段。")
        content = getattr(message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", None) or "unknown"
        model = getattr(completion, "model", "unknown")
        usage = self._extract_usage(completion)
        tool_calls = self._parse_tool_calls(message)
        return LlmResult(
            content=content,
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
        )

    def _parse_tool_calls(self, message: object) -> list[dict] | None:
        """从 OpenAI message 对象中提取 tool_calls 列表。

        Args:
            message: OpenAI completion choice 中的 message 对象。

        Returns:
            list[dict] | None: 工具调用列表，每项包含 id / type / function 字段；
            message 不包含 tool_calls 时返回 None。
        """
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return None
        return [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    def _extract_usage(self, completion: object) -> TokenUsage:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )

    def _translate_error(self, error: openai.OpenAIError) -> Exception:
        msg = str(error)
        if isinstance(error, openai.AuthenticationError):
            return PermissionError(msg)
        if isinstance(error, openai.RateLimitError):
            return RuntimeError(msg)
        if isinstance(error, openai.BadRequestError):
            return ValueError(msg)
        if isinstance(error, openai.APITimeoutError):
            return TimeoutError(msg)
        if isinstance(error, openai.APIConnectionError):
            return RuntimeError(msg)
        return RuntimeError(msg)

    def _process_stream(self, stream: object) -> Iterator[LlmStreamChunk]:
        for chunk in stream:
            choices = chunk.choices
            if not choices:
                yield LlmStreamChunk(content="", finish_reason=None, index=0)
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            content = (getattr(delta, "content", None) or "") if delta is not None else ""
            finish_reason = getattr(choice, "finish_reason", None)
            index = getattr(choice, "index", 0)
            yield LlmStreamChunk(content=content, finish_reason=finish_reason, index=index)
