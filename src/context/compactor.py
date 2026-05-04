"""主动 compact 与 reactive compact 的上下文摘要压缩器。

提供 Compactor，通过调用语言模型将旧对话历史压缩为摘要，
并原地写回消息列表。支持两种触发场景：
- auto-compact：投影超出阈值时主动触发；
- reactive compact：模型返回 context length 错误后的恢复性重试。
"""

from __future__ import annotations

import json

from src.core_contracts.model_contracts import Message
from src.core_contracts.client_contracts import LlmRequest
from src.core_contracts.context_contracts import CompactionResult, ContextModelClient
from src.context.token_estimator import TokenEstimator


_COMPACT_PROMPT: str = (
    "You are compressing earlier conversation history for a coding agent. "
    "Return plain text only. Summarize the essential state needed to continue the task. "
    "Include: user goal, important files or tools already used, key findings or edits, "
    "and the next concrete step. Do not ask follow-up questions. Do not call tools."
)

_COMPACT_SUMMARY_HEADER: str = "<system-reminder>\nCompact summary of earlier conversation:"
_COMPACT_SUMMARY_FOOTER: str = "</system-reminder>"

_CONTEXT_LENGTH_KEYWORDS: tuple[str, ...] = (
    "context length",
    "context window",
    "maximum context length",
    "prompt too long",
    "prompt is too long",
    "too many tokens",
    "context_length_exceeded",
    "token limit exceeded",
)


class Compactor:
    """通过 LLM 摘要压缩旧对话历史的 compact 执行器。

    核心职责：
    1. compact() 构造压缩请求，调用模型，将摘要原地写回消息列表；
    2. is_context_length_error() 供调用方识别 reactive compact 的触发条件。
    """

    def __init__(
        self,
        client: ContextModelClient,
        token_estimator: TokenEstimator,
    ) -> None:
        """通过依赖注入初始化压缩器。

        Args:
            client (ContextModelClient): 用于生成摘要的模型客户端。
            token_estimator (TokenEstimator): 共享的启发式 token 估算器。
        Returns:
            None
        Raises:
            None
        """
        self._client = client
        # ContextModelClient: 模型客户端，compact 调用的唯一出口，由外部注入。
        self._estimator = token_estimator
        # TokenEstimator: 用于统计 compact 前后 token 变化量，与其他组件共享。

    # =========================================================================
    # 公有接口
    # =========================================================================

    def compact(
        self,
        messages: list[Message],
        *,
        preserve_messages: int = 4,
    ) -> CompactionResult:
        """调用模型生成摘要并将旧消息原地替换为 compact summary。

        Args:
            messages (list[Message]): 当前会话消息列表（就地修改）。
            preserve_messages (int): 尾部保留不参与压缩的消息条数。
        Returns:
            CompactionResult: compact 执行结果，包含是否成功、摘要文本及 token 统计。
        Raises:
            无（模型调用异常被捕获并写入 CompactionResult.error）。
        """
        prefix_count = self._count_system_prefix(messages)
        upper_index = self._calculate_upper_index(len(messages), prefix_count, preserve_messages)

        if upper_index <= prefix_count:
            return CompactionResult(compacted=False, error="Not enough messages to compact")

        history_text = self._render_history(messages[prefix_count:upper_index])
        if not history_text:
            return CompactionResult(compacted=False, error="Nothing renderable to compact")

        request = self._build_request(history_text)

        try:
            response = self._client.chat(request)
        except Exception as exc:
            return CompactionResult(compacted=False, error=str(exc))

        if response.tool_calls:
            return CompactionResult(
                compacted=False,
                usage=response.usage,
                error="Compact response unexpectedly requested tool calls",
            )

        summary_text = response.content.strip()
        if not summary_text:
            return CompactionResult(
                compacted=False,
                usage=response.usage,
                error="Compact model returned empty summary",
            )

        pre_tokens = self._estimator.estimate_messages(messages)
        replaced_count = upper_index - prefix_count
        summary_message = self._build_summary_message(summary_text)

        del messages[prefix_count:upper_index]
        messages.insert(prefix_count, summary_message)

        post_tokens = self._estimator.estimate_messages(messages)
        preserved = min(preserve_messages, max(0, len(messages) - prefix_count - 1))

        return CompactionResult(
            compacted=True,
            summary_text=summary_text,
            messages_replaced=replaced_count,
            tokens_removed=max(0, pre_tokens - post_tokens),
            pre_tokens=pre_tokens,
            post_tokens=post_tokens,
            preserve_messages_used=preserved,
            usage=response.usage,
        )

    def is_context_length_error(self, exc: Exception) -> bool:
        """判断异常是否属于 context/prompt length 类错误。

        Args:
            exc (Exception): 模型调用抛出的异常对象。
        Returns:
            bool: True 表示该异常由上下文长度超限引起，可尝试 reactive compact 恢复。
        Raises:
            无。
        """
        if getattr(exc, "status_code", None) == 413:
            return True
        detail = f"{getattr(exc, 'detail', '')} {exc}".lower()
        return any(keyword in detail for keyword in _CONTEXT_LENGTH_KEYWORDS)

    # =========================================================================
    # 私有辅助（原子步骤）
    # =========================================================================

    def _count_system_prefix(self, messages: list[Message]) -> int:
        """返回头部连续 system 消息的数量（不参与压缩范围）。

        Args:
            messages (list[Message]): 完整消息列表。
        Returns:
            int: 头部连续 system 消息的条数。
        Raises:
            无。
        """
        count = 0
        for message in messages:
            if message.role == "system":
                count += 1
            else:
                break
        return count

    def _calculate_upper_index(self, total: int, prefix_count: int, preserve_messages: int) -> int:
        """计算压缩范围的上界索引（不含尾部保留消息）。

        Args:
            total (int): 消息列表总长度。
            prefix_count (int): 头部 system 消息数量。
            preserve_messages (int): 尾部保留消息数。
        Returns:
            int: 压缩范围的上界索引（exclusive）。
        Raises:
            无。
        """
        available = max(0, total - prefix_count)
        tail = min(max(preserve_messages, 0), available)
        return total - tail

    def _render_history(self, messages: list[Message]) -> str:
        """将消息子列表渲染为供 compact 模型阅读的纯文本历史记录。

        Args:
            messages (list[Message]): 待渲染的消息子列表。
        Returns:
            str: 多段落纯文本历史；所有消息均无可渲染内容时返回空字符串。
        Raises:
            无。
        """
        parts: list[str] = []
        for index, message in enumerate(messages, start=1):
            extras: list[str] = []
            if message.name:
                extras.append(f"name={message.name}")
            if message.tool_call_id:
                extras.append(f"tool_call_id={message.tool_call_id}")
            if message.tool_calls:
                extras.append(f"tool_calls={json.dumps(message.tool_calls, ensure_ascii=False)}")

            header = f"[{index}] role={message.role}"
            if extras:
                header = f"{header} ({', '.join(extras)})"

            content_text = (message.content or "").strip()
            parts.append(f"{header}\n{content_text or '(empty)'}")

        return "\n\n".join(parts).strip()

    def _build_request(self, history_text: str) -> LlmRequest:
        """构造发送给 compact 模型的 LlmRequest。

        Args:
            history_text (str): 已渲染好的对话历史纯文本。
        Returns:
            LlmRequest: 标准化的模型调用请求，不携带任何工具。
        Raises:
            无。
        """
        return LlmRequest(
            messages=[
                Message(role="system", content=_COMPACT_PROMPT),
                Message(
                    role="user",
                    content=(
                        "Summarize the following earlier conversation history for future turns.\n\n"
                        f"{history_text}"
                    ),
                ),
            ],
        )

    def _build_summary_message(self, summary_text: str) -> Message:
        """将摘要文本封装为插入消息列表的 system 消息。

        Args:
            summary_text (str): compact 模型生成的摘要文本。
        Returns:
            Message: 带 system-reminder 格式的摘要消息。
        Raises:
            无。
        """
        content = (
            f"{_COMPACT_SUMMARY_HEADER}\n"
            f"{summary_text}\n"
            f"{_COMPACT_SUMMARY_FOOTER}"
        )
        return Message(role="system", content=content)
