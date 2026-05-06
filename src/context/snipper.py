"""轻量级上下文剪裁器（tombstone 化）。

提供 Snipper，将旧消息替换为占位 tombstone 摘要，
在不丢失对话结构的前提下大幅降低上下文 token 开销。
操作就地修改消息列表，并返回 SnipResult 统计信息。
"""

from __future__ import annotations

from src.core_contracts.model_contracts import Message
from src.core_contracts.context_contracts import SnipResult
from src.context.token_estimator import TokenEstimator
from src.context._utils import count_system_prefix


_TOMBSTONE_MARKER: str = "<system-reminder>\nOlder "
"""str: 已是 tombstone 的消息 content 以此前缀开头，防止被重复剪裁。"""


class Snipper:
    """管理旧消息 tombstone 化的上下文轻量剪裁器。

    核心职责：
    1. snip() 接收消息列表，跳过头部 system 前缀与尾部保留窗口；
    2. 对中间段中所有可剪裁消息生成 tombstone 替代内容并就地写回；
    3. 累计 token 差值后返回 SnipResult 统计快照。
    """

    def __init__(
        self,
        token_estimator: TokenEstimator,
        long_assistant_threshold: int = 300,
        preview_max_chars: int = 120,
    ) -> None:
        """通过依赖注入初始化剪裁器。

        Args:
            token_estimator (TokenEstimator): 共享的启发式 token 估算器。
            long_assistant_threshold (int): assistant 消息超过此字符数才被视为可剪裁，默认 300。
            preview_max_chars (int): tombstone 中 preview 文本的最大字符数，默认 120。
        Returns:
            None
        Raises:
            None
        """
        self._estimator = token_estimator
        # TokenEstimator: 用于统计 snip 前后 token 差值，由工厂函数注入。
        self._long_threshold = long_assistant_threshold
        # int: assistant 消息被视为"长文本"的字符阈值，超过才允许剪裁。
        self._preview_max = preview_max_chars
        # int: tombstone preview 段落的最大字符数，超出后截断并加省略号。

    # =========================================================================
    # 公有接口
    # =========================================================================

    def snip(
        self,
        messages: list[Message],
        *,
        preserve_messages: int = 4,
    ) -> SnipResult:
        """就地剪裁消息列表中的旧候选消息并返回统计结果。

        Args:
            messages (list[Message]): 待剪裁的消息列表（就地修改）。
            preserve_messages (int): 尾部保留的消息条数，不参与剪裁。
        Returns:
            SnipResult: 本次剪裁的统计快照，含 snipped_count 与 tokens_removed。
        Raises:
            无。
        """
        prefix_count = count_system_prefix(messages)
        tail_count = self._calculate_tail(len(messages), prefix_count, preserve_messages)
        upper_index = len(messages) - tail_count

        snipped_count = 0
        tokens_removed = 0

        for index in range(prefix_count, upper_index):
            message = messages[index]
            if not self._is_snippable(message):
                continue
            original_tokens = self._estimator.estimate_message(message)
            tombstone = self._make_tombstone(message)
            tombstone_tokens = self._estimator.estimate_message(tombstone)
            messages[index] = tombstone
            snipped_count += 1
            tokens_removed += max(0, original_tokens - tombstone_tokens)

        return SnipResult(snipped_count=snipped_count, tokens_removed=tokens_removed)

    # =========================================================================
    # 私有辅助（原子步骤）
    # =========================================================================

    def _calculate_tail(self, total: int, prefix_count: int, preserve_messages: int) -> int:
        """计算尾部保留消息条数（不超过可用消息数）。

        Args:
            total (int): 消息列表总长度。
            prefix_count (int): 头部 system 消息数量。
            preserve_messages (int): 期望保留的尾部消息数。
        Returns:
            int: 实际尾部保留条数。
        Raises:
            无。
        """
        available = max(0, total - prefix_count)
        return min(max(preserve_messages, 0), available)

    def _is_snippable(self, message: Message) -> bool:
        """判断单条消息是否满足剪裁条件。

        tombstone 消息、system 消息和 user 消息不可剪裁；
        tool 消息和带 tool_calls 的 assistant 消息或超长 assistant 消息可剪裁。

        Args:
            message (Message): 待判断的消息对象。
        Returns:
            bool: True 表示可被 tombstone 化；False 表示需要保留原内容。
        Raises:
            无。
        """
        content = message.content or ""
        if content.startswith(_TOMBSTONE_MARKER):
            return False

        if message.role == "tool":
            return True

        if message.role == "assistant":
            if message.tool_calls:
                return True
            return len(content) > self._long_threshold

        return False

    def _make_tombstone(self, message: Message) -> Message:
        """为单条消息生成 tombstone 替代内容。

        Args:
            message (Message): 原始待替换消息。
        Returns:
            Message: 携带 tombstone 摘要的替代消息，保留 role 与关键字段。
        Raises:
            无。
        """
        role = message.role
        preview = self._build_preview(message.content or "")

        if role == "tool":
            label = f"tool result ({message.name or 'tool'})"
        elif role == "assistant" and message.tool_calls:
            label = "assistant message with tool calls"
        else:
            label = role

        tombstone_content = (
            f"{_TOMBSTONE_MARKER}{label} was snipped to save context.\n"
            f"Preview: {preview or '(empty)'}\n"
            f"</system-reminder>"
        )

        if role == "tool":
            return Message(
                role="tool",
                content=tombstone_content,
                tool_call_id=message.tool_call_id,
                name=message.name,
            )
        if role == "assistant" and message.tool_calls:
            return Message(
                role="assistant",
                content=tombstone_content,
                tool_calls=message.tool_calls,
            )
        return Message(role=role, content=tombstone_content)

    def _build_preview(self, content: str) -> str:
        """将原始内容折叠为可写入 tombstone 的短预览文本。

        Args:
            content (str): 消息 content 字段文本。
        Returns:
            str: 折叠后的预览文本；超过 preview_max_chars 时截断并追加省略号。
        Raises:
            无。
        """
        text = " ".join(content.split())
        if len(text) > self._preview_max:
            return text[: self._preview_max - 3] + "..."
        return text
