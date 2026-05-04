"""RAG 提示词构建器（RAG 模块内部实现）。

职责单一：将检索分块格式化为上下文，构建发送给 LLM 的 Message 列表。
不依赖任何 I/O、模型调用或向量运算，可独立单元测试。
"""

from __future__ import annotations

from src.core_contracts.model_contracts import Message
from src.core_contracts.rag_contracts import RagRetrievedChunk


_DEFAULT_SYSTEM_PROMPT = (
    "你是一个精准的知识问答助手。\n"
    "请仅根据下方参考资料中的内容回答用户的问题。\n"
    "如果参考资料中没有足够信息来回答问题，请明确告知用户无法从现有资料中得出答案，"
    "切勿编造或推断超出资料范围的内容。\n"
    "回答时保持简洁、准确，必要时可引用资料中的原文。"
)


class PromptBuilder:
    """将检索上下文与用户问题组合为 LLM 可消费的 Message 列表。

    无构造依赖，纯逻辑转换；调用方自行将返回的 messages 传入
    LLM 客户端完成生成。
    """

    def __init__(self) -> None:
        """初始化提示词构建器，不持有任何可变状态。"""

    # ── 公有接口 ────────────────────────────────────────────────────────────

    def build_messages(
        self,
        query: str,
        chunks: list[RagRetrievedChunk],
        max_tokens: int,
        system_override: str | None,
    ) -> list[Message]:
        """基于检索上下文构建发送给模型的 Message 列表。

        Args:
            query (str): 用户的原始问题文本。
            chunks (list[RagRetrievedChunk]): 检索得到的相关分块列表。
            max_tokens (int): 回答所允许的最大 token 数，写入提示词约束。
            system_override (str | None): 覆盖默认系统提示词；为 None 时使用内置默认值。
        Returns:
            list[Message]: 包含 system 和 user 两条消息的列表。
        """
        system_text = system_override if system_override is not None else _DEFAULT_SYSTEM_PROMPT
        context_text = self._format_context(chunks)
        user_text = self._format_user_message(query, context_text, max_tokens)
        return [
            Message(role='system', content=system_text),
            Message(role='user', content=user_text),
        ]

    # ── 私有辅助方法 ────────────────────────────────────────────────────────

    def _format_context(self, chunks: list[RagRetrievedChunk]) -> str:
        """将检索分块格式化为人类可读的参考资料文本块。"""
        if not chunks:
            return "（未找到相关参考资料）"

        lines: list[str] = []
        for idx, retrieved in enumerate(chunks, start=1):
            chunk = retrieved.chunk
            source_hint = chunk.metadata.get('source', chunk.doc_id)
            lines.append(
                f"【资料 {idx}】来源: {source_hint} | "
                f"文档: {chunk.doc_id} | 段落: #{chunk.position} | "
                f"相似度: {retrieved.score:.3f}\n"
                f"{chunk.content}"
            )
        return "\n\n---\n\n".join(lines)

    def _format_user_message(self, query: str, context_text: str, max_tokens: int) -> str:
        """将查询与上下文拼装为最终用户消息正文。"""
        return (
            f"以下是与问题相关的参考资料：\n\n"
            f"{context_text}\n\n"
            f"---\n\n"
            f"用户问题：{query}\n\n"
            f"请根据上述参考资料回答，回答字数请控制在 {max_tokens} tokens 以内。"
        )
