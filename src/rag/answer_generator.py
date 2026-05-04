"""基于 RagModelClient 的提示词构建与回答生成器（RAG 模块内部实现）。

职责单一：将检索分块格式化为上下文，调用注入的模型客户端生成自然语言回答。
不依赖任何向量运算或分块逻辑，可独立单元测试。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core_contracts.client_contracts import LlmRequest
from src.core_contracts.model_contracts import Message
from src.core_contracts.rag_contracts import RagQueryError, RagRetrievedChunk

if TYPE_CHECKING:
    from src.core_contracts.rag_contracts import RagModelClient


class AnswerGenerator:
    """将检索上下文与用户问题组合后调用模型生成回答。

    通过依赖注入接收 RagModelClient，自身只保留提示词构建与结果提取逻辑；
    ClientGateway 天然满足 RagModelClient 协议，可直接注入。
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "你是一个精准的知识问答助手。\n"
        "请仅根据下方参考资料中的内容回答用户的问题。\n"
        "如果参考资料中没有足够信息来回答问题，请明确告知用户无法从现有资料中得出答案，"
        "切勿编造或推断超出资料范围的内容。\n"
        "回答时保持简洁、准确，必要时可引用资料中的原文。"
    )

    def __init__(self, model_client: 'RagModelClient') -> None:
        """初始化回答生成器，注入模型调用依赖。

        Args:
            model_client (RagModelClient): 实现了 chat() 方法的模型客户端；
                                           ClientGateway 可直接传入。
        """
        self._model_client = model_client  # RagModelClient：注入的模型调用客户端。

    # ── 公有接口 ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        chunks: list[RagRetrievedChunk],
        max_tokens: int,
        system_override: str | None,
    ) -> tuple[str, int, int]:
        """基于检索上下文调用模型生成自然语言回答。

        Args:
            query (str): 用户的原始问题文本。
            chunks (list[RagRetrievedChunk]): 检索得到的相关分块列表（应按相似度降序排列）。
            max_tokens (int): 回答所允许的最大 token 数，作为提示写入用户消息。
            system_override (str | None): 覆盖默认系统提示词的字符串；为 None 时使用内置默认提示词。
        Returns:
            tuple[str, int, int]: 三元组 (answer, prompt_tokens, completion_tokens)：
                - answer (str): 模型返回的回答文本。
                - prompt_tokens (int): 输入提示词消耗的 token 数。
                - completion_tokens (int): 输出回答消耗的 token 数。
        Raises:
            RagQueryError: 模型调用失败或返回空回答时抛出。
        """
        messages = self._build_messages(query, chunks, max_tokens, system_override)
        request = LlmRequest(messages=messages, max_tokens=max_tokens)
        try:
            result = self._model_client.chat(request)
        except Exception as exc:
            raise RagQueryError(f"模型调用失败: {exc}") from exc

        answer = result.content.strip()
        if not answer:
            raise RagQueryError("模型返回了空回答，请检查提示词或模型配置。")

        return answer, result.usage.prompt_tokens, result.usage.completion_tokens

    # ── 私有辅助方法（深度优先）──────────────────────────────────────────────

    def _build_messages(
        self,
        query: str,
        chunks: list[RagRetrievedChunk],
        max_tokens: int,
        system_override: str | None,
    ) -> list[Message]:
        """构建发送给模型的 Message 列表。

        Args:
            query (str): 用户问题文本。
            chunks (list[RagRetrievedChunk]): 检索分块列表，用于构建上下文。
            max_tokens (int): 回答最大 token 数，写入用户消息提示模型控制长度。
            system_override (str | None): 自定义系统提示词；为 None 时使用内置默认提示词。
        Returns:
            list[Message]: 包含 system 和 user 两条消息的列表，可直接传入 LlmRequest。
        """
        system_text = system_override if system_override is not None else self._DEFAULT_SYSTEM_PROMPT
        context_text = self._format_context(chunks)
        user_text = self._format_user_message(query, context_text, max_tokens)
        return [
            Message(role='system', content=system_text),
            Message(role='user', content=user_text),
        ]

    def _format_context(self, chunks: list[RagRetrievedChunk]) -> str:
        """将检索分块格式化为人类可读的参考资料文本块。

        Args:
            chunks (list[RagRetrievedChunk]): 检索命中的分块列表；为空时返回占位提示文本。
        Returns:
            str: 格式化后的参考资料字符串，各分块间用分隔线隔开。
        """
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
        """将查询与上下文拼装为最终用户消息正文。

        Args:
            query (str): 用户的原始问题文本。
            context_text (str): 由 _format_context 生成的参考资料文本块。
            max_tokens (int): 回答最大 token 数，写入消息提示模型控制回答长度。
        Returns:
            str: 拼装完成的用户消息字符串，可直接作为 user 角色消息内容。
        """
        return (
            f"以下是与问题相关的参考资料：\n\n"
            f"{context_text}\n\n"
            f"---\n\n"
            f"用户问题：{query}\n\n"
            f"请根据上述参考资料回答，回答字数请控制在 {max_tokens} tokens 以内。"
        )
