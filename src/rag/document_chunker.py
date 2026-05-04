"""文档滑动窗口切分器（RAG 模块内部实现）。

职责单一：接收原始文档，按滑动窗口策略产出 RagChunk 列表。
不依赖任何 I/O、模型调用或向量运算，可独立单元测试。
"""

from __future__ import annotations

from src.core_contracts.rag_contracts import RagChunk, RagDocument, RagError

_SENTENCE_BREAK_CHARS = frozenset({'.', '!', '?', '。', '！', '？', ';', '；', ':', '：'})


class DocumentChunker:
    """将单篇文档按滑动窗口策略切分为多个文本分块。

    采用字符级计数，优先在句末标点、换行符或空白处断行，
    保留相邻分块间的重叠字符以维持跨块语义连贯性。
    """

    def __init__(self) -> None:
        """初始化无状态切分器，不持有任何可变配置。"""

    # ── 公有接口（唯一流程入口）─────────────────────────────────────────────

    def chunk(
        self,
        document: RagDocument,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[RagChunk]:
        """将一篇文档按滑动窗口策略切分为分块列表（唯一流程入口）。

        Args:
            document (RagDocument): 待切分的原始文档契约对象。
            chunk_size (int): 每个分块的最大字符数（基于 Unicode 字符计数）。
            chunk_overlap (int): 相邻分块间的重叠字符数，须严格小于 chunk_size。
        Returns:
            list[RagChunk]: 切分完成的分块列表；文档内容为空时返回空列表。
        Raises:
            RagError: chunk_size <= 0 或 chunk_overlap >= chunk_size 时抛出。
        """
        self._validate_chunk_parameters(chunk_size, chunk_overlap)
        step = self._calculate_step(chunk_size, chunk_overlap)
        segments = self._split_content_into_segments(document.content, chunk_size, step)
        return self._build_chunk_list(document, segments)

    # ── 私有原子步骤（按调用顺序排列）───────────────────────────────────────

    def _validate_chunk_parameters(self, chunk_size: int, chunk_overlap: int) -> None:
        """验证切分参数的有效性。

        Args:
            chunk_size (int): 每个分块的最大字符数。
            chunk_overlap (int): 相邻分块间的重叠字符数。
        Raises:
            RagError: chunk_size <= 0 或 chunk_overlap >= chunk_size 时抛出。
        """
        if chunk_size <= 0:
            raise RagError(f"chunk_size 必须 >= 1，当前值: {chunk_size}")
        if chunk_overlap >= chunk_size:
            raise RagError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )

    def _calculate_step(self, chunk_size: int, chunk_overlap: int) -> int:
        """计算滑动窗口步长。

        Args:
            chunk_size (int): 每个分块的最大字符数。
            chunk_overlap (int): 相邻分块间的重叠字符数。
        Returns:
            int: 滑动窗口步长（chunk_size - chunk_overlap）。
        """
        return chunk_size - chunk_overlap

    def _split_content_into_segments(self, content: str, chunk_size: int, step: int) -> list[str]:
        """使用滑动窗口策略将纯文本切分为字符串片段列表。

        Args:
            content (str): 待切分的文本全文。
            chunk_size (int): 每个片段的最大字符数。
            step (int): 滑动窗口步长。
        Returns:
            list[str]: 切分后的文本片段列表；内容为纯空白时返回空列表。
        """
        if not content.strip():
            return []

        segments: list[str] = []
        start = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            if end < len(content):
                end = self._find_optimal_break_point(content, start, end, chunk_size)
            stripped = content[start:end].strip()
            if stripped:
                segments.append(stripped)
            start += step
            if step <= 0:
                break

        return segments

    def _find_optimal_break_point(self, content: str, start: int, end: int, chunk_size: int) -> int:
        """在 [start, end] 窗口内寻找最靠近 end 的自然断点。

        断点优先级：
            1) 换行符
            2) 句末标点（中英文 . ! ? ; :）
            3) 任意空白符
            4) 词中间强切时，向前/向后探测最近词边界
            5) 回退到 end 强制截断

        Args:
            content (str): 被搜索的文本全文。
            start (int): 搜索窗口的起始位置（含）。
            end (int): 搜索窗口的终止位置（不含），也是找不到断点时的回退值。
            chunk_size (int): 当前分块上限字符数，用于控制向前探测的最大距离。
        Returns:
            int: 最佳断点位置（断点后第一个字符的索引）；找不到时返回 end。
        """
        newline_pos = self._is_newline_break(content, start, end)
        if newline_pos is not None:
            return newline_pos

        sentence_break = self._is_sentence_break(content, start, end)
        if sentence_break is not None:
            return sentence_break

        whitespace_break = self._is_whitespace_break(content, start, end)
        if whitespace_break is not None:
            return whitespace_break

        if self._is_mid_word_cut(content, end):
            backward_word_break = self._find_backward_word_boundary(content, start, end)
            if backward_word_break is not None:
                return backward_word_break

            forward_probe_span = max(8, chunk_size // 2)
            forward_limit = min(len(content), end + forward_probe_span)
            forward_word_break = self._find_forward_word_boundary(content, end, forward_limit)
            if forward_word_break is not None:
                return forward_word_break

        return end

    def _is_newline_break(self, content: str, start: int, end: int) -> int | None:
        """检测窗口内是否存在换行符断点。

        Args:
            content (str): 被搜索的文本全文。
            start (int): 搜索窗口起始索引（含）。
            end (int): 搜索窗口终止索引（不含）。
        Returns:
            int | None: 命中时返回断点位置（换行符后一个字符），否则返回 None。
        """
        newline_pos = content.rfind('\n', start, end)
        if newline_pos > start:
            return newline_pos + 1
        return None

    def _is_sentence_break(self, content: str, start: int, end: int) -> int | None:
        """在窗口内自右向左查找句末标点断点。

        Args:
            content (str): 被搜索的文本全文。
            start (int): 搜索窗口起始索引（含）。
            end (int): 搜索窗口终止索引（不含）。
        Returns:
            int | None: 命中时返回断点位置（标点后一个字符），否则返回 None。
        """
        for idx in range(end - 1, start - 1, -1):
            if content[idx] in _SENTENCE_BREAK_CHARS:
                return idx + 1
        return None

    def _is_whitespace_break(self, content: str, start: int, end: int) -> int | None:
        """在窗口内自右向左查找空白符断点。

        Args:
            content (str): 被搜索的文本全文。
            start (int): 搜索窗口起始索引（含）。
            end (int): 搜索窗口终止索引（不含）。
        Returns:
            int | None: 命中时返回空白符后第一个非空白字符的位置，否则返回 None。
        """
        for idx in range(end - 1, start - 1, -1):
            if content[idx].isspace():
                return idx + 1
        return None

    def _is_mid_word_cut(self, content: str, end: int) -> bool:
        """判断当前截断点是否切在了英文单词中间。

        Args:
            content (str): 被搜索的文本全文。
            end (int): 截断点位置索引（不含）。
        Returns:
            bool: 若截断点两侧均为字母或数字则返回 True，否则返回 False。
        """
        if end <= 0 or end >= len(content):
            return False
        return content[end - 1].isalnum() and content[end].isalnum()

    def _find_backward_word_boundary(self, content: str, start: int, end: int) -> int | None:
        """向截断点左侧探测最近的词边界。

        Args:
            content (str): 被搜索的文本全文。
            start (int): 向左探测的最远边界（含）。
            end (int): 探测起始位置（从此向左搜索）。
        Returns:
            int | None: 找到非字母数字字符时返回其后一个字符的位置，否则返回 None。
        """
        for idx in range(end - 1, start - 1, -1):
            if not content[idx].isalnum():
                return idx + 1
        return None

    def _find_forward_word_boundary(self, content: str, end: int, limit: int) -> int | None:
        """向截断点右侧探测最近的词边界。

        Args:
            content (str): 被搜索的文本全文。
            end (int): 探测起始位置（从此向右搜索）。
            limit (int): 向右探测的最远边界（不含）。
        Returns:
            int | None: 找到非字母数字字符时返回其位置，否则返回 None。
        """
        for idx in range(end, limit):
            if not content[idx].isalnum():
                return idx
        return None

    def _build_chunk_list(self, document: RagDocument, segments: list[str]) -> list[RagChunk]:
        """将文本片段列表转换为 RagChunk 列表。

        Args:
            document (RagDocument): 原始文档契约对象，用于提取 doc_id 和 metadata。
            segments (list[str]): 已切分的文本片段列表。
        Returns:
            list[RagChunk]: 构建完成的分块列表。
        """
        return [
            RagChunk(
                chunk_id=f"{document.doc_id}#{position}",
                doc_id=document.doc_id,
                content=segment,
                position=position,
                metadata=dict(document.metadata),
            )
            for position, segment in enumerate(segments)
        ]
