"""RAG 文档加载器：从文件系统读取文本资料并转换为 RagDocument 契约。

本模块是 src.rag 的内部实现，外部调用方应通过 src.rag 包入口导入
DocumentLoader，禁止直接依赖本文件。
"""

from __future__ import annotations

import importlib
from pathlib import Path

from src.core_contracts.rag_contracts import RagDocument


_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
        {
            ".pdf",
            ".md",
            ".txt",
            ".rst",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".php",
            ".rb",
            ".sh",
            ".ps1",
            ".html",
            ".css",
            ".scss",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
            ".ini",
            ".xml",
            ".sql",
            ".docx",
            ".xlsx",
            ".pptx",
        }
    )


class DocumentLoader:
    """从文件系统读取文本资料并转换为 RagDocument 契约。

    load() 是唯一流程入口，所有私有方法均为其直接子步骤，
    私有方法间禁止互相调用。

    遵循无状态设计：实例化后不持有任何可变状态，可安全复用。
    """

    def __init__(self) -> None:
        """初始化文档加载器，不持有可变状态。"""

    # ── 唯一流程入口 (Single Orchestrator) ──────────────────────────────────

    def load(self, source: str) -> list[RagDocument]:
        """唯一流程入口：解析路径 → 收集文件 → 读取内容 → 组装文档。

        Args:
            source (str): 文件路径或目录路径。

        Returns:
            list[RagDocument]: 可直接提交给 RagGateway.index_documents 的文档列表。

        Raises:
            FileNotFoundError: source 指向的路径不存在时抛出。
            ValueError: 目录中没有可索引文本文件时抛出。
            OSError: 文件读取失败时抛出。
            RuntimeError: PDF 解析失败时抛出。
        """
        path = self._resolve_path(source)
        file_paths = self._collect_files(path)
        if not file_paths:
            raise ValueError("目录中没有可索引的文本文件。")
        documents: list[RagDocument] = []
        for file_path in file_paths:
            content = self._read_content(file_path)
            documents.append(self._assemble_document(file_path, content))
        return documents

    # ── 原子步骤 (Atomic Steps) — 均为 load() 的直接子步骤 ──────────────────

    def _resolve_path(self, source: str) -> Path:
        """解析用户输入并验证路径存在。

        Args:
            source (str): 用户输入的路径字符串。

        Returns:
            Path: 展开用户目录并解析后的绝对路径。

        Raises:
            FileNotFoundError: 路径不存在时抛出。
        """
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        return path

    def _collect_files(self, path: Path) -> list[Path]:
        """收集待索引的文件列表。

        若 path 为文件则直接返回单元素列表；若为目录则递归扫描，
        按路径排序后返回所有后缀匹配 _SUPPORTED_SUFFIXES 的文件。

        Args:
            path (Path): 已解析的文件或目录路径。

        Returns:
            list[Path]: 待处理的文件路径列表（已排序）。
        """
        if path.is_file():
            return [path]
        return sorted(
            child for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in _SUPPORTED_SUFFIXES
        )

    def _read_content(self, path: Path) -> str:
        """按文件类型读取文档内容。

        PDF 文件委派给 _read_pdf，其他文件按 UTF-8 文本读取。

        Args:
            path (Path): 待读取的文件路径。

        Returns:
            str: 提取后的文档文本内容。

        Raises:
            OSError: 文件读取失败时抛出。
            RuntimeError: PDF 解析失败时抛出。
        """
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".docx":
            return self._read_docx(path)
        if suffix == ".xlsx":
            return self._read_xlsx(path)
        if suffix == ".pptx":
            return self._read_pptx(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        """从 PDF 文件中提取纯文本内容。

        通过 pypdf.PdfReader 按页提取文本并以双换行拼接。

        Args:
            path (Path): 待读取的 PDF 文件路径。

        Returns:
            str: 按页拼接后的 PDF 文本内容。

        Raises:
            RuntimeError: pypdf 未安装或 PDF 解析失败时抛出。
        """
        try:
            pdf_module = importlib.import_module("pypdf")
        except ImportError as exc:
            raise RuntimeError("缺少 PDF 解析依赖，请安装 pypdf。") from exc

        try:
            reader = pdf_module.PdfReader(str(path))
            return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        except Exception as exc:
            raise RuntimeError(f"PDF 解析失败: {path}") from exc

    def _read_docx(self, path: Path) -> str:
        """从 Word 文档中提取纯文本内容。

        Args:
            path (Path): 待读取的 .docx 文件路径。

        Returns:
            str: 按段落拼接后的文本内容。

        Raises:
            RuntimeError: python-docx 未安装或解析失败时抛出。
        """
        try:
            docx_module = importlib.import_module("docx")
        except ImportError as exc:
            raise RuntimeError("缺少 Word 解析依赖，请安装 python-docx。") from exc
        try:
            doc = docx_module.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except Exception as exc:
            raise RuntimeError(f"Word 文档解析失败: {path}") from exc

    def _read_xlsx(self, path: Path) -> str:
        """从 Excel 表格中提取纯文本内容。

        Args:
            path (Path): 待读取的 .xlsx 文件路径。

        Returns:
            str: 按工作表分区拼接后的文本内容。

        Raises:
            RuntimeError: openpyxl 未安装或解析失败时抛出。
        """
        try:
            openpyxl = importlib.import_module("openpyxl")
        except ImportError as exc:
            raise RuntimeError("缺少 Excel 解析依赖，请安装 openpyxl。") from exc
        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            parts: list[str] = []
            for name in wb.sheetnames:
                ws = wb[name]
                rows: list[str] = []
                for row in ws.iter_rows(values_only=True):
                    cells = [str(c) for c in row if c is not None]
                    if cells:
                        rows.append("\t".join(cells))
                if rows:
                    parts.append(f"[Sheet: {name}]\n" + "\n".join(rows))
            wb.close()
            return "\n\n".join(parts).strip()
        except Exception as exc:
            raise RuntimeError(f"Excel 解析失败: {path}") from exc

    def _read_pptx(self, path: Path) -> str:
        """从 PowerPoint 演示文稿中提取纯文本内容。

        Args:
            path (Path): 待读取的 .pptx 文件路径。

        Returns:
            str: 按幻灯片分区拼接后的文本内容。

        Raises:
            RuntimeError: python-pptx 未安装或解析失败时抛出。
        """
        try:
            pptx_module = importlib.import_module("pptx")
        except ImportError as exc:
            raise RuntimeError("缺少 PowerPoint 解析依赖，请安装 python-pptx。") from exc
        try:
            prs = pptx_module.Presentation(str(path))
            slides: list[str] = []
            for i, slide in enumerate(prs.slides, start=1):
                texts = [shape.text for shape in slide.shapes if shape.has_text_frame and shape.text.strip()]
                if texts:
                    slides.append(f"[Slide {i}]\n" + "\n".join(texts))
            return "\n\n".join(slides).strip()
        except Exception as exc:
            raise RuntimeError(f"PowerPoint 解析失败: {path}") from exc

    def _assemble_document(self, path: Path, content: str) -> RagDocument:
        """将文件路径与内容组装为 RagDocument 契约对象。

        Args:
            path (Path): 已读取的文件路径。
            content (str): 文件文本内容。

        Returns:
            RagDocument: 包含 doc_id、content 与 source 元数据的文档契约。
        """
        posix_path = path.resolve().as_posix()
        return RagDocument(
            doc_id=posix_path,
            content=content,
            metadata={"source": posix_path},
        )
