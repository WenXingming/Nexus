"""src.rag.document_loader 的测试。

验证：
1. 单文件加载。
2. 目录扫描与过滤。
3. PDF 解析委派。
4. 路径不存在与无支持文件的错误路径。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.rag.document_loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


class TestLoadDocuments:
    def test_loads_single_file_as_one_document(self, loader: DocumentLoader, tmp_path: Path) -> None:
        file_path = tmp_path / "note.md"
        file_path.write_text("hello rag", encoding="utf-8")

        documents = loader.load(str(file_path))

        assert len(documents) == 1
        assert documents[0].content == "hello rag"
        assert documents[0].metadata["source"] == file_path.resolve().as_posix()

    def test_loads_supported_files_from_directory(self, loader: DocumentLoader, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("a", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('b')", encoding="utf-8")
        (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

        documents = loader.load(str(tmp_path))

        assert len(documents) == 2
        assert {Path(doc.doc_id).name for doc in documents} == {"a.md", "b.py"}

    def test_loads_pdf_via_pdf_reader_helper(self, loader: DocumentLoader, tmp_path: Path) -> None:
        file_path = tmp_path / "paper.pdf"
        file_path.write_bytes(b"%PDF-test")

        with patch.object(loader, "_read_pdf", return_value="pdf text") as mock_reader:
            documents = loader.load(str(file_path))

        mock_reader.assert_called_once_with(file_path.resolve())
        assert len(documents) == 1
        assert documents[0].content == "pdf text"

    def test_raises_when_path_missing(self, loader: DocumentLoader, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "missing"))

    def test_raises_when_directory_has_no_supported_files(self, loader: DocumentLoader, tmp_path: Path) -> None:
        (tmp_path / "data.bin").write_bytes(b"123")

        with pytest.raises(ValueError, match="没有可索引"):
            loader.load(str(tmp_path))
