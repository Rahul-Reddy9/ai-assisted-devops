# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Format-specific document extractors for RAG ingestion."""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from devops_ai.core.exceptions import (
    DocumentFormatError,
    DocumentIngestionError,
    DocumentSizeLimitError,
)
from devops_ai.rag.models import DocumentFormat

try:
    import pypdf

    _PYPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    pypdf = None
    _PYPDF_AVAILABLE = False


class BaseExtractor(ABC):
    """Abstract interface for format-specific text and metadata extractors."""

    @abstractmethod
    def extract(self, file_path: Path, max_bytes: int) -> Tuple[str, Dict[str, Any]]:
        """Extract text content and document-level metadata from file.

        Args:
            file_path: Path to the target document.
            max_bytes: Maximum allowed file size in bytes.

        Returns:
            Tuple of (extracted_text, metadata_dict).

        Raises:
            DocumentSizeLimitError: If file exceeds max_bytes.
            DocumentFormatError: If file content is malformed or invalid.
            DocumentIngestionError: If file I/O or permission error occurs.
        """

    def _validate_file_size(self, file_path: Path, max_bytes: int) -> int:
        """Verify file exists and its size does not exceed the allowed threshold."""
        try:
            stat = file_path.stat()
        except OSError as e:
            raise DocumentIngestionError(f"Cannot stat file: {e}", source=str(file_path)) from e

        size = stat.st_size
        if size > max_bytes:
            raise DocumentSizeLimitError(
                f"File size of {size} bytes exceeds allowed limit of {max_bytes} bytes",
                size=size,
                limit=max_bytes,
            )
        return size


class PlainTextExtractor(BaseExtractor):
    """Extractor for plain text (.txt) documents."""

    def extract(self, file_path: Path, max_bytes: int) -> Tuple[str, Dict[str, Any]]:
        size = self._validate_file_size(file_path, max_bytes)
        try:
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as fh:
                content = fh.read()
        except OSError as e:
            raise DocumentIngestionError(
                f"I/O error reading text file: {e}", source=str(file_path)
            ) from e

        metadata: Dict[str, Any] = {
            "format": DocumentFormat.TEXT.value,
            "raw_size_bytes": size,
        }
        return content, metadata


class MarkdownExtractor(BaseExtractor):
    """Extractor for Markdown (.md) documents with title and section detection."""

    def extract(self, file_path: Path, max_bytes: int) -> Tuple[str, Dict[str, Any]]:
        size = self._validate_file_size(file_path, max_bytes)
        try:
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as fh:
                content = fh.read()
        except OSError as e:
            raise DocumentIngestionError(
                f"I/O error reading markdown file: {e}", source=str(file_path)
            ) from e

        # Extract top-level markdown heading as title if present
        title: Optional[str] = None
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and len(stripped) > 2:
                title = stripped[2:].strip()
                break

        metadata: Dict[str, Any] = {
            "format": DocumentFormat.MARKDOWN.value,
            "raw_size_bytes": size,
        }
        if title:
            metadata["title"] = title

        return content, metadata


class JSONExtractor(BaseExtractor):
    """Extractor for JSON (.json) documents with safe parsing and structural preservation."""

    def extract(self, file_path: Path, max_bytes: int) -> Tuple[str, Dict[str, Any]]:
        size = self._validate_file_size(file_path, max_bytes)
        try:
            with open(file_path, "r", encoding="utf-8-sig") as fh:
                raw_text = fh.read()
        except UnicodeDecodeError as e:
            raise DocumentFormatError(
                f"JSON document must be valid UTF-8 encoded: {e}",
                format_type="json",
            ) from e
        except OSError as e:
            raise DocumentIngestionError(
                f"I/O error reading JSON file: {e}", source=str(file_path)
            ) from e

        if not raw_text.strip():
            return "", {
                "format": DocumentFormat.JSON.value,
                "raw_size_bytes": size,
                "keys_count": 0,
            }

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise DocumentFormatError(f"Malformed JSON syntax: {e}", format_type="json") from e

        # Format structured JSON into clean, normalized indentation for RAG indexing
        normalized_text = json.dumps(parsed, indent=2, ensure_ascii=False)
        keys_count = len(parsed) if isinstance(parsed, (dict, list)) else 1

        metadata: Dict[str, Any] = {
            "format": DocumentFormat.JSON.value,
            "raw_size_bytes": size,
            "root_type": type(parsed).__name__,
            "keys_count": keys_count,
        }
        return normalized_text, metadata


class YAMLExtractor(BaseExtractor):
    """Extractor for YAML (.yaml, .yml) documents using safe loader."""

    def extract(self, file_path: Path, max_bytes: int) -> Tuple[str, Dict[str, Any]]:
        size = self._validate_file_size(file_path, max_bytes)
        try:
            with open(file_path, "r", encoding="utf-8-sig") as fh:
                raw_text = fh.read()
        except UnicodeDecodeError as e:
            raise DocumentFormatError(
                f"YAML document must be valid UTF-8 encoded: {e}",
                format_type="yaml",
            ) from e
        except OSError as e:
            raise DocumentIngestionError(
                f"I/O error reading YAML file: {e}", source=str(file_path)
            ) from e

        if not raw_text.strip():
            return "", {"format": DocumentFormat.YAML.value, "raw_size_bytes": size}

        try:
            # Strictly safe load (prevents arbitrary object instantiation)
            parsed_docs = list(yaml.safe_load_all(raw_text))
        except yaml.YAMLError as e:
            raise DocumentFormatError(f"Malformed YAML syntax: {e}", format_type="yaml") from e

        # Normalize YAML docs for embedding
        cleaned_text = raw_text.strip()
        metadata: Dict[str, Any] = {
            "format": DocumentFormat.YAML.value,
            "raw_size_bytes": size,
            "document_count": len(parsed_docs),
        }
        return cleaned_text, metadata


class PDFExtractor(BaseExtractor):
    """Extractor for PDF (.pdf) documents extracting text streams per page."""

    def extract(self, file_path: Path, max_bytes: int) -> Tuple[str, Dict[str, Any]]:
        size = self._validate_file_size(file_path, max_bytes)
        if not _PYPDF_AVAILABLE:
            raise DocumentIngestionError(
                "pypdf package is required for PDF ingestion. Please install pypdf.",
                source=str(file_path),
            )

        try:
            reader = pypdf.PdfReader(str(file_path))
        except Exception as e:
            raise DocumentFormatError(
                f"Failed to parse PDF document: {e}", format_type="pdf"
            ) from e

        if reader.is_encrypted:
            raise DocumentFormatError(
                "Encrypted or password-protected PDF files are not supported", format_type="pdf"
            )

        page_texts = []
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_texts.append(page_text.strip())
            except Exception as e:
                raise DocumentFormatError(
                    f"Error extracting text from PDF page {idx + 1}: {e}",
                    format_type="pdf",
                ) from e

        combined_text = "\n\n".join(page_texts)
        pdf_meta = reader.metadata or {}
        doc_title = pdf_meta.get("/Title") if hasattr(pdf_meta, "get") else None

        metadata: Dict[str, Any] = {
            "format": DocumentFormat.PDF.value,
            "raw_size_bytes": size,
            "page_count": len(reader.pages),
        }
        if doc_title:
            metadata["title"] = str(doc_title)

        return combined_text, metadata


def get_extractor(file_path: Union[str, Path]) -> BaseExtractor:
    """Resolve the appropriate extractor instance for the given file path.

    Raises:
        DocumentFormatError: If file format is not supported or is an executable extension.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Block executable or script file extensions strictly
    prohibited_extensions = {
        ".exe",
        ".dll",
        ".so",
        ".bin",
        ".sh",
        ".bat",
        ".cmd",
        ".ps1",
        ".py",
        ".rb",
        ".js",
        ".ts",
    }
    if ext in prohibited_extensions:
        raise DocumentFormatError(
            f"File extension '{ext}' is an executable or code script and is strictly prohibited from RAG document ingestion",
            format_type=ext,
        )

    # Map supported formats
    doc_format = DocumentFormat.from_extension(ext)
    if doc_format == DocumentFormat.MARKDOWN:
        return MarkdownExtractor()
    elif doc_format == DocumentFormat.TEXT:
        return PlainTextExtractor()
    elif doc_format == DocumentFormat.JSON:
        return JSONExtractor()
    elif doc_format == DocumentFormat.YAML:
        return YAMLExtractor()
    elif doc_format == DocumentFormat.PDF:
        return PDFExtractor()
    else:  # pragma: no cover
        raise DocumentFormatError(
            f"No extractor registered for format '{doc_format}'", format_type=str(doc_format)
        )
