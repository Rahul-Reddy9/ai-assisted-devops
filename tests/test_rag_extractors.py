# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying format-specific document extractors and file validations."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pypdf
import pytest

from devops_ai.core.exceptions import (
    DocumentFormatError,
    DocumentIngestionError,
    DocumentPageLimitError,
    DocumentSizeLimitError,
)
from devops_ai.rag.extractors import (
    JSONExtractor,
    MarkdownExtractor,
    PDFExtractor,
    PlainTextExtractor,
    YAMLExtractor,
    get_extractor,
)
from devops_ai.rag.models import DocumentFormat


def test_markdown_extractor(tmp_path: Path):
    """Verify MarkdownExtractor extracts title and preserves technical blocks."""
    md_file = tmp_path / "k8s_runbook.md"
    md_content = (
        "# Kubernetes Cluster Outage Runbook\n\n"
        "## Diagnostic Steps\n"
        "1. Check node status:\n"
        "```bash\n"
        "kubectl get nodes -o wide\n"
        "```\n"
        "2. Review pod restart counts.\n"
    )
    md_file.write_text(md_content, encoding="utf-8")

    extractor = MarkdownExtractor()
    text, meta = extractor.extract(md_file, max_bytes=1024 * 1024)

    assert meta["format"] == DocumentFormat.MARKDOWN.value
    assert meta["title"] == "Kubernetes Cluster Outage Runbook"
    assert "kubectl get nodes -o wide" in text
    assert meta["raw_size_bytes"] == md_file.stat().st_size


def test_plain_text_extractor(tmp_path: Path):
    """Verify PlainTextExtractor extracts plain text documents."""
    txt_file = tmp_path / "incident.txt"
    txt_content = "Outage on ingress gateway observed at 14:00 UTC."
    txt_file.write_text(txt_content, encoding="utf-8")

    extractor = PlainTextExtractor()
    text, meta = extractor.extract(txt_file, max_bytes=1024 * 1024)

    assert text == txt_content
    assert meta["format"] == DocumentFormat.TEXT.value
    assert meta["raw_size_bytes"] == len(txt_content.encode("utf-8"))


def test_json_extractor_valid_and_malformed(tmp_path: Path):
    """Verify JSONExtractor normalizes structured JSON and catches syntax errors."""
    valid_json = tmp_path / "service_catalog.json"
    valid_json.write_text('{"service": "auth-api", "replicas": 3, "tier": 1}', encoding="utf-8")

    extractor = JSONExtractor()
    text, meta = extractor.extract(valid_json, max_bytes=1024 * 1024)

    assert meta["format"] == DocumentFormat.JSON.value
    assert meta["keys_count"] == 3
    assert '"auth-api"' in text

    # Malformed JSON
    invalid_json = tmp_path / "bad.json"
    invalid_json.write_text('{"service": "broken", invalid_syntax', encoding="utf-8")

    with pytest.raises(DocumentFormatError) as exc_info:
        extractor.extract(invalid_json, max_bytes=1024 * 1024)
    assert "Malformed JSON syntax" in str(exc_info.value)


def test_yaml_extractor_valid_and_malformed(tmp_path: Path):
    """Verify YAMLExtractor parses multi-doc YAML safely and catches syntax errors."""
    valid_yaml = tmp_path / "deploy.yaml"
    valid_yaml.write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: payment-svc\n---\napiVersion: v1\nkind: Service\n",
        encoding="utf-8",
    )

    extractor = YAMLExtractor()
    text, meta = extractor.extract(valid_yaml, max_bytes=1024 * 1024)

    assert meta["format"] == DocumentFormat.YAML.value
    assert meta["document_count"] == 2
    assert "payment-svc" in text

    # Malformed YAML
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed_list", encoding="utf-8")

    with pytest.raises(DocumentFormatError) as exc_info:
        extractor.extract(bad_yaml, max_bytes=1024 * 1024)
    assert "Malformed YAML syntax" in str(exc_info.value)


def test_pdf_extractor_success_and_encrypted(tmp_path: Path):
    """Verify PDFExtractor extracts text across pages and rejects encrypted PDFs."""
    pdf_file = tmp_path / "manual.pdf"

    # Create synthetic test PDF using pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_blank_page(width=300, height=300)
    with open(pdf_file, "wb") as fh:
        writer.write(fh)

    extractor = PDFExtractor()

    # Mock extract_text on page objects to simulate text extraction
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        p1 = MagicMock()
        p1.extract_text.return_value = "Page 1: Architecture Overview"
        p2 = MagicMock()
        p2.extract_text.return_value = "Page 2: Failover Procedures"
        mock_instance.pages = [p1, p2]
        mock_instance.metadata = {"/Title": "SRE Manual"}
        mock_reader_cls.return_value = mock_instance

        text, meta = extractor.extract(pdf_file, max_bytes=1024 * 1024)

        assert "Page 1: Architecture Overview" in text
        assert "Page 2: Failover Procedures" in text
        assert meta["page_count"] == 2
        assert meta["title"] == "SRE Manual"

    # Encrypted PDF rejection
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_instance = MagicMock()
        mock_instance.is_encrypted = True
        mock_reader_cls.return_value = mock_instance

        with pytest.raises(DocumentFormatError) as exc_info:
            extractor.extract(pdf_file, max_bytes=1024 * 1024)
        assert "Encrypted or password-protected" in str(exc_info.value)


def test_pdf_extractor_page_limit_enforcement(tmp_path: Path):
    """Verify PDFExtractor enforces configured page limit and validates bounds."""
    pdf_file = tmp_path / "doc.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(pdf_file, "wb") as fh:
        writer.write(fh)

    # Invalid max_pages validation
    with pytest.raises(ValueError):
        PDFExtractor(max_pages=0)
    with pytest.raises(ValueError):
        PDFExtractor(max_pages=-5)

    # Below page limit: accepted
    extractor_limit_5 = PDFExtractor(max_pages=5)
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        p_mock = MagicMock()
        p_mock.extract_text.return_value = "Page content"
        mock_instance.pages = [p_mock] * 3  # 3 pages
        mock_reader_cls.return_value = mock_instance

        text, meta = extractor_limit_5.extract(pdf_file, max_bytes=1024 * 1024)
        assert meta["page_count"] == 3

    # Exactly at page limit: accepted
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        p_mock = MagicMock()
        p_mock.extract_text.return_value = "Page content"
        mock_instance.pages = [p_mock] * 5  # Exactly 5 pages
        mock_reader_cls.return_value = mock_instance

        text, meta = extractor_limit_5.extract(pdf_file, max_bytes=1024 * 1024)
        assert meta["page_count"] == 5

    # Exceeds page limit: rejected with DocumentPageLimitError
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        p_mock = MagicMock()
        p_mock.extract_text.return_value = "Page content"
        mock_instance.pages = [p_mock] * 6  # 6 pages > 5 limit
        mock_reader_cls.return_value = mock_instance

        with pytest.raises(DocumentPageLimitError) as exc_info:
            extractor_limit_5.extract(pdf_file, max_bytes=1024 * 1024)

        error_msg = str(exc_info.value)
        assert "exceeds page limit of 5 pages" in error_msg
        assert "document contains 6 pages" in error_msg
        assert "bytes" not in error_msg
        assert "Document size limit exceeded" not in error_msg
        assert exc_info.value.page_count == 6
        assert exc_info.value.limit == 5
        assert exc_info.value.size == 6
        assert isinstance(exc_info.value, DocumentSizeLimitError)


def test_extractor_file_size_limit(tmp_path: Path):
    """Verify DocumentSizeLimitError is raised when file exceeds max_bytes."""
    large_file = tmp_path / "large.txt"
    large_file.write_text("A" * 5000, encoding="utf-8")

    extractor = PlainTextExtractor()
    with pytest.raises(DocumentSizeLimitError) as exc_info:
        extractor.extract(large_file, max_bytes=1000)
    assert "exceeds allowed limit of 1000 bytes" in str(exc_info.value)


def test_extractor_non_existent_file():
    """Verify DocumentIngestionError when file cannot be accessed."""
    extractor = PlainTextExtractor()
    with pytest.raises(DocumentIngestionError):
        extractor.extract(Path("/non/existent/missing_file.txt"), max_bytes=1000)


def test_get_extractor_resolution_and_prohibitions():
    """Verify get_extractor resolves correct extractors and rejects executable extensions."""
    assert isinstance(get_extractor("doc.md"), MarkdownExtractor)
    assert isinstance(get_extractor("doc.markdown"), MarkdownExtractor)
    assert isinstance(get_extractor("log.txt"), PlainTextExtractor)
    assert isinstance(get_extractor("data.json"), JSONExtractor)
    assert isinstance(get_extractor("spec.yaml"), YAMLExtractor)
    assert isinstance(get_extractor("spec.yml"), YAMLExtractor)
    assert isinstance(get_extractor("guide.pdf"), PDFExtractor)

    # Prohibited executable / script extensions
    for ext in [".exe", ".sh", ".py", ".bin", ".dll", ".bat"]:
        with pytest.raises(DocumentFormatError) as exc_info:
            get_extractor(f"danger{ext}")
        assert "strictly prohibited" in str(exc_info.value)

    # Unsupported non-executable extension
    with pytest.raises(DocumentFormatError) as exc_info:
        get_extractor("data.parquet")
    assert "Unsupported file extension" in str(exc_info.value)
