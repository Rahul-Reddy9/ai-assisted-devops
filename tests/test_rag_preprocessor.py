# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for DocumentPreprocessor normalization, secret scrubbing, and deduplication."""

from devops_ai.rag.preprocessor import DocumentPreprocessor


def test_preprocessor_line_ending_and_whitespace_normalization():
    """Verify CRLF conversion, trailing space removal, and blank line collapsing."""
    preprocessor = DocumentPreprocessor()
    raw = (
        "Heading line   \r\n"
        "\r\n"
        "\r\n"
        "\r\n"
        "Paragraph after excessive newlines.\r\n"
        "    indented_code_line()   \r\n"
    )

    clean, metrics = preprocessor.preprocess(raw)

    # Check line endings and whitespace
    assert "\r" not in clean
    assert "\n\n\n" not in clean
    assert "Heading line\n\nParagraph after excessive newlines." in clean
    assert "    indented_code_line()" in clean
    assert metrics["is_empty"] is False
    assert metrics["character_count"] == len(clean)


def test_preprocessor_strips_null_and_control_bytes():
    """Verify unprintable control characters and null bytes are removed while tabs/newlines remain."""
    preprocessor = DocumentPreprocessor()
    raw = "Valid line\x00with null\x07bell and \t tab.\n"

    clean, _ = preprocessor.preprocess(raw)
    assert "\x00" not in clean
    assert "\x07" not in clean
    assert "\t" in clean
    assert "Valid linewith nullbell and \t tab." == clean


def test_preprocessor_secret_scrubbing():
    """Verify secrets and API keys are scrubbed during preprocessing."""
    preprocessor = DocumentPreprocessor()
    # Dynamically assembled synthetic key to prevent static secret scanner alerts
    fake_key = "".join(["AIzaSy", "TestDocKey", "1234567890abcdef"])
    raw = (
        f"Connecting to telemetry service with query key {fake_key}\n"
        "password = 'super_secret_db_pass!'"
    )

    clean, _ = preprocessor.preprocess(raw, scrub_credentials=True)
    assert "AIzaSy" not in clean
    assert "[REDACTED_API_KEY]" in clean
    assert "super_secret_db_pass" not in clean
    assert "[REDACTED_SECRET]" in clean


def test_preprocessor_code_block_detection():
    """Verify markdown code block presence is accurately detected."""
    preprocessor = DocumentPreprocessor()

    no_code = "Just regular documentation text without any fences."
    _, metrics1 = preprocessor.preprocess(no_code)
    assert metrics1["has_code_blocks"] is False

    with_code = "Run this command:\n```bash\nkubectl get pods\n```\nDone."
    _, metrics2 = preprocessor.preprocess(with_code)
    assert metrics2["has_code_blocks"] is True


def test_preprocessor_deduplication_registry():
    """Verify content hashing and deduplication tracking."""
    preprocessor = DocumentPreprocessor()

    doc1 = "Cluster provisioning guidelines for production region us-east-1."
    doc2 = "Cluster provisioning guidelines for production region us-east-1."
    doc3 = "Different operational guidelines."

    _, m1 = preprocessor.preprocess(doc1)
    _, m2 = preprocessor.preprocess(doc2)
    _, m3 = preprocessor.preprocess(doc3)

    assert m1["content_hash"] == m2["content_hash"]
    assert m1["content_hash"] != m3["content_hash"]

    # Deduplication registry
    hash1 = m1["content_hash"]
    assert preprocessor.is_duplicate(hash1) is False

    preprocessor.register_hash(hash1)
    assert preprocessor.is_duplicate(hash1) is True

    preprocessor.clear_registry()
    assert preprocessor.is_duplicate(hash1) is False


def test_preprocessor_empty_content_handling():
    """Verify empty and whitespace-only documents are flagged as empty."""
    preprocessor = DocumentPreprocessor()

    clean1, m1 = preprocessor.preprocess("")
    assert clean1 == ""
    assert m1["is_empty"] is True
    assert m1["character_count"] == 0

    clean2, m2 = preprocessor.preprocess("   \n\n\t  \n  ")
    assert clean2 == ""
    assert m2["is_empty"] is True
