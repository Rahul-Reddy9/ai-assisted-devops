# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Text preprocessing, normalization, secret scrubbing, and deduplication for RAG ingestion."""

import hashlib
import re
from typing import Any, Dict, Set, Tuple

from devops_ai.core.logging import scrub_secrets


class DocumentPreprocessor:
    """Preprocesses raw extracted document text for downstream chunking and indexing."""

    def __init__(self) -> None:
        self._seen_hashes: Set[str] = set()

    @staticmethod
    def compute_hash(text: str) -> str:
        """Calculate deterministic SHA-256 hash of normalized text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if a document with this content hash has already been processed."""
        return content_hash in self._seen_hashes

    def register_hash(self, content_hash: str) -> None:
        """Register a content hash into the seen deduplication registry."""
        self._seen_hashes.add(content_hash)

    def clear_registry(self) -> None:
        """Clear the deduplication registry."""
        self._seen_hashes.clear()

    def preprocess(
        self,
        raw_text: str,
        scrub_credentials: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Normalize, sanitize, and extract structural metrics from raw document text.

        Args:
            raw_text: Raw text extracted from document.
            scrub_credentials: If True, scrubs API keys, passwords, and tokens.

        Returns:
            Tuple of (normalized_text, preprocessor_metrics).
        """
        if not raw_text:
            return "", {
                "character_count": 0,
                "line_count": 0,
                "is_empty": True,
                "content_hash": hashlib.sha256(b"").hexdigest(),
                "has_code_blocks": False,
            }

        # 1. Normalize line endings (CRLF and CR -> LF)
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Strip null bytes and non-printable control characters (preserve \t and \n)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # 3. Secret scrubbing across documentation text
        if scrub_credentials:
            text = scrub_secrets(text)

        # 4. Whitespace cleanup: strip trailing spaces on lines while preserving code indentation
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 5. Collapse excessive blank lines (3 or more newlines become 2 newlines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # 6. Check for code blocks
        has_code_blocks = bool(re.search(r"```[\s\S]*?```", text))

        # 7. Compute deterministic content hash
        content_hash = self.compute_hash(text)
        is_empty = len(text.strip()) == 0

        metrics: Dict[str, Any] = {
            "character_count": len(text),
            "line_count": len(text.splitlines()) if text else 0,
            "is_empty": is_empty,
            "content_hash": content_hash,
            "has_code_blocks": has_code_blocks,
        }
        return text, metrics
