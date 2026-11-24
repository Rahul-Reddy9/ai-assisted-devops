# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Intelligent structure-aware document chunker for RAG document processing."""

import hashlib
import re
from typing import Any, Dict, List, Optional

from devops_ai.core.exceptions import ChunkingError
from devops_ai.rag.models import DocumentChunk, SourceDocument


class SmartChunker:
    """Structure-aware chunker preserving markdown code blocks, paragraphs, and sentence boundaries."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _split_into_structural_blocks(text: str) -> List[str]:
        """Split text into structural blocks, keeping markdown code blocks intact."""
        # Regex matching fenced code blocks (```...```)
        pattern = re.compile(r"(```[\s\S]*?```)")
        parts = pattern.split(text)

        blocks: List[str] = []
        for part in parts:
            if not part:
                continue
            if part.startswith("```") and part.endswith("```"):
                # Atomic code block
                blocks.append(part.strip())
            else:
                # Regular text: split by paragraph breaks (\n\n)
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", part) if p.strip()]
                blocks.extend(paragraphs)

        return blocks

    def _split_oversized_block(self, block: str) -> List[str]:
        """Sub-divide a block that exceeds chunk_size into smaller line or sentence segments."""
        if len(block) <= self.chunk_size:
            return [block]

        # For code blocks or structured text, prefer splitting on line breaks
        if block.startswith("```") and block.endswith("```"):
            lines = block.split("\n")
            sub_chunks: List[str] = []
            curr: List[str] = []
            curr_len = 0

            for line in lines:
                line_len = len(line) + 1
                if curr_len + line_len > self.chunk_size and curr:
                    sub_chunks.append("\n".join(curr))
                    curr = [line]
                    curr_len = line_len
                else:
                    curr.append(line)
                    curr_len += line_len

            if curr:
                sub_chunks.append("\n".join(curr))
            return sub_chunks

        # For regular paragraphs, prefer splitting on sentence boundaries
        sentence_delimiters = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_delimiters.split(block)
        sub_chunks = []
        curr = []
        curr_len = 0

        for s in sentences:
            s_len = len(s) + 1
            if curr_len + s_len > self.chunk_size and curr:
                sub_chunks.append(" ".join(curr))
                curr = [s]
                curr_len = s_len
            else:
                curr.append(s)
                curr_len += s_len

        if curr:
            sub_chunks.append(" ".join(curr))

        # Fallback for unbroken strings with no spaces or sentences
        final_chunks: List[str] = []
        for sc in sub_chunks:
            if len(sc) > self.chunk_size:
                for i in range(0, len(sc), self.chunk_size - self.chunk_overlap):
                    slice_text = sc[i : i + self.chunk_size].strip()
                    if slice_text:
                        final_chunks.append(slice_text)
            else:
                final_chunks.append(sc)

        return final_chunks

    def chunk_document(
        self,
        document: SourceDocument,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split a SourceDocument into ordered, indexed DocumentChunk objects with overlap."""
        text = document.raw_content.strip()
        if not text:
            return []

        doc_id = document.document_id
        doc_meta = document.metadata.model_dump()
        if extra_metadata:
            doc_meta.update(extra_metadata)

        try:
            # 1. Break into structural units
            raw_blocks = self._split_into_structural_blocks(text)
            blocks: List[str] = []
            for b in raw_blocks:
                if len(b) > self.chunk_size:
                    blocks.extend(self._split_oversized_block(b))
                else:
                    blocks.append(b)

            # 2. Assemble chunks with configured overlap
            chunk_texts: List[str] = []
            current_chunk_parts: List[str] = []
            current_len = 0

            for block in blocks:
                block_len = len(block)
                # If adding block exceeds chunk_size and we already have content
                if current_len + block_len + 2 > self.chunk_size and current_chunk_parts:
                    chunk_str = "\n\n".join(current_chunk_parts).strip()
                    if chunk_str:
                        chunk_texts.append(chunk_str)

                    # Build overlap from tail of emitted chunk
                    if self.chunk_overlap > 0 and len(chunk_str) > self.chunk_overlap:
                        overlap_seed = chunk_str[-self.chunk_overlap :].lstrip()
                        current_chunk_parts = [overlap_seed, block]
                        current_len = len(overlap_seed) + block_len + 2
                    else:
                        current_chunk_parts = [block]
                        current_len = block_len
                else:
                    current_chunk_parts.append(block)
                    current_len += block_len + 2

            if current_chunk_parts:
                chunk_str = "\n\n".join(current_chunk_parts).strip()
                if chunk_str and (not chunk_texts or chunk_str != chunk_texts[-1]):
                    chunk_texts.append(chunk_str)

            # 3. Create DocumentChunk models
            chunks: List[DocumentChunk] = []
            total_chunks = len(chunk_texts)
            doc_id_clean = re.sub(r"[^A-Za-z0-9_\-]", "", doc_id)

            for idx, c_text in enumerate(chunk_texts):
                c_clean = c_text.strip()
                if not c_clean:
                    continue

                chunk_hash = hashlib.sha256(c_clean.encode("utf-8")).hexdigest()
                chunk_id = f"CHK-{doc_id_clean}-{idx:04d}"
                has_code = bool(re.search(r"```[\s\S]*?```", c_clean))

                chunk_meta = dict(doc_meta)
                chunk_meta.update(
                    {
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "has_code_block": has_code,
                        "chunk_size": len(c_clean),
                    }
                )

                chunk_obj = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    content=c_clean,
                    character_count=len(c_clean),
                    content_hash=chunk_hash,
                    metadata=chunk_meta,
                )
                chunks.append(chunk_obj)

            # Re-normalize total_chunks in case any empty chunks were discarded
            actual_total = len(chunks)
            for c in chunks:
                c.total_chunks = actual_total
                c.metadata["total_chunks"] = actual_total

            return chunks

        except Exception as e:
            raise ChunkingError(f"Failed to chunk document: {e}", doc_id=doc_id) from e
