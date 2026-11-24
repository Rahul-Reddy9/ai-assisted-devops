# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""High-level document ingestion and chunking pipeline for RAG subsystems."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from devops_ai.core.config import Settings, get_settings
from devops_ai.core.exceptions import (
    DocumentFormatError,
    DocumentIngestionError,
    DocumentSecurityError,
    DocumentSizeLimitError,
    EmbeddingProviderError,
    VectorStoreError,
)
from devops_ai.core.logging import get_logger
from devops_ai.rag.chunker import SmartChunker
from devops_ai.rag.embeddings.base import BaseEmbeddingProvider
from devops_ai.rag.extractors import get_extractor
from devops_ai.rag.models import (
    DocumentChunk,
    DocumentFormat,
    DocumentMetadata,
    IngestionResult,
    IngestionStatus,
    SourceDocument,
)
from devops_ai.rag.preprocessor import DocumentPreprocessor
from devops_ai.rag.vector_store.base import BaseVectorStore

logger = get_logger("rag.pipeline")


class DocumentIngestionPipeline:
    """Coordinates format extraction, normalization, deduplication, and chunking."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        max_file_size_bytes: Optional[int] = None,
        max_pdf_pages: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        skip_duplicates: bool = True,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_file_size_bytes = (
            max_file_size_bytes
            if max_file_size_bytes is not None
            else self.settings.rag_max_file_size_bytes
        )
        self.max_pdf_pages = (
            max_pdf_pages if max_pdf_pages is not None else self.settings.rag_max_pdf_pages
        )
        c_size = chunk_size if chunk_size is not None else self.settings.rag_chunk_size
        c_overlap = chunk_overlap if chunk_overlap is not None else self.settings.rag_chunk_overlap

        self.preprocessor = DocumentPreprocessor()
        self.chunker = SmartChunker(chunk_size=c_size, chunk_overlap=c_overlap)
        self.skip_duplicates = skip_duplicates
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.logger = get_logger("rag.pipeline")

    def _index_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Generate vector embeddings and index chunks into vector store if configured."""
        if not chunks or self.vector_store is None or self.embedding_provider is None:
            return {}

        texts = [chunk.content for chunk in chunks]
        batch_size = self.settings.rag_embedding_batch_size
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.embedding_provider.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        self.vector_store.add_chunks(chunks=chunks, embeddings=all_embeddings)
        return {
            "indexed_chunks": len(chunks),
            "vector_store": getattr(
                self.vector_store, "store_name", type(self.vector_store).__name__
            ),
            "embedding_provider": getattr(
                self.embedding_provider,
                "provider_name",
                type(self.embedding_provider).__name__,
            ),
        }

    def _validate_path_security(
        self,
        file_path: Path,
        allowed_root: Optional[Path] = None,
    ) -> Path:
        """Enforce path traversal and directory containment security checks."""
        # Check raw parts for path traversal attempts
        parts = file_path.parts
        if ".." in parts:
            raise DocumentSecurityError(f"Path traversal attempt detected in path: '{file_path}'")

        try:
            resolved = file_path.resolve()
        except OSError as e:
            raise DocumentIngestionError(
                f"Cannot resolve path '{file_path}': {e}", source=str(file_path)
            ) from e

        if allowed_root:
            try:
                root_resolved = allowed_root.resolve()
            except OSError as e:
                raise DocumentSecurityError(
                    f"Invalid allowed root directory '{allowed_root}': {e}"
                ) from e

            try:
                resolved.relative_to(root_resolved)
            except ValueError as e:
                raise DocumentSecurityError(
                    f"Access denied: file path '{resolved}' escapes allowed root directory '{root_resolved}'"
                ) from e

        return resolved

    def ingest_file(
        self,
        file_path: Union[str, Path],
        allowed_root: Optional[Union[str, Path]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> IngestionResult:
        """Ingest, extract, normalize, and chunk a document from disk."""
        path_obj = Path(file_path)
        root_obj = Path(allowed_root) if allowed_root else None

        registered_hash: Optional[str] = None
        doc_id: Optional[str] = None
        try:
            # 1. Path traversal & security checks
            resolved_path = self._validate_path_security(path_obj, root_obj)

            if not resolved_path.exists():
                raise DocumentIngestionError(
                    f"File not found: {resolved_path}", source=str(resolved_path)
                )
            if not resolved_path.is_file():
                raise DocumentIngestionError(
                    f"Target path is not a regular file: {resolved_path}", source=str(resolved_path)
                )

            # 2. Select extractor and extract text
            extractor = get_extractor(resolved_path, max_pdf_pages=self.max_pdf_pages)
            raw_text, extractor_meta = extractor.extract(
                resolved_path, max_bytes=self.max_file_size_bytes
            )

            # 3. Preprocess and normalize text
            normalized_text, preprocessor_metrics = self.preprocessor.preprocess(raw_text)

            if preprocessor_metrics.get("is_empty", False):
                self.logger.warning(
                    f"Document '{resolved_path.name}' is empty after normalization; skipping chunking"
                )
                return IngestionResult(
                    source_path=str(resolved_path),
                    status=IngestionStatus.SKIPPED,
                    error="Document is empty or contains no readable non-whitespace text",
                    metadata={"empty_content": True},
                )

            # 4. Deduplication
            content_hash = preprocessor_metrics["content_hash"]
            if self.skip_duplicates and self.preprocessor.is_duplicate(content_hash):
                self.logger.info(
                    f"Skipping duplicate document '{resolved_path.name}' with content hash: {content_hash[:12]}"
                )
                return IngestionResult(
                    source_path=str(resolved_path),
                    status=IngestionStatus.SKIPPED,
                    content_hash=content_hash,
                    error="Duplicate document detected (identical content hash already ingested)",
                    metadata={"duplicate": True, "content_hash": content_hash},
                )

            self.preprocessor.register_hash(content_hash)
            registered_hash = content_hash

            # 5. Build SourceDocument model
            doc_format = DocumentFormat.from_extension(resolved_path.suffix)
            doc_id = f"DOC-{content_hash[:12].upper()}"

            all_custom_attrs = dict(extractor_meta)
            if custom_metadata:
                all_custom_attrs.update(custom_metadata)

            file_size = resolved_path.stat().st_size
            doc_meta = DocumentMetadata(
                source_filename=resolved_path.name,
                source_format=doc_format,
                source_path=str(resolved_path),
                content_hash=content_hash,
                file_size_bytes=file_size,
                character_count=len(normalized_text),
                custom_attributes=all_custom_attrs,
            )

            source_doc = SourceDocument(
                document_id=doc_id,
                raw_content=normalized_text,
                metadata=doc_meta,
            )

            # 6. Generate intelligent chunks
            chunks = self.chunker.chunk_document(source_doc)

            # 7. Generate embeddings and index chunks into vector store (if configured)
            indexing_meta = self._index_chunks(chunks)

            # 8. Purge obsolete chunks originating from prior versions of this file path
            if self.vector_store is not None:
                purged = self.vector_store.delete_by_source_path(
                    source_path=str(resolved_path),
                    exclude_document_id=doc_id,
                )
                if purged > 0:
                    self.logger.debug(
                        f"Purged {purged} obsolete chunks for source file '{resolved_path.name}'"
                    )

            self.logger.info(
                f"Ingested document '{resolved_path.name}' ({doc_id}) into {len(chunks)} chunks "
                f"({len(normalized_text)} chars)"
            )

            result_metadata: Dict[str, Any] = {
                "file_size_bytes": file_size,
                "character_count": len(normalized_text),
                "has_code_blocks": preprocessor_metrics.get("has_code_blocks", False),
            }
            if indexing_meta:
                result_metadata.update(indexing_meta)

            return IngestionResult(
                document_id=doc_id,
                source_path=str(resolved_path),
                status=IngestionStatus.SUCCESS,
                chunks=chunks,
                total_chunks=len(chunks),
                content_hash=content_hash,
                metadata=result_metadata,
            )

        except (
            DocumentSecurityError,
            DocumentSizeLimitError,
            DocumentFormatError,
            DocumentIngestionError,
            VectorStoreError,
            EmbeddingProviderError,
        ) as e:
            if registered_hash:
                self.preprocessor.unregister_hash(registered_hash)
            if doc_id and self.vector_store:
                try:
                    self.vector_store.delete_document(doc_id)
                except Exception as cleanup_err:
                    self.logger.warning(
                        f"Failed to cleanup vectors for document '{doc_id}': {cleanup_err}"
                    )
            self.logger.error(f"Ingestion error on '{path_obj}': {e}")
            if raise_on_error:
                raise
            return IngestionResult(
                source_path=str(path_obj),
                status=IngestionStatus.FAILED,
                error=str(e),
                metadata={"error_type": type(e).__name__},
            )
        except Exception as e:
            if registered_hash:
                self.preprocessor.unregister_hash(registered_hash)
            if doc_id and self.vector_store:
                try:
                    self.vector_store.delete_document(doc_id)
                except Exception as cleanup_err:
                    self.logger.warning(
                        f"Failed to cleanup vectors for document '{doc_id}': {cleanup_err}"
                    )
            self.logger.error(f"Unexpected error ingesting '{path_obj}': {e}")
            if raise_on_error:
                raise DocumentIngestionError(
                    f"Unexpected failure: {e}", source=str(path_obj)
                ) from e
            return IngestionResult(
                source_path=str(path_obj),
                status=IngestionStatus.FAILED,
                error=f"Unexpected error: {e}",
                metadata={"error_type": "UnexpectedException"},
            )

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        allowed_root: Optional[Union[str, Path]] = None,
        raise_on_error: bool = False,
    ) -> List[IngestionResult]:
        """Ingest all supported documents from a directory."""
        directory = Path(dir_path)
        if not directory.exists():
            raise DocumentIngestionError(f"Directory not found: {directory}", source=str(directory))
        if not directory.is_dir():
            raise DocumentIngestionError(
                f"Path is not a directory: {directory}", source=str(directory)
            )

        results: List[IngestionResult] = []
        supported_exts = {".md", ".markdown", ".txt", ".text", ".pdf", ".json", ".yaml", ".yml"}

        pattern = "**/*" if recursive else "*"
        for item in sorted(directory.glob(pattern)):
            if item.is_file() and item.suffix.lower() in supported_exts:
                res = self.ingest_file(
                    file_path=item,
                    allowed_root=allowed_root,
                    raise_on_error=raise_on_error,
                )
                results.append(res)

        return results

    def ingest_text(
        self,
        content: str,
        source_name: str,
        format: DocumentFormat = DocumentFormat.TEXT,
        custom_metadata: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> IngestionResult:
        """Directly ingest an in-memory text payload as a document."""
        registered_hash: Optional[str] = None
        doc_id: Optional[str] = None
        source_uri = f"memory://{source_name}"
        try:
            normalized_text, preprocessor_metrics = self.preprocessor.preprocess(content)
            if preprocessor_metrics.get("is_empty", False):
                return IngestionResult(
                    source_path=source_name,
                    status=IngestionStatus.SKIPPED,
                    error="Document text is empty",
                )

            content_hash = preprocessor_metrics["content_hash"]
            if self.skip_duplicates and self.preprocessor.is_duplicate(content_hash):
                return IngestionResult(
                    source_path=source_name,
                    status=IngestionStatus.SKIPPED,
                    content_hash=content_hash,
                    error="Duplicate text content detected",
                )

            self.preprocessor.register_hash(content_hash)
            registered_hash = content_hash
            doc_id = f"DOC-{content_hash[:12].upper()}"

            meta = DocumentMetadata(
                source_filename=source_name,
                source_format=format,
                source_path=source_uri,
                content_hash=content_hash,
                file_size_bytes=len(content.encode("utf-8")),
                character_count=len(normalized_text),
                custom_attributes=custom_metadata or {},
            )
            source_doc = SourceDocument(
                document_id=doc_id,
                raw_content=normalized_text,
                metadata=meta,
            )

            chunks = self.chunker.chunk_document(source_doc)
            indexing_meta = self._index_chunks(chunks)

            # Purge obsolete chunks originating from prior in-memory versions of this source
            if self.vector_store is not None:
                self.vector_store.delete_by_source_path(
                    source_path=source_uri,
                    exclude_document_id=doc_id,
                )

            result_metadata: Dict[str, Any] = {
                "in_memory": True,
                "character_count": len(normalized_text),
                "has_code_blocks": preprocessor_metrics.get("has_code_blocks", False),
            }
            if indexing_meta:
                result_metadata.update(indexing_meta)

            return IngestionResult(
                document_id=doc_id,
                source_path=source_uri,
                status=IngestionStatus.SUCCESS,
                chunks=chunks,
                total_chunks=len(chunks),
                content_hash=content_hash,
                metadata=result_metadata,
            )
        except (
            DocumentSecurityError,
            DocumentSizeLimitError,
            DocumentFormatError,
            DocumentIngestionError,
            VectorStoreError,
            EmbeddingProviderError,
        ) as e:
            if registered_hash:
                self.preprocessor.unregister_hash(registered_hash)
            if doc_id and self.vector_store:
                try:
                    self.vector_store.delete_document(doc_id)
                except Exception as cleanup_err:
                    self.logger.warning(
                        f"Failed to cleanup vectors for in-memory document '{doc_id}': {cleanup_err}"
                    )
            self.logger.error(f"Ingestion error on text '{source_name}': {e}")
            if raise_on_error:
                raise
            return IngestionResult(
                source_path=source_uri,
                status=IngestionStatus.FAILED,
                error=str(e),
                metadata={"error_type": type(e).__name__},
            )
        except Exception as e:
            if registered_hash:
                self.preprocessor.unregister_hash(registered_hash)
            if doc_id and self.vector_store:
                try:
                    self.vector_store.delete_document(doc_id)
                except Exception as cleanup_err:
                    self.logger.warning(
                        f"Failed to cleanup vectors for in-memory document '{doc_id}': {cleanup_err}"
                    )
            self.logger.error(f"Unexpected error ingesting text '{source_name}': {e}")
            if raise_on_error:
                raise DocumentIngestionError(f"Unexpected failure: {e}", source=source_uri) from e
            return IngestionResult(
                source_path=source_uri,
                status=IngestionStatus.FAILED,
                error=f"Unexpected error: {e}",
                metadata={"error_type": "UnexpectedException"},
            )
