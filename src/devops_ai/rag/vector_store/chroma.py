# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""ChromaDB persistent vector store implementation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import (
    EmbeddingDimensionMismatchError,
    VectorStoreError,
)
from devops_ai.core.logging import get_logger, scrub_secrets
from devops_ai.rag.models import DocumentChunk, RetrievalResult
from devops_ai.rag.vector_store.base import BaseVectorStore

logger = get_logger("rag.vector_store.chroma")


class ChromaVectorStore(BaseVectorStore):
    """Production-grade vector store backed by ChromaDB with local disk persistence."""

    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        collection_name: str = "devops_ai_rag",
        dimension: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._collection_name = collection_name
        self._dimension = dimension if dimension is not None else settings.rag_embedding_dimension
        self._embedding_model = embedding_model or settings.rag_embedding_model
        self.store_name = "chroma"

        if self._dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {self._dimension}")

        raw_path = persist_directory or settings.rag_vector_store_path
        self._is_ephemeral = raw_path == ":memory:"

        try:
            import chromadb

            if self._is_ephemeral:
                self._client = chromadb.EphemeralClient()
                self._persist_path: Optional[Path] = None
            else:
                self._persist_path = Path(raw_path).resolve()
                self._persist_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self._persist_path))

            # Initialize or retrieve collection with cosine distance space and model metadata
            target_metadata = {
                "hnsw:space": "cosine",
                "dimension": self._dimension,
                "embedding_model": self._embedding_model,
            }
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata=target_metadata,
            )

            # Inspect and validate existing collection metadata
            existing_meta = self._collection.metadata or {}
            stored_dim = existing_meta.get("dimension")
            stored_model = existing_meta.get("embedding_model")

            if stored_dim is None or stored_model is None:
                raise VectorStoreError(
                    f"Vector store collection '{self._collection_name}' lacks required embedding metadata "
                    f"('dimension' and 'embedding_model'). Cannot safely verify compatibility.",
                    store_name=self.store_name,
                )

            if int(stored_dim) != self._dimension:
                raise EmbeddingDimensionMismatchError(
                    f"Vector store collection '{self._collection_name}' dimension {stored_dim} "
                    f"does not match configured {self._dimension}",
                    expected=self._dimension,
                    actual=int(stored_dim),
                )

            if str(stored_model) != self._embedding_model:
                raise VectorStoreError(
                    f"Vector store collection '{self._collection_name}' was created with embedding model "
                    f"'{stored_model}', but configured model is '{self._embedding_model}'",
                    store_name=self.store_name,
                )
        except (EmbeddingDimensionMismatchError, VectorStoreError):
            raise
        except ImportError:
            raise VectorStoreError(
                "ChromaDB is not installed. Please install via: pip install 'devops-ai[chromadb]'",
                store_name=self.store_name,
            ) from None
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"Failed to initialize ChromaDB vector store: {sanitized}",
                store_name=self.store_name,
            ) from e

    @property
    def persist_directory(self) -> Optional[Path]:
        return self._persist_path

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def _sanitize_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """Ensure all metadata values conform to ChromaDB supported scalar primitives."""
        clean: Dict[str, Union[str, int, float, bool]] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                try:
                    clean[k] = json.dumps(v)
                except Exception:
                    clean[k] = str(v)
        return clean

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Persist or update document chunks with their dense vector embeddings."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})",
                store_name=self.store_name,
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Union[str, int, float, bool]]] = []
        validated_embeddings: List[List[float]] = []

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            actual_dim = len(emb)
            if self._dimension is not None and actual_dim != self._dimension:
                raise EmbeddingDimensionMismatchError(
                    f"Chunk '{chunk.chunk_id}' at index {idx} embedding dimension {actual_dim} != expected {self._dimension}",
                    expected=self._dimension,
                    actual=actual_dim,
                )

            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            validated_embeddings.append(emb)

            meta: Dict[str, Any] = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "content_hash": chunk.content_hash,
                "character_count": chunk.character_count,
            }
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if k not in meta:
                        meta[k] = v

            metadatas.append(self._sanitize_metadata(meta))

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=validated_embeddings,
                metadatas=metadatas,
            )
            logger.debug(
                f"Successfully indexed {len(ids)} chunks in Chroma collection '{self._collection_name}'"
            )
        except EmbeddingDimensionMismatchError:
            raise
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"ChromaDB upsert failure: {sanitized}",
                store_name=self.store_name,
            ) from e

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Execute semantic similarity search against indexed vector chunks.

        Note on Distance vs. Cosine Similarity Conversion:
            ChromaDB configured with 'hnsw:space: cosine' returns cosine distance,
            where distance = 1.0 - cosine_similarity.
            Therefore, cosine_similarity is computed as:
                similarity_score = max(0.0, min(1.0, 1.0 - distance))
            When two vectors are identical, distance is 0.0, yielding similarity_score = 1.0.
            When vectors are orthogonal, distance is 1.0, yielding similarity_score = 0.0.
        """
        if not query_embedding:
            raise VectorStoreError(
                "Query embedding vector cannot be empty", store_name=self.store_name
            )

        actual_dim = len(query_embedding)
        if self._dimension is not None and actual_dim != self._dimension:
            raise EmbeddingDimensionMismatchError(
                f"Query embedding dimension {actual_dim} != expected {self._dimension}",
                expected=self._dimension,
                actual=actual_dim,
            )

        total_items = self.count()
        if total_items == 0:
            return []

        effective_k = min(top_k, total_items)
        if effective_k <= 0:
            return []

        # Build where filter
        where_conditions: List[Dict[str, Any]] = []
        if document_ids:
            if len(document_ids) == 1:
                where_conditions.append({"document_id": document_ids[0]})
            else:
                where_conditions.append({"document_id": {"$in": document_ids}})

        if filter_metadata:
            for k, v in filter_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    where_conditions.append({k: v})

        where_clause: Optional[Dict[str, Any]] = None
        if len(where_conditions) == 1:
            where_clause = where_conditions[0]
        elif len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}

        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": effective_k,
            }
            if where_clause:
                query_kwargs["where"] = where_clause

            raw_results = self._collection.query(**query_kwargs)
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"ChromaDB similarity query failure: {sanitized}",
                store_name=self.store_name,
            ) from e

        results: List[RetrievalResult] = []
        ids_list = raw_results.get("ids", [[]])[0]
        docs_list = raw_results.get("documents", [[]])[0]
        meta_list = raw_results.get("metadatas", [[]])[0]
        dist_list = raw_results.get("distances", [[]])[0]

        for chunk_id, content, meta_dict, distance in zip(
            ids_list, docs_list, meta_list, dist_list
        ):
            meta_dict = meta_dict or {}
            # Exact Cosine Similarity Conversion from Chroma HNSW cosine distance
            sim_score = max(0.0, min(1.0, 1.0 - float(distance)))

            if score_threshold is not None and sim_score < score_threshold:
                continue

            doc_id = str(meta_dict.get("document_id", "unknown"))
            src_file = str(meta_dict.get("source_filename", "unknown"))
            src_path = str(meta_dict.get("source_path", src_file))
            c_index = int(meta_dict.get("chunk_index", 0))
            t_chunks = int(meta_dict.get("total_chunks", 1))

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    source_filename=src_file,
                    source_path=src_path,
                    chunk_index=c_index,
                    total_chunks=t_chunks,
                    content=content,
                    similarity_score=round(sim_score, 6),
                    metadata=meta_dict,
                )
            )

        # Deterministic ordering: similarity descending, chunk_id ascending as tie-breaker
        results.sort(key=lambda r: (-r.similarity_score, r.chunk_id))
        return results

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to the given document identifier."""
        try:
            records = self._collection.get(where={"document_id": document_id})
            ids_to_delete = records.get("ids", [])
            if not ids_to_delete:
                return 0

            self._collection.delete(ids=ids_to_delete)
            logger.debug(f"Deleted {len(ids_to_delete)} chunks for document '{document_id}'")
            return len(ids_to_delete)
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"Failed to delete document '{document_id}' from ChromaDB: {sanitized}",
                store_name=self.store_name,
            ) from e

    def delete_by_source_path(
        self,
        source_path: str,
        exclude_document_id: Optional[str] = None,
    ) -> int:
        """Delete all chunks originating from the given canonical source path.

        Args:
            source_path: Canonical filesystem path or URI of the source document.
            exclude_document_id: Optional document ID to preserve (e.g. when replacing
                an older version with a newly indexed version).

        Returns:
            Number of chunks removed.
        """
        try:
            records = self._collection.get(where={"source_path": source_path})
            ids = records.get("ids", [])
            metadatas = records.get("metadatas", []) or []

            if not ids:
                return 0

            if exclude_document_id:
                ids_to_delete = [
                    chunk_id
                    for chunk_id, meta in zip(ids, metadatas)
                    if (meta or {}).get("document_id") != exclude_document_id
                ]
            else:
                ids_to_delete = list(ids)

            if not ids_to_delete:
                return 0

            self._collection.delete(ids=ids_to_delete)
            logger.debug(
                f"Deleted {len(ids_to_delete)} obsolete chunks for source path '{source_path}'"
            )
            return len(ids_to_delete)
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"Failed to delete chunks for source path '{source_path}' from ChromaDB: {sanitized}",
                store_name=self.store_name,
            ) from e

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by its deterministic identifier."""
        try:
            records = self._collection.get(ids=[chunk_id])
            ids = records.get("ids", [])
            if not ids:
                return None

            doc_text = records.get("documents", [""])[0]
            meta = records.get("metadatas", [{}])[0] or {}

            return DocumentChunk(
                chunk_id=chunk_id,
                document_id=str(meta.get("document_id", "unknown")),
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 1)),
                content=doc_text,
                character_count=len(doc_text),
                content_hash=str(meta.get("content_hash", "")),
                metadata=meta,
            )
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"Failed to get chunk '{chunk_id}' from ChromaDB: {sanitized}",
                store_name=self.store_name,
            ) from e

    def count(self) -> int:
        """Return the count of vector chunks indexed in this collection."""
        try:
            return self._collection.count()
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"Failed to count ChromaDB records: {sanitized}",
                store_name=self.store_name,
            ) from e

    def clear(self) -> None:
        """Purge and reset the vector collection."""
        try:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": self._dimension,
                    "embedding_model": self._embedding_model,
                },
            )
            logger.debug(f"Cleared Chroma collection '{self._collection_name}'")
        except Exception as e:
            sanitized = scrub_secrets(str(e))
            raise VectorStoreError(
                f"Failed to clear ChromaDB collection: {sanitized}",
                store_name=self.store_name,
            ) from e
