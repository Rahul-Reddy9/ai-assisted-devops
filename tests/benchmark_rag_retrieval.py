# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Standalone benchmark measuring vector embedding throughput, indexing speed, and retrieval latency."""

from pathlib import Path
import tempfile
import time

from devops_ai.rag.embeddings import MockEmbeddingProvider
from devops_ai.rag.models import DocumentChunk
from devops_ai.rag.retriever import RAGRetriever
from devops_ai.rag.vector_store import ChromaVectorStore


def run_rag_benchmark(num_chunks: int = 100, dimension: int = 768) -> dict:
    """Run reproducible performance benchmarks across RAG vector subsystem components."""
    results = {}

    # 1. Benchmark Embedding Generation
    provider = MockEmbeddingProvider(dimension=dimension)
    sample_texts = [
        f"Operational incident INC-{i:04d}: Service degradation on tier-{i % 3}. "
        f"Check logs for HTTP 500 error bursts and inspect container cgroups."
        for i in range(num_chunks)
    ]

    t0 = time.perf_counter()
    embeddings = provider.embed_batch(sample_texts)
    embed_elapsed = time.perf_counter() - t0

    results["num_chunks"] = num_chunks
    results["embedding_dimension"] = dimension
    results["embed_batch_time_ms"] = round(embed_elapsed * 1000, 2)
    results["embed_throughput_chunks_per_sec"] = round(num_chunks / embed_elapsed, 1)

    # 2. Benchmark Vector Storage Indexing
    tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        store_path = Path(tmp_dir.name) / "bench_chroma"
        store = ChromaVectorStore(
            persist_directory=store_path,
            collection_name="benchmark_collection",
            dimension=dimension,
        )

        chunks = [
            DocumentChunk(
                chunk_id=f"BENCH-CHK-{i:04d}",
                document_id=f"DOC-BENCH-{i // 5:03d}",
                chunk_index=i % 5,
                total_chunks=5,
                content=sample_texts[i],
                character_count=len(sample_texts[i]),
                content_hash=f"hash-benchmark-content-{i:06d}",
                metadata={"source_filename": f"runbook_{i // 5}.md"},
            )
            for i in range(num_chunks)
        ]

        t1 = time.perf_counter()
        store.add_chunks(chunks=chunks, embeddings=embeddings)
        index_elapsed = time.perf_counter() - t1

        results["index_time_ms"] = round(index_elapsed * 1000, 2)
        results["index_throughput_chunks_per_sec"] = round(num_chunks / index_elapsed, 1)
        results["indexed_count"] = store.count()

        # Measure on-disk storage size
        storage_bytes = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())
        results["storage_size_kb"] = round(storage_bytes / 1024, 2)

        # 3. Benchmark Retrieval Latency
        retriever = RAGRetriever(embedding_provider=provider, vector_store=store)
        query = sample_texts[0]

        query_latencies = []
        for _ in range(10):
            t_q = time.perf_counter()
            retrieved = retriever.retrieve(query, top_k=5)
            query_latencies.append(time.perf_counter() - t_q)

        avg_query_ms = (sum(query_latencies) / len(query_latencies)) * 1000
        results["avg_retrieval_latency_ms"] = round(avg_query_ms, 2)
        results["retrieved_count"] = len(retrieved)
    finally:
        tmp_dir.cleanup()

    return results


if __name__ == "__main__":
    print("Executing RAG Vector Subsystem Benchmark...")
    bench_data = run_rag_benchmark(num_chunks=100, dimension=768)
    print("\n=== Benchmark Results ===")
    for k, v in bench_data.items():
        print(f"  {k}: {v}")
