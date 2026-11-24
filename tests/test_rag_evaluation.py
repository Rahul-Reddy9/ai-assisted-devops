# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit and integration tests for RAGEvaluator, evaluation datasets, and quality gates."""

from pathlib import Path
import pytest

from devops_ai.core.config import Settings
from devops_ai.core.exceptions import DatasetValidationError, EvaluationThresholdError
from devops_ai.rag.embeddings.mock import MockEmbeddingProvider
from devops_ai.rag.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
    EvaluationReport,
    RAGEvaluator,
    get_builtin_evaluation_dataset,
    load_evaluation_dataset,
    save_evaluation_dataset,
)
from devops_ai.rag.models import DocumentChunk
from devops_ai.rag.retriever import RAGRetriever
from devops_ai.rag.vector_store.chroma import ChromaVectorStore


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Fixture providing isolated settings and persistent directory for Chroma."""
    return Settings(
        rag_embedding_provider="mock",
        rag_vector_store="chroma",
        rag_vector_store_path=str(tmp_path / "chroma_eval_db"),
        rag_eval_default_k=[1, 3, 5],
        rag_similarity_threshold=0.0,
    )


@pytest.fixture
def populated_retriever(test_settings: Settings) -> RAGRetriever:
    """Fixture providing a populated RAGRetriever using MockEmbeddingProvider and Chroma."""
    provider = MockEmbeddingProvider(dimension=768)
    store = ChromaVectorStore(
        persist_directory=test_settings.rag_vector_store_path,
        collection_name="eval_test_collection",
        dimension=768,
    )
    retriever = RAGRetriever(
        embedding_provider=provider,
        vector_store=store,
        settings=test_settings,
    )

    # Ingest synthetic chunks corresponding to built-in dataset queries
    chunks = [
        DocumentChunk(
            chunk_id="DOC-K8S-RUNBOOK-01-C0",
            document_id="DOC-K8S-RUNBOOK-01",
            chunk_index=0,
            total_chunks=1,
            content="Kubernetes pod CrashLoopBackOff OOMKilled exit code 137 container memory limit remediation runbook.",
            character_count=100,
            content_hash="hash_k8s_01_c0_0123456789",
            metadata={"source_filename": "k8s_runbook.md", "source_path": "/docs/k8s_runbook.md"},
        ),
        DocumentChunk(
            chunk_id="DOC-PG-OPS-03-C0",
            document_id="DOC-PG-OPS-03",
            chunk_index=0,
            total_chunks=1,
            content="PostgreSQL FATAL remaining connection slots are reserved for non-replication superuser connections pool tuning.",
            character_count=114,
            content_hash="hash_pg_03_c0_0123456789",
            metadata={"source_filename": "pg_ops.md", "source_path": "/docs/pg_ops.md"},
        ),
        DocumentChunk(
            chunk_id="DOC-REDIS-OPS-04-C0",
            document_id="DOC-REDIS-OPS-04",
            chunk_index=0,
            total_chunks=1,
            content="Redis high latency spike instantaneous ops per second drop memory fragmentation ratio activedefrag.",
            character_count=108,
            content_hash="hash_redis_04_c0_0123456789",
            metadata={"source_filename": "redis_ops.md", "source_path": "/docs/redis_ops.md"},
        ),
    ]

    embeddings = provider.embed_batch([c.content for c in chunks])
    store.add_chunks(chunks, embeddings)
    return retriever


class TestEvaluationDataset:
    def test_builtin_dataset_integrity(self):
        dataset = get_builtin_evaluation_dataset()
        assert dataset.dataset_id == "builtin-devops-v1"
        assert len(dataset.queries) == 10
        for q in dataset.queries:
            assert len(q.query_id) > 0
            assert len(q.query) > 0
            assert len(q.expected_document_ids) > 0

    def test_load_save_json_dataset(self, tmp_path: Path):
        dataset = get_builtin_evaluation_dataset()
        save_path = tmp_path / "custom_eval.json"
        save_evaluation_dataset(dataset, save_path)
        assert save_path.exists()

        loaded = load_evaluation_dataset(save_path, allowed_root=tmp_path)
        assert loaded.dataset_id == dataset.dataset_id
        assert len(loaded.queries) == len(dataset.queries)

    def test_load_yaml_dataset(self, tmp_path: Path):
        yaml_content = """
dataset_id: test-yaml-eval
name: YAML Benchmark
version: 1.0.0
queries:
  - query_id: Q-TEST-01
    query: "Test query for YAML loader"
    expected_document_ids: ["DOC-TEST-1"]
"""
        yaml_path = tmp_path / "test_eval.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        loaded = load_evaluation_dataset(yaml_path, allowed_root=tmp_path)
        assert loaded.dataset_id == "test-yaml-eval"
        assert len(loaded.queries) == 1
        assert loaded.queries[0].expected_document_ids == ["DOC-TEST-1"]

    def test_load_dataset_path_traversal_rejection(self, tmp_path: Path):
        outside_path = tmp_path / "outside.json"
        outside_path.write_text(
            '{"dataset_id": "out", "name": "out", "queries": []}', encoding="utf-8"
        )
        allowed_subdir = tmp_path / "allowed_subdir"
        allowed_subdir.mkdir()

        with pytest.raises(DatasetValidationError, match="breaches permitted root"):
            load_evaluation_dataset(outside_path, allowed_root=allowed_subdir)

    def test_load_dataset_missing_file(self, tmp_path: Path):
        with pytest.raises(DatasetValidationError, match="does not exist"):
            load_evaluation_dataset(tmp_path / "nonexistent.json")

    def test_load_dataset_empty_file(self, tmp_path: Path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("   \n", encoding="utf-8")
        with pytest.raises(DatasetValidationError, match="empty"):
            load_evaluation_dataset(empty_file)

    def test_load_dataset_invalid_schema(self, tmp_path: Path):
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text('{"dataset_id": "bad", "name": "bad"}', encoding="utf-8")
        # Missing required queries
        with pytest.raises(DatasetValidationError, match="at least one query"):
            load_evaluation_dataset(invalid_file)


class TestRAGEvaluator:
    def test_evaluate_query(self, populated_retriever: RAGRetriever):
        evaluator = RAGEvaluator(retriever=populated_retriever)
        query = EvaluationQuery(
            query_id="Q-K8S-01",
            query="Kubernetes pod CrashLoopBackOff OOMKilled exit code 137 container memory limit",
            expected_document_ids=["DOC-K8S-RUNBOOK-01"],
            expected_chunk_ids=["DOC-K8S-RUNBOOK-01-C0"],
            reference_answer="Check container memory limits [DOC-K8S-RUNBOOK-01]. Exit code 137.",
        )

        result = evaluator.evaluate_query(query, k_values=[1, 3])
        assert result.query_id == "Q-K8S-01"
        assert len(result.retrieved_chunks) > 0
        assert result.retrieval_metrics.precision_at_k[1] == 1.0
        assert result.retrieval_metrics.recall_at_k[1] == 1.0
        assert result.retrieval_metrics.mrr == 1.0
        assert result.latency_metrics.embedding_latency_ms >= 0.0
        assert result.latency_metrics.search_latency_ms >= 0.0
        assert result.latency_metrics.total_latency_ms >= 0.0
        assert result.groundedness_metrics is not None
        assert result.groundedness_metrics.citation_coverage == 1.0

    def test_evaluate_dataset_end_to_end(self, populated_retriever: RAGRetriever):
        evaluator = RAGEvaluator(retriever=populated_retriever)
        dataset = EvaluationDataset(
            dataset_id="mini-eval",
            name="Mini Benchmark",
            queries=[
                EvaluationQuery(
                    query_id="Q1",
                    query="Kubernetes pod CrashLoopBackOff OOMKilled exit code 137",
                    expected_document_ids=["DOC-K8S-RUNBOOK-01"],
                    reference_answer="Increase memory limit [DOC-K8S-RUNBOOK-01].",
                ),
                EvaluationQuery(
                    query_id="Q2",
                    query="PostgreSQL FATAL remaining connection slots are reserved",
                    expected_document_ids=["DOC-PG-OPS-03"],
                    reference_answer="Deploy PgBouncer pooler [DOC-PG-OPS-03].",
                ),
            ],
        )

        report = evaluator.evaluate_dataset(
            dataset=dataset,
            k_values=[1, 3],
            quality_thresholds={"min_precision": 0.3, "min_recall": 0.5, "min_mrr": 0.5},
        )

        assert isinstance(report, EvaluationReport)
        assert report.total_queries == 2
        assert report.passed_quality_gates is True
        assert report.mean_retrieval_metrics.mrr >= 0.5
        assert report.latency_summary.total_latency.mean >= 0.0

    def test_quality_gate_threshold_failure(self, populated_retriever: RAGRetriever):
        evaluator = RAGEvaluator(retriever=populated_retriever)
        dataset = EvaluationDataset(
            dataset_id="mini-eval",
            name="Mini Benchmark",
            queries=[
                EvaluationQuery(
                    query_id="Q-UNMATCHED",
                    query="Unrelated query about astronomy telescopes and galaxies",
                    expected_document_ids=["DOC-ASTRONOMY-99"],
                ),
            ],
        )

        # Set impossible threshold of precision 1.0 for an unmatched query
        report = evaluator.evaluate_dataset(
            dataset=dataset,
            k_values=[1],
            quality_thresholds={"min_precision": 1.0},
            raise_on_threshold_failure=False,
        )
        assert report.passed_quality_gates is False
        assert any("precision" in k and not v for k, v in report.threshold_checks.items())

        with pytest.raises(EvaluationThresholdError):
            evaluator.evaluate_dataset(
                dataset=dataset,
                k_values=[1],
                quality_thresholds={"min_precision": 1.0},
                raise_on_threshold_failure=True,
            )

    def test_evaluate_empty_dataset(self, populated_retriever: RAGRetriever):
        evaluator = RAGEvaluator(retriever=populated_retriever)
        empty_dataset = EvaluationDataset(
            dataset_id="empty-eval",
            name="Empty Benchmark",
            queries=[],
        )
        report = evaluator.evaluate_dataset(dataset=empty_dataset)
        assert report.total_queries == 0
        assert report.passed_quality_gates is True
