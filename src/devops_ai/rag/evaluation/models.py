# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Data models for RAG evaluation, retrieval quality assessment, and observability."""

import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationQuery(BaseModel):
    """A single evaluation inquiry with ground-truth relevant documents and facts."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1, description="Deterministic unique identifier for the query")
    query: str = Field(min_length=1, description="Natural language operational query text")
    expected_document_ids: List[str] = Field(
        min_length=1, description="List of expected relevant source document IDs"
    )
    expected_chunk_ids: Optional[List[str]] = Field(
        default=None, description="Optional expected chunk IDs"
    )
    graded_relevance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional mapping of document_id or chunk_id to graded relevance score (>= 0.0)",
    )
    reference_answer: Optional[str] = Field(
        default=None, description="Optional ground-truth reference answer"
    )
    expected_facts: Optional[List[str]] = Field(
        default=None, description="Key factual assertions expected in grounded answers"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary query metadata (e.g. incident category, severity, domain)",
    )


class EvaluationDataset(BaseModel):
    """Collection of evaluation queries forming an operational benchmark suite."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(
        min_length=1, description="Unique identifier for the evaluation dataset"
    )
    name: str = Field(min_length=1, description="Human-readable name of the dataset")
    description: Optional[str] = Field(default=None, description="Purpose and scope description")
    version: str = Field(default="1.0.0", description="Semantic version of the dataset")
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC creation timestamp",
    )
    queries: List[EvaluationQuery] = Field(
        default_factory=list, description="List of evaluation queries"
    )


class RetrievalMetrics(BaseModel):
    """Information retrieval quality metrics at specified rank cutoffs."""

    model_config = ConfigDict(extra="forbid")

    precision_at_k: Dict[int, float] = Field(
        default_factory=dict,
        description="Precision at cutoff K (fraction of top-K that are relevant)",
    )
    recall_at_k: Dict[int, float] = Field(
        default_factory=dict,
        description="Recall at cutoff K (fraction of all relevant docs retrieved in top-K)",
    )
    mrr: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Reciprocal rank of the first relevant document"
    )
    ndcg_at_k: Dict[int, float] = Field(
        default_factory=dict, description="Normalized Discounted Cumulative Gain at cutoff K"
    )


class GroundednessMetrics(BaseModel):
    """Measures faithfulness, citation validity, and hallucination in generated answers."""

    model_config = ConfigDict(extra="forbid")

    citation_coverage: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Fraction of cited document/chunk references that exist in retrieved evidence",
    )
    evidence_overlap_score: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Lexical evidence overlap score (proxy metric for semantic grounding)",
    )
    total_claims: int = Field(
        ge=0, default=0, description="Total sentence-level factual claims evaluated"
    )
    supported_claims: int = Field(
        ge=0, default=0, description="Claims with evidence overlap >= threshold"
    )
    unsupported_claims: int = Field(
        ge=0, default=0, description="Claims lacking sufficient evidence in retrieved context"
    )
    hallucinated_citations: List[str] = Field(
        default_factory=list,
        description="Citations referenced in answer that were not in retrieved results",
    )


class LatencyMetrics(BaseModel):
    """Monotonic latency measurements in milliseconds for a single query."""

    model_config = ConfigDict(extra="forbid")

    embedding_latency_ms: float = Field(
        ge=0.0, default=0.0, description="Query embedding duration in milliseconds"
    )
    search_latency_ms: float = Field(
        ge=0.0, default=0.0, description="Vector store search duration in milliseconds"
    )
    total_latency_ms: float = Field(
        ge=0.0, default=0.0, description="End-to-end retrieval duration in milliseconds"
    )


class LatencyDistribution(BaseModel):
    """Summary statistics for latency across multiple queries."""

    model_config = ConfigDict(extra="forbid")

    mean: float = Field(ge=0.0, default=0.0)
    p50: float = Field(ge=0.0, default=0.0)
    p90: float = Field(ge=0.0, default=0.0)
    p95: float = Field(ge=0.0, default=0.0)
    p99: float = Field(ge=0.0, default=0.0)
    min: float = Field(ge=0.0, default=0.0)
    max: float = Field(ge=0.0, default=0.0)


class AggregateLatencyMetrics(BaseModel):
    """Decomposed latency distributions for benchmark observability."""

    model_config = ConfigDict(extra="forbid")

    embedding_latency: LatencyDistribution = Field(default_factory=LatencyDistribution)
    search_latency: LatencyDistribution = Field(default_factory=LatencyDistribution)
    total_latency: LatencyDistribution = Field(default_factory=LatencyDistribution)


class QueryEvaluationResult(BaseModel):
    """Detailed evaluation result for a single query."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(description="Identifier of the evaluated query")
    query: str = Field(description="Operational query text")
    retrieved_chunks: List[str] = Field(
        default_factory=list, description="Ordered chunk IDs returned by retriever"
    )
    retrieved_documents: List[str] = Field(
        default_factory=list, description="Ordered document IDs returned by retriever"
    )
    retrieval_metrics: RetrievalMetrics = Field(description="IR metrics for this query")
    groundedness_metrics: Optional[GroundednessMetrics] = Field(
        default=None, description="Groundedness and citation metrics if answer evaluated"
    )
    latency_metrics: LatencyMetrics = Field(description="Observed latencies for this query")


class EvaluationReport(BaseModel):
    """Aggregated evaluation report across an entire benchmark dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(description="Identifier of the evaluated dataset")
    dataset_name: str = Field(description="Name of the dataset")
    total_queries: int = Field(ge=0, description="Total number of queries evaluated")
    evaluation_timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC timestamp of evaluation run",
    )
    k_values: List[int] = Field(description="Top-K cutoff points evaluated")
    mean_retrieval_metrics: RetrievalMetrics = Field(
        description="Averaged retrieval quality metrics across all queries"
    )
    mean_groundedness_metrics: Optional[GroundednessMetrics] = Field(
        default=None, description="Averaged groundedness metrics if answers evaluated"
    )
    latency_summary: AggregateLatencyMetrics = Field(
        description="Latency distributions across all queries"
    )
    query_results: List[QueryEvaluationResult] = Field(
        default_factory=list, description="Per-query evaluation breakdowns"
    )
    threshold_checks: Dict[str, bool] = Field(
        default_factory=dict, description="Status of quality-gate threshold checks"
    )
    passed_quality_gates: bool = Field(
        default=True, description="Whether all configured quality gates were satisfied"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Diagnostic and environment metadata"
    )
