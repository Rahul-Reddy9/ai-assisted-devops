# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""RAG evaluation, retrieval quality metrics, and observability subsystem."""

from devops_ai.rag.evaluation.dataset import (
    get_builtin_evaluation_dataset,
    load_evaluation_dataset,
    save_evaluation_dataset,
)
from devops_ai.rag.evaluation.evaluator import RAGEvaluator
from devops_ai.rag.evaluation.metrics import (
    compute_dcg_at_k,
    compute_idcg_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_retrieval_metrics,
    evaluate_groundedness,
    extract_citations,
)
from devops_ai.rag.evaluation.models import (
    AggregateLatencyMetrics,
    EvaluationDataset,
    EvaluationQuery,
    EvaluationReport,
    GroundednessMetrics,
    LatencyDistribution,
    LatencyMetrics,
    QueryEvaluationResult,
    RetrievalMetrics,
)

__all__ = [
    "AggregateLatencyMetrics",
    "EvaluationDataset",
    "EvaluationQuery",
    "EvaluationReport",
    "GroundednessMetrics",
    "LatencyDistribution",
    "LatencyMetrics",
    "QueryEvaluationResult",
    "RAGEvaluator",
    "RetrievalMetrics",
    "compute_dcg_at_k",
    "compute_idcg_at_k",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_reciprocal_rank",
    "compute_retrieval_metrics",
    "evaluate_groundedness",
    "extract_citations",
    "get_builtin_evaluation_dataset",
    "load_evaluation_dataset",
    "save_evaluation_dataset",
]
