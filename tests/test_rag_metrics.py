# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests for deterministic RAG evaluation metrics, grounding, and latency percentiles."""

import math
import pytest

from devops_ai.rag.evaluation.metrics import (
    calculate_latency_distribution,
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    evaluate_groundedness,
    extract_citations,
)


class TestPrecisionAtK:
    def test_precision_perfect(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        assert compute_precision_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_precision_partial(self):
        retrieved = ["doc1", "other", "doc2", "none"]
        relevant = {"doc1", "doc2"}
        # Top-3 has doc1 and doc2 -> 2/3
        assert compute_precision_at_k(retrieved, relevant, k=3) == pytest.approx(2.0 / 3.0)

    def test_precision_zero(self):
        retrieved = ["other1", "other2"]
        relevant = {"doc1", "doc2"}
        assert compute_precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)

    def test_precision_edge_cases(self):
        assert compute_precision_at_k([], {"doc1"}, k=5) == 0.0
        assert compute_precision_at_k(["doc1"], set(), k=5) == 0.0
        assert compute_precision_at_k(["doc1"], {"doc1"}, k=0) == 0.0
        assert compute_precision_at_k(["doc1"], {"doc1"}, k=-1) == 0.0
        # When fewer items retrieved than K, denominator is still K
        assert compute_precision_at_k(["doc1"], {"doc1"}, k=5) == pytest.approx(0.2)


class TestRecallAtK:
    def test_recall_perfect(self):
        retrieved = ["doc1", "doc2", "other"]
        relevant = {"doc1", "doc2"}
        assert compute_recall_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_recall_partial(self):
        retrieved = ["doc1", "other1", "other2"]
        relevant = {"doc1", "doc2"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)

    def test_recall_zero(self):
        retrieved = ["other1", "other2"]
        relevant = {"doc1"}
        assert compute_recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)

    def test_recall_edge_cases(self):
        assert compute_recall_at_k([], {"doc1"}, k=5) == 0.0
        assert compute_recall_at_k(["doc1"], {"doc1"}, k=0) == 0.0
        assert compute_recall_at_k(["doc1"], {"doc1"}, k=-2) == 0.0
        # Empty relevant items: if retrieved is empty -> 1.0, else 0.0
        assert compute_recall_at_k([], set(), k=5) == 1.0
        assert compute_recall_at_k(["doc1"], set(), k=5) == 0.0


class TestReciprocalRank:
    def test_mrr_first_rank(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert compute_reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)

    def test_mrr_third_rank(self):
        retrieved = ["other1", "other2", "doc1", "doc2"]
        relevant = {"doc1"}
        assert compute_reciprocal_rank(retrieved, relevant) == pytest.approx(1.0 / 3.0)

    def test_mrr_none_found(self):
        retrieved = ["other1", "other2"]
        relevant = {"doc1"}
        assert compute_reciprocal_rank(retrieved, relevant) == pytest.approx(0.0)

    def test_mrr_empty_input(self):
        assert compute_reciprocal_rank([], {"doc1"}) == 0.0
        assert compute_reciprocal_rank(["doc1"], set()) == 0.0


class TestDCGAndNDCG:
    def test_dcg_calculation(self):
        retrieved = ["doc1", "doc2"]
        rel = {"doc1": 3.0, "doc2": 1.0}
        # rank 1: (2^3 - 1) / log2(2) = 7.0 / 1 = 7.0
        # rank 2: (2^1 - 1) / log2(3) = 1.0 / 1.5849625 = 0.63092975
        expected = 7.0 + (1.0 / math.log2(3))
        assert compute_dcg_at_k(retrieved, rel, k=2) == pytest.approx(expected, rel=1e-4)

    def test_ndcg_perfect_ranking(self):
        retrieved = ["doc1", "doc2", "doc3"]
        rel = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        assert compute_ndcg_at_k(retrieved, rel, k=3) == pytest.approx(1.0)

    def test_ndcg_suboptimal_ranking(self):
        retrieved = ["doc3", "doc2", "doc1"]  # reversed
        rel = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        score = compute_ndcg_at_k(retrieved, rel, k=3)
        assert 0.0 < score < 1.0

    def test_ndcg_zero_relevant_edge_cases(self):
        assert compute_ndcg_at_k([], {}, k=3) == 1.0
        assert compute_ndcg_at_k(["doc1"], {}, k=3) == 0.0
        assert compute_ndcg_at_k(["doc1"], {"doc1": 1.0}, k=0) == 0.0


class TestCitationAndGroundedness:
    def test_extract_citations(self):
        text = "See runbook [DOC-K8S-01] and also chunk [CHUNK: C-42] or [DOCUMENT: DOC-PG-02]."
        citations = extract_citations(text)
        assert "DOC-K8S-01" in citations
        assert "C-42" in citations
        assert "DOC-PG-02" in citations

    def test_groundedness_valid_citations(self):
        answer = "Check pod memory limits [DOC-K8S-01]. The OOM killer exited code 137."
        contexts = ["Kubernetes pod memory limits exceeded. Container killed with exit code 137."]
        valid_citations = {"DOC-K8S-01"}
        metrics = evaluate_groundedness(answer, contexts, valid_citations, overlap_threshold=0.3)
        assert metrics.citation_coverage == 1.0
        assert len(metrics.hallucinated_citations) == 0
        assert metrics.supported_claims >= 1

    def test_groundedness_hallucinated_citation(self):
        answer = "Check pod memory limits [DOC-FABRICATED-99]."
        contexts = ["Some context about k8s."]
        valid_citations = {"DOC-K8S-01"}
        metrics = evaluate_groundedness(answer, contexts, valid_citations)
        assert metrics.citation_coverage == 0.0
        assert "DOC-FABRICATED-99" in metrics.hallucinated_citations

    def test_groundedness_empty_answer(self):
        metrics = evaluate_groundedness("", ["context"], {"DOC-1"})
        assert metrics.total_claims == 0
        assert metrics.evidence_overlap_score == 0.0


class TestLatencyDistribution:
    def test_latency_distribution_calculation(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        dist = calculate_latency_distribution(latencies)
        assert dist.mean == pytest.approx(30.0)
        assert dist.min == 10.0
        assert dist.max == 50.0
        assert dist.p50 == pytest.approx(30.0)
        assert dist.p90 > dist.p50

    def test_latency_distribution_empty(self):
        dist = calculate_latency_distribution([])
        assert dist.mean == 0.0
        assert dist.p50 == 0.0
        assert dist.min == 0.0
        assert dist.max == 0.0
