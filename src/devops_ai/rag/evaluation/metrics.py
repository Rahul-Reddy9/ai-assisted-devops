# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Deterministic metric calculation functions for RAG retrieval quality, grounding, and latency.

IMPORTANT NOTE ON GROUNDEDNESS:
Lexical evidence overlap is a heuristic proxy metric for groundedness and lexical containment.
It measures whether non-stopword tokens from generated claims appear in the retrieved operational
evidence. It is NOT mathematical proof of semantic correctness, causal validity, or reasoning
faithfulness. Full semantic verification requires human-in-the-loop review or formal verification.
"""

import math
import re
from typing import Dict, List, Optional, Set

from devops_ai.rag.evaluation.models import (
    GroundednessMetrics,
    LatencyDistribution,
    RetrievalMetrics,
)

# Standard English operational stopwords for lexical evidence overlap calculation
STOPWORDS: Set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "were",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

CITATION_REGEX = re.compile(
    r"\[(?:DOCUMENT:\s*|DOC:\s*|CHUNK:\s*)?([A-Za-z0-9_\-\.:]+)\]", re.IGNORECASE
)


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Calculate Precision@K: fraction of top-K retrieved items that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document or chunk identifiers.
        relevant_ids: Set of ground-truth relevant identifiers.
        k: Cutoff rank (must be > 0).

    Returns:
        Precision in range [0.0, 1.0]. Returns 0.0 if k <= 0 or retrieved is empty.
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    distinct_top_k: List[str] = []
    seen: Set[str] = set()
    for item in top_k:
        if item not in seen:
            seen.add(item)
            distinct_top_k.append(item)

    relevant_count = sum(1 for item in distinct_top_k if item in relevant_ids)
    return relevant_count / float(k)


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Calculate Recall@K: fraction of all relevant items retrieved in top-K.

    Args:
        retrieved_ids: Ordered list of retrieved document or chunk identifiers.
        relevant_ids: Set of ground-truth relevant identifiers.
        k: Cutoff rank (must be > 0).

    Returns:
        Recall in range [0.0, 1.0].
        If relevant_ids is empty: returns 1.0 if retrieved is empty, else 0.0.
    """
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 0.0

    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    distinct_top_k = set(top_k)
    relevant_retrieved = sum(1 for item in distinct_top_k if item in relevant_ids)
    return relevant_retrieved / float(len(relevant_ids))


def compute_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant item.

    Args:
        retrieved_ids: Ordered list of retrieved document or chunk identifiers.
        relevant_ids: Set of ground-truth relevant identifiers.

    Returns:
        Reciprocal rank in range [0.0, 1.0]. Returns 0.0 if no relevant item is found.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    for rank_idx, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank_idx
    return 0.0


def compute_dcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain at cutoff K (DCG@K).

    Uses the standard exponential formulation: DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1).

    Args:
        retrieved_ids: Ordered list of retrieved document or chunk identifiers.
        relevance_scores: Mapping from identifier to relevance grade (>= 0.0).
        k: Cutoff rank (must be > 0).

    Returns:
        DCG@K value (>= 0.0).
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(item, 0.0)
        if rel > 0.0:
            gain = (2.0**rel) - 1.0
            discount = math.log2(i + 1)
            dcg += gain / discount
    return dcg


def compute_idcg_at_k(
    relevance_scores: Dict[str, float],
    k: int,
) -> float:
    """Calculate Ideal Discounted Cumulative Gain at cutoff K (IDCG@K).

    Args:
        relevance_scores: Mapping of all relevant identifiers to their relevance grade.
        k: Cutoff rank (must be > 0).

    Returns:
        IDCG@K value (>= 0.0).
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    sorted_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_scores[:k], start=1):
        if rel > 0.0:
            gain = (2.0**rel) - 1.0
            discount = math.log2(i + 1)
            idcg += gain / discount
    return idcg


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at cutoff K (nDCG@K).

    Args:
        retrieved_ids: Ordered list of retrieved document or chunk identifiers.
        relevance_scores: Mapping of identifiers to relevance grades.
        k: Cutoff rank.

    Returns:
        nDCG in range [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    idcg = compute_idcg_at_k(relevance_scores, k)
    dcg = compute_dcg_at_k(retrieved_ids, relevance_scores, k)

    if idcg == 0.0:
        return 1.0 if not retrieved_ids else 0.0

    score = dcg / idcg
    return min(1.0, max(0.0, score))


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k_values: List[int],
    graded_relevance: Optional[Dict[str, float]] = None,
) -> RetrievalMetrics:
    """Compute precision@K, recall@K, MRR, and nDCG@K for all configured K cutoffs.

    Args:
        retrieved_ids: Ranked retrieved item identifiers.
        relevant_ids: Ground-truth relevant identifiers.
        k_values: List of rank cutoffs.
        graded_relevance: Optional dictionary mapping IDs to continuous relevance grades.

    Returns:
        Populated RetrievalMetrics instance.
    """
    precision_map: Dict[int, float] = {}
    recall_map: Dict[int, float] = {}
    ndcg_map: Dict[int, float] = {}

    rel_scores: Dict[str, float] = {}
    if graded_relevance:
        rel_scores = {k: max(0.0, float(v)) for k, v in graded_relevance.items()}
    else:
        rel_scores = {item_id: 1.0 for item_id in relevant_ids}

    for k in k_values:
        precision_map[k] = round(compute_precision_at_k(retrieved_ids, relevant_ids, k), 4)
        recall_map[k] = round(compute_recall_at_k(retrieved_ids, relevant_ids, k), 4)
        ndcg_map[k] = round(compute_ndcg_at_k(retrieved_ids, rel_scores, k), 4)

    mrr = round(compute_reciprocal_rank(retrieved_ids, relevant_ids), 4)

    return RetrievalMetrics(
        precision_at_k=precision_map,
        recall_at_k=recall_map,
        mrr=mrr,
        ndcg_at_k=ndcg_map,
    )


def extract_citations(text: str) -> List[str]:
    """Extract citation markers from generated answer text.

    Matches patterns like [DOC-123], [CHUNK-ABC], [DOCUMENT: DOC-1], etc.
    """
    if not text:
        return []
    matches = CITATION_REGEX.findall(text)
    return [m.strip() for m in matches if m.strip()]


def tokenize_for_overlap(text: str) -> Set[str]:
    """Extract normalized non-stopword tokens from text."""
    if not text:
        return set()
    words = re.findall(r"[a-z0-9_\-]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def split_into_claims(text: str) -> List[str]:
    """Split text into sentence-level claims for evidence support verification."""
    if not text:
        return []
    # Split by common sentence terminators and newlines
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = [s.strip() for s in raw_sentences if s.strip()]
    return cleaned if cleaned else [text.strip()]


def evaluate_groundedness(
    answer: str,
    retrieved_contexts: List[str],
    valid_citation_ids: Set[str],
    overlap_threshold: float = 0.3,
) -> GroundednessMetrics:
    """Evaluate citation faithfulness and lexical evidence overlap of an answer.

    Note: Lexical evidence overlap is a heuristic proxy metric for groundedness,
    not formal proof of semantic validity.

    Args:
        answer: Generated answer or diagnostic action plan text.
        retrieved_contexts: Texts of the chunks retrieved for this query.
        valid_citation_ids: Set of valid chunk/document IDs present in retrieved evidence.
        overlap_threshold: Minimum token containment fraction for a claim to be considered supported.

    Returns:
        Populated GroundednessMetrics instance.
    """
    if not answer or not answer.strip():
        return GroundednessMetrics(
            citation_coverage=1.0,
            evidence_overlap_score=0.0,
            total_claims=0,
            supported_claims=0,
            unsupported_claims=0,
            hallucinated_citations=[],
        )

    # 1. Citation verification
    citations = extract_citations(answer)
    hallucinated: List[str] = []
    if citations:
        valid_count = 0
        for cite in citations:
            if cite in valid_citation_ids:
                valid_count += 1
            else:
                hallucinated.append(cite)
        citation_coverage = valid_count / float(len(citations))
    else:
        # If no citations were made, coverage is vacuously 1.0 (no false citations)
        citation_coverage = 1.0

    # 2. Lexical evidence overlap & sentence-level claim evaluation
    claims = split_into_claims(answer)
    combined_context = " ".join(retrieved_contexts)
    context_tokens = tokenize_for_overlap(combined_context)

    claim_overlaps: List[float] = []
    supported_count = 0

    for claim in claims:
        claim_tokens = tokenize_for_overlap(claim)
        if not claim_tokens:
            # Vacuous claim with no meaningful keywords (e.g. "Step 1:")
            claim_overlaps.append(1.0)
            supported_count += 1
            continue

        overlap_tokens = claim_tokens.intersection(context_tokens)
        overlap_ratio = len(overlap_tokens) / float(len(claim_tokens))
        claim_overlaps.append(overlap_ratio)

        if overlap_ratio >= overlap_threshold:
            supported_count += 1

    total_claims = len(claims)
    unsupported_count = total_claims - supported_count
    mean_overlap = sum(claim_overlaps) / float(total_claims) if total_claims > 0 else 0.0

    return GroundednessMetrics(
        citation_coverage=round(citation_coverage, 4),
        evidence_overlap_score=round(mean_overlap, 4),
        total_claims=total_claims,
        supported_claims=supported_count,
        unsupported_claims=unsupported_count,
        hallucinated_citations=hallucinated,
    )


def calculate_latency_distribution(latencies_ms: List[float]) -> LatencyDistribution:
    """Compute summary statistics for a collection of latency measurements.

    Args:
        latencies_ms: List of latency measurements in milliseconds.

    Returns:
        LatencyDistribution with mean, min, max, p50, p90, p95, p99.
    """
    if not latencies_ms:
        return LatencyDistribution()

    sorted_vals = sorted(latencies_ms)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return d0 + d1

    mean_val = sum(sorted_vals) / float(n)

    return LatencyDistribution(
        mean=round(mean_val, 3),
        p50=round(percentile(50.0), 3),
        p90=round(percentile(90.0), 3),
        p95=round(percentile(95.0), 3),
        p99=round(percentile(99.0), 3),
        min=round(sorted_vals[0], 3),
        max=round(sorted_vals[-1], 3),
    )
