# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""RAG retrieval evaluation orchestrator measuring IR metrics, grounding, and latency."""

import time
from typing import Any, Dict, List, Optional, Set

from devops_ai.core.config import Settings, get_settings
from devops_ai.core.exceptions import EvaluationThresholdError
from devops_ai.core.logging import get_logger, scrub_secrets
from devops_ai.rag.evaluation.metrics import (
    calculate_latency_distribution,
    compute_retrieval_metrics,
    evaluate_groundedness,
)
from devops_ai.rag.evaluation.models import (
    AggregateLatencyMetrics,
    EvaluationDataset,
    EvaluationQuery,
    EvaluationReport,
    GroundednessMetrics,
    LatencyMetrics,
    QueryEvaluationResult,
    RetrievalMetrics,
)
from devops_ai.rag.retriever import RAGRetriever

logger = get_logger("rag.evaluation.evaluator")


class RAGEvaluator:
    """Evaluates RAG retrieval quality, answer grounding, and query latency."""

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or RAGRetriever(settings=self.settings)

    def evaluate_query(
        self,
        query: EvaluationQuery,
        k_values: List[int],
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryEvaluationResult:
        """Execute and measure a single evaluation query against the retriever.

        Decomposes query execution to separately measure embedding latency,
        vector storage search latency, and end-to-end retrieval latency.
        """
        clean_query = query.query.strip()
        max_k = max(k_values) if k_values else 5

        # 1. Monotonic latency measurement: embedding generation
        t_start = time.perf_counter()
        t_emb_0 = time.perf_counter()
        query_embedding = self.retriever.embedding_provider.embed_text(clean_query)
        t_emb_1 = time.perf_counter()
        embedding_latency_ms = (t_emb_1 - t_emb_0) * 1000.0

        # 2. Monotonic latency measurement: vector store search
        t_search_0 = time.perf_counter()
        effective_threshold = (
            score_threshold if score_threshold is not None else self.retriever.default_threshold
        )
        raw_results = self.retriever.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=max_k,
            score_threshold=effective_threshold,
            filter_metadata=filter_metadata,
        )
        t_search_1 = time.perf_counter()
        search_latency_ms = (t_search_1 - t_search_0) * 1000.0

        # 3. Content deduplication and deterministic sorting (mirroring retriever logic)
        filtered_results = []
        seen_hashes = set()
        seen_ids = set()
        for item in raw_results:
            if item.chunk_id in seen_ids:
                continue
            seen_ids.add(item.chunk_id)
            c_hash = item.metadata.get("content_hash", "")
            if c_hash:
                if c_hash in seen_hashes:
                    continue
                seen_hashes.add(c_hash)
            filtered_results.append(item)

        # Sort: similarity descending, chunk_id ascending as tie-breaker
        filtered_results.sort(key=lambda r: (-r.similarity_score, r.chunk_id))
        final_results = filtered_results[:max_k]

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        if total_latency_ms > self.settings.rag_eval_latency_warn_threshold_ms:
            logger.warning(
                f"Query '{query.query_id}' latency ({total_latency_ms:.2f}ms) exceeded warning threshold "
                f"({self.settings.rag_eval_latency_warn_threshold_ms:.2f}ms)"
            )

        latency_metrics = LatencyMetrics(
            embedding_latency_ms=round(embedding_latency_ms, 3),
            search_latency_ms=round(search_latency_ms, 3),
            total_latency_ms=round(total_latency_ms, 3),
        )

        retrieved_chunk_ids = [item.chunk_id for item in final_results]
        retrieved_doc_ids = [item.document_id for item in final_results]

        # 4. Map relevance: determine which retrieved items are relevant
        expected_docs = set(query.expected_document_ids)
        expected_chunks = set(query.expected_chunk_ids or [])

        # An item is relevant if its chunk_id is in expected_chunks OR its document_id is in expected_docs
        relevant_identifiers: Set[str] = set()
        for item in final_results:
            if item.document_id in expected_docs or item.chunk_id in expected_chunks:
                relevant_identifiers.add(item.chunk_id)

        # For relevance scoring dictionary (graded relevance)
        graded_rel_map: Dict[str, float] = {}
        if query.graded_relevance:
            for item in final_results:
                if item.chunk_id in query.graded_relevance:
                    graded_rel_map[item.chunk_id] = query.graded_relevance[item.chunk_id]
                elif item.document_id in query.graded_relevance:
                    graded_rel_map[item.chunk_id] = query.graded_relevance[item.document_id]
                elif item.chunk_id in relevant_identifiers:
                    graded_rel_map[item.chunk_id] = 1.0
        else:
            for item_id in relevant_identifiers:
                graded_rel_map[item_id] = 1.0

        # Also populate relevant set with all expected items so recall is evaluated against full ground truth
        all_relevant_ground_truth: Set[str] = set()
        if expected_chunks:
            all_relevant_ground_truth.update(expected_chunks)
        else:
            all_relevant_ground_truth.update(expected_docs)

        # Calculate retrieval metrics
        # For precision and recall, if expected_chunks was not provided, map retrieved item to doc_id for evaluation
        eval_retrieved_ids = retrieved_chunk_ids if expected_chunks else retrieved_doc_ids
        eval_relevant_ids = expected_chunks if expected_chunks else expected_docs

        eval_graded_map: Dict[str, float] = {}
        if query.graded_relevance:
            for item_id in eval_retrieved_ids:
                if item_id in query.graded_relevance:
                    eval_graded_map[item_id] = query.graded_relevance[item_id]
                elif item_id in eval_relevant_ids:
                    eval_graded_map[item_id] = 1.0
        else:
            eval_graded_map = {item_id: 1.0 for item_id in eval_relevant_ids}

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_ids=eval_retrieved_ids,
            relevant_ids=eval_relevant_ids,
            k_values=k_values,
            graded_relevance=eval_graded_map,
        )

        # 5. Answer groundedness evaluation (if reference_answer is provided)
        groundedness_metrics: Optional[GroundednessMetrics] = None
        if query.reference_answer:
            retrieved_contexts = [item.content for item in final_results]
            valid_citations = set(retrieved_chunk_ids).union(set(retrieved_doc_ids))
            groundedness_metrics = evaluate_groundedness(
                answer=query.reference_answer,
                retrieved_contexts=retrieved_contexts,
                valid_citation_ids=valid_citations,
                overlap_threshold=self.settings.rag_eval_groundedness_overlap_threshold,
            )

        return QueryEvaluationResult(
            query_id=query.query_id,
            query=clean_query,
            retrieved_chunks=retrieved_chunk_ids,
            retrieved_documents=retrieved_doc_ids,
            retrieval_metrics=retrieval_metrics,
            groundedness_metrics=groundedness_metrics,
            latency_metrics=latency_metrics,
        )

    def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        k_values: Optional[List[int]] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
        raise_on_threshold_failure: bool = False,
    ) -> EvaluationReport:
        """Run full evaluation over an EvaluationDataset and aggregate quality metrics.

        Args:
            dataset: The evaluation benchmark dataset.
            k_values: Rank cutoffs to evaluate (defaults to settings.rag_eval_default_k).
            quality_thresholds: Optional quality gate thresholds, e.g.:
                {"min_precision": 0.7, "min_recall": 0.8, "min_mrr": 0.7, "max_latency_ms": 200.0}
            raise_on_threshold_failure: If True, raises EvaluationThresholdError on threshold breach.

        Returns:
            Aggregated EvaluationReport instance.
        """
        raw_k = k_values or self.settings.rag_eval_default_k
        clean_k = sorted(list({int(k) for k in raw_k if int(k) > 0}))
        if not clean_k:
            clean_k = [1, 3, 5]

        thresholds = quality_thresholds or {}

        # Handle empty dataset edge case
        if not dataset.queries:
            logger.warning(f"Evaluation dataset '{dataset.dataset_id}' contains 0 queries")
            empty_report = EvaluationReport(
                dataset_id=dataset.dataset_id,
                dataset_name=dataset.name,
                total_queries=0,
                k_values=clean_k,
                mean_retrieval_metrics=RetrievalMetrics(
                    precision_at_k={k: 0.0 for k in clean_k},
                    recall_at_k={k: 0.0 for k in clean_k},
                    mrr=0.0,
                    ndcg_at_k={k: 0.0 for k in clean_k},
                ),
                mean_groundedness_metrics=None,
                latency_summary=AggregateLatencyMetrics(),
                query_results=[],
                threshold_checks={},
                passed_quality_gates=True,
                metadata={"warning": "Empty evaluation dataset"},
            )
            return empty_report

        query_results: List[QueryEvaluationResult] = []
        embedding_latencies: List[float] = []
        search_latencies: List[float] = []
        total_latencies: List[float] = []

        # Run per-query evaluation
        for q in dataset.queries:
            try:
                res = self.evaluate_query(query=q, k_values=clean_k)
                query_results.append(res)
                embedding_latencies.append(res.latency_metrics.embedding_latency_ms)
                search_latencies.append(res.latency_metrics.search_latency_ms)
                total_latencies.append(res.latency_metrics.total_latency_ms)
            except Exception as e:
                sanitized = scrub_secrets(str(e))
                logger.error(f"Error evaluating query '{q.query_id}': {sanitized}")
                raise

        n_queries = len(query_results)

        # 1. Compute mean retrieval metrics
        mean_precision: Dict[int, float] = {}
        mean_recall: Dict[int, float] = {}
        mean_ndcg: Dict[int, float] = {}

        for k in clean_k:
            prec_sum = sum(qr.retrieval_metrics.precision_at_k.get(k, 0.0) for qr in query_results)
            rec_sum = sum(qr.retrieval_metrics.recall_at_k.get(k, 0.0) for qr in query_results)
            ndcg_sum = sum(qr.retrieval_metrics.ndcg_at_k.get(k, 0.0) for qr in query_results)
            mean_precision[k] = round(prec_sum / float(n_queries), 4)
            mean_recall[k] = round(rec_sum / float(n_queries), 4)
            mean_ndcg[k] = round(ndcg_sum / float(n_queries), 4)

        mean_mrr = round(
            sum(qr.retrieval_metrics.mrr for qr in query_results) / float(n_queries), 4
        )

        mean_retrieval = RetrievalMetrics(
            precision_at_k=mean_precision,
            recall_at_k=mean_recall,
            mrr=mean_mrr,
            ndcg_at_k=mean_ndcg,
        )

        # 2. Compute mean groundedness metrics if answers were evaluated
        grounded_results = [
            qr.groundedness_metrics for qr in query_results if qr.groundedness_metrics is not None
        ]
        mean_groundedness: Optional[GroundednessMetrics] = None
        if grounded_results:
            n_g = len(grounded_results)
            mean_cite_cov = sum(g.citation_coverage for g in grounded_results) / float(n_g)
            mean_overlap = sum(g.evidence_overlap_score for g in grounded_results) / float(n_g)
            total_claims_sum = sum(g.total_claims for g in grounded_results)
            supported_sum = sum(g.supported_claims for g in grounded_results)
            unsupported_sum = sum(g.unsupported_claims for g in grounded_results)

            all_hallucinations = []
            for g in grounded_results:
                all_hallucinations.extend(g.hallucinated_citations)

            mean_groundedness = GroundednessMetrics(
                citation_coverage=round(mean_cite_cov, 4),
                evidence_overlap_score=round(mean_overlap, 4),
                total_claims=total_claims_sum,
                supported_claims=supported_sum,
                unsupported_claims=unsupported_sum,
                hallucinated_citations=sorted(list(set(all_hallucinations))),
            )

        # 3. Compute aggregate latency distributions
        latency_summary = AggregateLatencyMetrics(
            embedding_latency=calculate_latency_distribution(embedding_latencies),
            search_latency=calculate_latency_distribution(search_latencies),
            total_latency=calculate_latency_distribution(total_latencies),
        )

        # 4. Check quality gate thresholds
        threshold_checks: Dict[str, bool] = {}
        passed_quality_gates = True

        # Check precision threshold (checked against max evaluated k)
        max_k = max(clean_k)
        if "min_precision" in thresholds:
            target = thresholds["min_precision"]
            actual = mean_precision.get(max_k, 0.0)
            passed = actual >= target
            threshold_checks[f"precision_at_{max_k}>={target}"] = passed
            if not passed:
                passed_quality_gates = False
                if raise_on_threshold_failure:
                    raise EvaluationThresholdError(f"precision_at_{max_k}", target, actual)

        if "min_recall" in thresholds:
            target = thresholds["min_recall"]
            actual = mean_recall.get(max_k, 0.0)
            passed = actual >= target
            threshold_checks[f"recall_at_{max_k}>={target}"] = passed
            if not passed:
                passed_quality_gates = False
                if raise_on_threshold_failure:
                    raise EvaluationThresholdError(f"recall_at_{max_k}", target, actual)

        if "min_mrr" in thresholds:
            target = thresholds["min_mrr"]
            actual = mean_mrr
            passed = actual >= target
            threshold_checks[f"mrr>={target}"] = passed
            if not passed:
                passed_quality_gates = False
                if raise_on_threshold_failure:
                    raise EvaluationThresholdError("mrr", target, actual)

        if "min_ndcg" in thresholds:
            target = thresholds["min_ndcg"]
            actual = mean_ndcg.get(max_k, 0.0)
            passed = actual >= target
            threshold_checks[f"ndcg_at_{max_k}>={target}"] = passed
            if not passed:
                passed_quality_gates = False
                if raise_on_threshold_failure:
                    raise EvaluationThresholdError(f"ndcg_at_{max_k}", target, actual)

        if "max_latency_ms" in thresholds:
            target = thresholds["max_latency_ms"]
            actual = latency_summary.total_latency.p95
            passed = actual <= target
            threshold_checks[f"p95_latency<={target}ms"] = passed
            if not passed:
                passed_quality_gates = False
                if raise_on_threshold_failure:
                    raise EvaluationThresholdError("p95_total_latency_ms", target, actual)

        return EvaluationReport(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.name,
            total_queries=n_queries,
            k_values=clean_k,
            mean_retrieval_metrics=mean_retrieval,
            mean_groundedness_metrics=mean_groundedness,
            latency_summary=latency_summary,
            query_results=query_results,
            threshold_checks=threshold_checks,
            passed_quality_gates=passed_quality_gates,
            metadata={
                "embedding_provider": self.retriever.embedding_provider.provider_name,
                "vector_store": self.retriever.vector_store.store_name,
            },
        )
