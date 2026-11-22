# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Log analytics, temporal spike detection, and AI-assisted anomaly identification."""

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional
from devops_ai.core.logging import get_logger
from devops_ai.observability.log_parser import LogEntry

logger = get_logger("observability.analyzer")


@dataclass
class BurstDetectionResult:
    """Incident report for detected error log volume spikes."""

    has_burst: bool
    error_count: int
    window_start: Optional[datetime.datetime]
    window_end: Optional[datetime.datetime]
    message: str


@dataclass
class AnomalyDetectionResult:
    """Report for statistical or machine learning log anomalies."""

    total_logs: int
    anomaly_count: int
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "isolation_forest"


class LogAnalyzer:
    """Analyzes structured log streams for operational anomalies and error bursts."""

    def detect_error_bursts(
        self,
        logs: List[LogEntry],
        window_seconds: int = 30,
        threshold: int = 3,
    ) -> List[BurstDetectionResult]:
        """Detect temporal error spikes using a sliding window algorithm.

        Modernizes Day 6 simple_log_analysis.py logic into a robust, pure-Python
        sliding-window detector without mandatory external dependencies.
        """
        if not logs or threshold < 1:
            return []

        # Filter error and critical logs
        error_logs = [
            log for log in logs if log.level.upper() in ("ERROR", "ERR", "CRITICAL", "FATAL")
        ]
        if not error_logs:
            return []

        # Sort chronologically
        sorted_errors = sorted(error_logs, key=lambda x: x.timestamp)
        window_delta = datetime.timedelta(seconds=window_seconds)

        bursts: List[BurstDetectionResult] = []
        i = 0
        n = len(sorted_errors)

        while i < n:
            window_start = sorted_errors[i].timestamp
            window_end = window_start + window_delta

            # Find all errors falling within [window_start, window_end]
            j = i
            while j < n and sorted_errors[j].timestamp <= window_end:
                j += 1

            count = j - i
            if count > threshold:
                burst = BurstDetectionResult(
                    has_burst=True,
                    error_count=count,
                    window_start=window_start,
                    window_end=sorted_errors[j - 1].timestamp,
                    message=(
                        f"Detected {count} error logs within {window_seconds}s "
                        f"starting at {window_start.strftime('%Y-%m-%d %H:%M:%S')} (threshold: {threshold})"
                    ),
                )
                bursts.append(burst)
                # Advance i to j to avoid reporting duplicate sub-windows of the same spike
                i = j
            else:
                i += 1

        return bursts

    def detect_anomalies(
        self,
        logs: List[LogEntry],
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> AnomalyDetectionResult:
        """Detect operational anomalies using IsolationForest or statistical fallback.

        Modernizes Day 6 aiops_log_analysis.py by engineering level severity
        scores and message length features with deterministic testing support.
        """
        if not logs:
            return AnomalyDetectionResult(
                total_logs=0, anomaly_count=0, anomalies=[], method="none"
            )

        bounded_contamination = max(0.01, min(0.5, float(contamination)))

        # Feature extraction
        features = [[entry.level_score, len(entry.message)] for entry in logs]

        # 1. Attempt Scikit-Learn IsolationForest
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]

            X = np.array(features, dtype=float)
            model = IsolationForest(
                contamination=bounded_contamination,
                random_state=random_state,
            )
            predictions = model.fit_predict(X)

            anomalies: List[Dict[str, Any]] = []
            for idx, (entry, pred) in enumerate(zip(logs, predictions)):
                if pred == -1:
                    anomalies.append(
                        {
                            "index": idx,
                            "timestamp": entry.timestamp.isoformat(),
                            "level": entry.level,
                            "message": entry.message,
                            "level_score": entry.level_score,
                            "message_length": len(entry.message),
                        }
                    )

            return AnomalyDetectionResult(
                total_logs=len(logs),
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                method="isolation_forest",
            )
        except ImportError:
            logger.info(
                "scikit-learn is not available; falling back to statistical percentile anomaly detection."
            )

        # 2. Statistical fallback (zero-dependency)
        # Compute combined anomaly score: level_score * log(message_length + 1)
        import math

        scores = [
            (idx, entry, entry.level_score * math.log1p(len(entry.message)))
            for idx, entry in enumerate(logs)
        ]
        # Sort descending by severity score
        scores.sort(key=lambda x: x[2], reverse=True)

        cutoff = max(1, int(len(logs) * bounded_contamination))
        top_anomalies = scores[:cutoff]

        anomaly_records: List[Dict[str, Any]] = [
            {
                "index": idx,
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level,
                "message": entry.message,
                "score": round(score, 3),
            }
            for idx, entry, score in top_anomalies
        ]

        return AnomalyDetectionResult(
            total_logs=len(logs),
            anomaly_count=len(anomaly_records),
            anomalies=anomaly_records,
            method="statistical_percentile",
        )
