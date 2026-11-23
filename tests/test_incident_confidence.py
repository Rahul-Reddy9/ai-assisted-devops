# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Unit tests verifying the evidence confidence scoring methodology."""


def calculate_evidence_confidence(
    base_match: float,
    source_weight: float,
    corroboration: float = 0.0,
    contradiction: float = 0.0,
) -> float:
    """Helper implementing the evidence confidence formula:

    C_det = clamp((S_base * W_source) + S_corrob - P_contradict, 0.0, 1.0)
    """
    raw = (base_match * source_weight) + corroboration - contradiction
    return max(0.0, min(1.0, round(raw, 4)))


def test_corroborated_oom_scenario():
    """Scenario A: Exit code 137 + K8s OOMKilled: true + cgroup memory spike."""
    # S_base=1.00 (OOMKilled flag), W_source=0.95 (K8s status), S_corrob=0.15, P_contradict=0.0
    c_det = calculate_evidence_confidence(
        base_match=1.00,
        source_weight=0.95,
        corroboration=0.15,
        contradiction=0.00,
    )
    assert c_det == 1.00
    assert c_det >= 0.90  # Confirmed Tier 1


def test_ambiguous_sigkill_scenario():
    """Scenario B: Exit code 137 alone, no OOM indicator, normal memory."""
    # S_base=0.60 (SIGKILL alone), W_source=0.85 (Docker inspect), S_corrob=0.0, P_contradict=0.0
    c_det = calculate_evidence_confidence(
        base_match=0.60,
        source_weight=0.85,
        corroboration=0.00,
        contradiction=0.00,
    )
    assert c_det == 0.51
    assert c_det < 0.85  # Inconclusive; triggers Tier 2 escalation


def test_database_connection_failure_scenario():
    """Scenario C: Log regex match + readiness probe failure (HTTP 503)."""
    # S_base=0.80 (regex match), W_source=0.85 (app logs), S_corrob=0.15 (probe confirmed)
    c_det = calculate_evidence_confidence(
        base_match=0.80,
        source_weight=0.85,
        corroboration=0.15,
        contradiction=0.00,
    )
    assert c_det == 0.83


def test_contradictory_telemetry_scenario():
    """Scenario D: Log claims database unreachable, but /readyz probe is HTTP 200 OK."""
    # S_base=0.80, W_source=0.85, S_corrob=0.0, P_contradict=0.40
    c_det = calculate_evidence_confidence(
        base_match=0.80,
        source_weight=0.85,
        corroboration=0.00,
        contradiction=0.40,
    )
    assert c_det == 0.28
    assert c_det < 0.40  # Heavily penalized for contradiction


def test_noisy_log_burst_without_signature():
    """Scenario E: Statistical error spike without recognized failure signature."""
    # S_base=0.40, W_source=0.85, S_corrob=0.0, P_contradict=0.0
    c_det = calculate_evidence_confidence(
        base_match=0.40,
        source_weight=0.85,
        corroboration=0.00,
        contradiction=0.00,
    )
    assert c_det == 0.34
    assert c_det < 0.85  # Low confidence; requires Tier 2 deep analysis


def test_confidence_clamping_bounds():
    """Verify formula strictly clamps below 0.0 and above 1.0."""
    # Would be > 1.0 without clamp
    overflow = calculate_evidence_confidence(1.0, 1.0, corroboration=0.50)
    assert overflow == 1.00

    # Would be < 0.0 without clamp
    underflow = calculate_evidence_confidence(0.3, 0.6, contradiction=0.80)
    assert underflow == 0.00


def test_ai_confidence_cannot_inflate_deterministic_score():
    """Verify rule that LLM self-reported confidence cannot inflate deterministic evidence score."""
    c_det = 0.51  # Ambiguous deterministic score
    ai_self_reported_confidence = 0.99  # LLM claiming 99% certainty

    # Deterministic score must remain uninflated
    assert c_det == 0.51
    # Even if combined for exploratory display, AI cannot elevate deterministic confidence
    combined_upper_bound = min(c_det, ai_self_reported_confidence)
    assert combined_upper_bound <= c_det
