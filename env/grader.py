"""
grader.py - Deterministic per-task graders for AIRC environment.

Each grader returns a normalized score in [0.0, 1.0].
Graders are deterministic — same episode state always produces same score.

Scoring philosophy:
    - Resolution rate (weighted by severity) is the primary signal
    - SLA compliance adds bonus
    - System health preservation adds bonus
    - SLA breaches penalize
"""


def compute_score(env) -> float:
    """
    Generic score computation used by inference.py.
    Returns normalized float in [0.0, 1.0].
    """
    incidents = env.incidents
    if not incidents:
        return 0.0

    total_severity = sum(i.severity for i in incidents)
    if total_severity == 0:
        return 0.0

    # Weighted resolution rate
    resolved_severity = sum(
        i.severity for i in incidents if i.status == "resolved"
    )
    resolution_score = resolved_severity / total_severity

    # Health preservation bonus (up to 0.15)
    health_bonus = max(0.0, env.system_health) * 0.15

    # SLA breach penalty
    total = len(incidents)
    breach_penalty = (env.sla_breaches / max(total, 1)) * 0.15

    raw = (resolution_score * 0.7) + health_bonus - breach_penalty
    return float(min(max(raw, 0.0), 1.0))


def grade_easy(env) -> float:
    """
    Grader for airc_easy task.

    Scenario: 2 incidents (server + network), moderate severity.
    Objective: Resolve both before system health drops below 0.5.

    Scoring:
        - 70% weight: resolution rate (both incidents resolved = full score)
        - 20% weight: system health at episode end
        - 10% weight: no SLA breaches
    """
    incidents = env.incidents
    if not incidents:
        return 0.0

    resolved = sum(1 for i in incidents if i.status == "resolved")
    total = len(incidents)
    resolution_rate = resolved / total

    health_score = max(0.0, min(1.0, env.system_health))
    sla_score = 1.0 if env.sla_breaches == 0 else max(0.0, 1.0 - env.sla_breaches * 0.3)

    score = (resolution_rate * 0.7) + (health_score * 0.2) + (sla_score * 0.1)
    return float(min(max(score, 0.0), 1.0))


def grade_medium(env) -> float:
    """
    Grader for airc_medium task.

    Scenario: 4 incidents (server P1, security P1, network P2, database P1).
    Objective: Resolve high-severity incidents first, maintain SLA compliance.

    Scoring:
        - 60% weight: severity-weighted resolution rate
        - 25% weight: system health preservation
        - 15% weight: SLA compliance (fewer breaches = higher score)
    """
    incidents = env.incidents
    if not incidents:
        return 0.0

    total_severity = sum(i.severity for i in incidents)
    if total_severity == 0:
        return 0.0

    resolved_severity = sum(
        i.severity for i in incidents if i.status == "resolved"
    )
    severity_score = resolved_severity / total_severity

    health_score = max(0.0, min(1.0, env.system_health))

    total = len(incidents)
    sla_score = max(0.0, 1.0 - (env.sla_breaches / max(total, 1)))

    score = (severity_score * 0.6) + (health_score * 0.25) + (sla_score * 0.15)
    return float(min(max(score, 0.0), 1.0))


def grade_hard(env) -> float:
    """
    Grader for airc_hard task.

    Scenario: 6 incidents (all P1/P2), tight deadlines, dynamic spawning.
    Objective: Maximize resolution of critical incidents under extreme pressure.
    This task is designed to genuinely challenge frontier models.

    Scoring:
        - 55% weight: severity-weighted resolution rate
        - 30% weight: system health (critical in hard mode)
        - 15% weight: SLA compliance (very hard to maintain in hard mode)

    Partial credit: Even resolving 1-2 P1 incidents scores meaningfully.
    """
    incidents = env.incidents
    if not incidents:
        return 0.0

    total_severity = sum(i.severity for i in incidents)
    if total_severity == 0:
        return 0.0

    resolved_severity = sum(
        i.severity for i in incidents if i.status == "resolved"
    )
    severity_score = resolved_severity / total_severity

    health_score = max(0.0, min(1.0, env.system_health))

    total = len(incidents)
    sla_score = max(0.0, 1.0 - (env.sla_breaches / max(total, 1)) * 0.5)

    score = (severity_score * 0.55) + (health_score * 0.30) + (sla_score * 0.15)
    return float(min(max(score, 0.0), 1.0))