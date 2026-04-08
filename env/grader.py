"""
grader.py - Per-task graders for AIRC environment.
Each grader returns a normalized score in [0.0, 1.0].
"""


def compute_score(env) -> float:
    """Generic score computation used by inference.py"""
    incidents = env.incidents
    if not incidents:
        return 0.0
    total_severity = sum(i.severity for i in incidents)
    if total_severity == 0:
        return 0.0
    resolved_severity = sum(i.severity for i in incidents if i.status == "resolved")
    resolution_score = resolved_severity / total_severity
    health_bonus = max(0.0, env.system_health) * 0.2
    raw_score = (resolution_score * 0.8) + health_bonus
    return float(min(max(raw_score, 0.0), 1.0))


def grade_easy(env) -> float:
    """
    Grader for airc_easy task.
    2 incidents, generous deadlines.
    Score based on resolution rate and system health.
    """
    incidents = env.incidents
    if not incidents:
        return 0.0
    resolved = sum(1 for i in incidents if i.status == "resolved")
    total = len(incidents)
    resolution_rate = resolved / total
    health_score = max(0.0, env.system_health)
    score = (resolution_rate * 0.7) + (health_score * 0.3)
    return float(min(max(score, 0.0), 1.0))


def grade_medium(env) -> float:
    """
    Grader for airc_medium task.
    4 incidents, mixed severity, tighter deadlines.
    Score weighted by severity of resolved incidents.
    """
    incidents = env.incidents
    if not incidents:
        return 0.0
    total_severity = sum(i.severity for i in incidents)
    if total_severity == 0:
        return 0.0
    resolved_severity = sum(i.severity for i in incidents if i.status == "resolved")
    severity_score = resolved_severity / total_severity
    health_score = max(0.0, env.system_health)
    score = (severity_score * 0.75) + (health_score * 0.25)
    return float(min(max(score, 0.0), 1.0))


def grade_hard(env) -> float:
    """
    Grader for airc_hard task.
    6+ incidents, dynamic spawning, tight deadlines.
    Penalizes for unresolved high-severity incidents.
    """
    incidents = env.incidents
    if not incidents:
        return 0.0
    total_severity = sum(i.severity for i in incidents)
    if total_severity == 0:
        return 0.0
    resolved_severity = sum(i.severity for i in incidents if i.status == "resolved")
    severity_score = resolved_severity / total_severity
    health_score = max(0.0, env.system_health)
    # Hard mode: health matters more, penalty for low resolution
    score = (severity_score * 0.6) + (health_score * 0.4)
    return float(min(max(score, 0.0), 1.0))