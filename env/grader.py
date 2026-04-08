"""
grader.py - Scoring logic for AIRC environment.
compute_score() returns a normalized float in [0.0, 1.0].
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import AIRCEnv


def compute_score(env: "AIRCEnv") -> float:
    """
    Compute normalized score in [0.0, 1.0] based on:
    - Fraction of incidents resolved
    - System health remaining
    - Time efficiency
    """
    incidents = env.incidents

    if not incidents:
        return 0.0

    total = len(incidents)
    resolved = sum(1 for i in incidents if i.status == "resolved")

    # Base score: fraction resolved (weighted by severity)
    total_severity = sum(i.severity for i in incidents)
    if total_severity == 0:
        resolution_score = resolved / total
    else:
        resolved_severity = sum(
            i.severity for i in incidents if i.status == "resolved"
        )
        resolution_score = resolved_severity / total_severity

    # Health bonus (0 to 0.2)
    health_bonus = max(0.0, env.system_health) * 0.2

    # Combined score
    raw_score = (resolution_score * 0.8) + health_bonus

    # Clamp to [0.0, 1.0]
    return float(min(max(raw_score, 0.0), 1.0))