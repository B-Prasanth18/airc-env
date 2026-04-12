"""
models.py - Pydantic models for AIRC environment.

Action Space:
    - resolve <id>     : Resolve a pending incident immediately
    - triage <id>      : Investigate incident, reducing severity penalty
    - escalate <id>    : Escalate high-severity incident, extending deadline
    - assign <id>      : Assign an agent to work on the incident (reduces time cost)

Observation Space:
    State with time, system_health, active_agents, incidents list

Each Incident has:
    - id, type, severity, deadline, status, priority, sla_risk
"""

from pydantic import BaseModel, Field
from typing import List


class Incident(BaseModel):
    id: int = Field(description="Unique incident identifier")
    type: str = Field(description="Incident type: server | security | network | database")
    severity: float = Field(ge=0.0, le=1.0, description="Severity score 0.0-1.0")
    deadline: int = Field(description="Steps remaining before SLA breach")
    status: str = Field(description="pending | triaged | escalated | resolved")
    priority: str = Field(description="P1 | P2 | P3 based on severity")
    sla_risk: float = Field(ge=0.0, le=1.0, description="SLA breach risk 0.0-1.0")


class State(BaseModel):
    episode_id: str = Field(description="Unique episode identifier")
    time: int = Field(description="Current time step")
    system_health: float = Field(ge=0.0, le=1.0, description="Overall system health 0.0-1.0")
    active_agents: int = Field(description="Number of available response agents")
    incidents: List[Incident] = Field(description="List of all incidents in this episode")
    resolved_count: int = Field(description="Number of incidents resolved so far")
    sla_breaches: int = Field(description="Number of SLA breaches so far")