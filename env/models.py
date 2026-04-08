"""
models.py - Pydantic models for AIRC environment.
"""

from pydantic import BaseModel
from typing import List


class Incident(BaseModel):
    id: int
    type: str
    severity: float
    deadline: int
    status: str  # "pending" | "resolved"


class State(BaseModel):
    time: int
    system_health: float
    incidents: List[Incident]