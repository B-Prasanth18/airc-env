"""
environment.py - AIRC (AI Incident Response Commander) OpenEnv environment.

Real-world task: An AI agent acts as an SRE on-call responder, managing
IT incidents under time pressure. The agent must triage, escalate, assign,
and resolve incidents based on severity, SLA deadlines, and system health.

This models real enterprise incident management workflows used by companies
like Google, Meta, and Amazon for reliability engineering.

Action Space:
    resolve <id>  - Resolve a pending/triaged/escalated incident
    triage <id>   - Investigate incident: reduces ongoing penalty by 40%
    escalate <id> - Escalate P1 incident: extends deadline by 2 steps
    assign <id>   - Assign agent to incident: speeds resolution next step

Observation Space:
    State(episode_id, time, system_health, active_agents, incidents,
          resolved_count, sla_breaches)

Reward Design:
    Dense reward signal at every step (not just terminal):
    - +severity*10 + 5 for on-time resolution
    - +severity*10 - 3 for late resolution
    - +2 for successful triage (partial progress)
    - +1 for successful escalation of P1
    - -severity*2 per step for each unresolved incident (urgency)
    - -0.5 per step for each triaged (partial, reduced penalty)
    - -10 if system health reaches 0
    - -3 for invalid actions
"""

import random
import uuid
from .models import Incident, State


def _compute_priority(severity: float) -> str:
    """Compute priority label from severity score."""
    if severity >= 0.8:
        return "P1"
    elif severity >= 0.6:
        return "P2"
    else:
        return "P3"


def _compute_sla_risk(deadline: int, time: int) -> float:
    """Compute SLA breach risk based on deadline proximity."""
    if deadline <= 0:
        return 1.0
    steps_remaining = deadline - time
    if steps_remaining <= 1:
        return 1.0
    elif steps_remaining <= 2:
        return 0.8
    elif steps_remaining <= 3:
        return 0.5
    else:
        return max(0.0, 1.0 - (steps_remaining / 8.0))


# Fixed incident scenarios per difficulty for deterministic grading
EASY_SCENARIOS = [
    {"type": "server", "severity": 0.72, "deadline": 6},
    {"type": "network", "severity": 0.55, "deadline": 5},
]

MEDIUM_SCENARIOS = [
    {"type": "server", "severity": 0.91, "deadline": 4},
    {"type": "security", "severity": 0.78, "deadline": 3},
    {"type": "network", "severity": 0.61, "deadline": 5},
    {"type": "database", "severity": 0.85, "deadline": 4},
]

HARD_SCENARIOS = [
    {"type": "security", "severity": 0.97, "deadline": 3},
    {"type": "server", "severity": 0.88, "deadline": 2},
    {"type": "database", "severity": 0.93, "deadline": 3},
    {"type": "network", "severity": 0.75, "deadline": 4},
    {"type": "server", "severity": 0.82, "deadline": 3},
    {"type": "security", "severity": 0.79, "deadline": 2},
]


class AIRCEnv:
    """
    AI CrisisOps Commander — OpenEnv-compliant incident response environment.

    Models real SRE on-call incident management with:
    - Multi-type incidents (server, security, network, database)
    - Priority classification (P1/P2/P3)
    - SLA deadline tracking
    - Dense reward shaping throughout episode
    - Dynamic incident spawning in hard mode
    """

    DIFFICULTY_SCENARIOS = {
        "easy": EASY_SCENARIOS,
        "medium": MEDIUM_SCENARIOS,
        "hard": HARD_SCENARIOS,
    }

    def __init__(self):
        self.time = 0
        self.system_health = 1.0
        self.active_agents = 3
        self.incidents: list = []
        self.resolved_count = 0
        self.sla_breaches = 0
        self._episode_id = ""
        self._difficulty = "easy"
        self._assigned: set = set()  # incident ids with assigned agents

    def reset(self, difficulty: str = "easy") -> State:
        """
        Reset environment for a new episode.
        Uses fixed scenarios per difficulty for reproducibility.
        """
        self.time = 0
        self.system_health = 1.0
        self.active_agents = 3
        self.resolved_count = 0
        self.sla_breaches = 0
        self._episode_id = str(uuid.uuid4())[:8]
        self._difficulty = difficulty
        self._assigned = set()

        scenarios = self.DIFFICULTY_SCENARIOS.get(difficulty, EASY_SCENARIOS)

        self.incidents = [
            Incident(
                id=i + 1,
                type=s["type"],
                severity=s["severity"],
                deadline=s["deadline"],
                status="pending",
                priority=_compute_priority(s["severity"]),
                sla_risk=_compute_sla_risk(s["deadline"], 0),
            )
            for i, s in enumerate(scenarios)
        ]

        return self.state()

    def step(self, action: str):
        """
        Execute one action. Returns (State, reward, done, info).

        Supported actions:
            resolve <id>  - Fully resolve an incident
            triage <id>   - Investigate/triage an incident (partial progress)
            escalate <id> - Escalate a P1 incident to extend SLA deadline
            assign <id>   - Assign an agent to speed up resolution
        """
        reward = 0.0
        done = False
        info = {}

        self.time += 1
        self.system_health = round(self.system_health - 0.05, 4)

        parts = action.strip().split()
        action_type = parts[0] if parts else ""
        target_id = None

        if len(parts) == 2:
            try:
                target_id = int(parts[1])
            except ValueError:
                reward -= 3.0
                info["error"] = "invalid_id_format"
                return self._finalize_step(reward, done, info)

        # --- Action: RESOLVE ---
        if action_type == "resolve" and target_id is not None:
            matched = False
            for inc in self.incidents:
                if inc.id == target_id and inc.status in ("pending", "triaged", "escalated"):
                    base_reward = inc.severity * 10

                    # Bonus for assigned agent
                    if target_id in self._assigned:
                        base_reward += 2.0
                        self._assigned.discard(target_id)

                    # SLA timing bonus/penalty
                    if self.time <= inc.deadline:
                        reward += base_reward + 5.0
                        info["sla_met"] = True
                    else:
                        reward += base_reward - 3.0
                        self.sla_breaches += 1
                        info["sla_breached"] = True

                    inc.status = "resolved"
                    self.resolved_count += 1
                    matched = True
                    break

            if not matched:
                reward -= 3.0
                info["error"] = "invalid_target_or_already_resolved"

        # --- Action: TRIAGE ---
        elif action_type == "triage" and target_id is not None:
            matched = False
            for inc in self.incidents:
                if inc.id == target_id and inc.status == "pending":
                    inc.status = "triaged"
                    reward += 2.0  # partial progress reward
                    info["triaged"] = target_id
                    matched = True
                    break
            if not matched:
                reward -= 2.0
                info["error"] = "cannot_triage"

        # --- Action: ESCALATE ---
        elif action_type == "escalate" and target_id is not None:
            matched = False
            for inc in self.incidents:
                if inc.id == target_id and inc.status in ("pending", "triaged") and inc.priority == "P1":
                    inc.deadline += 2  # extend SLA deadline
                    inc.status = "escalated"
                    reward += 1.0
                    info["escalated"] = target_id
                    matched = True
                    break
            if not matched:
                reward -= 2.0
                info["error"] = "cannot_escalate_not_p1_or_invalid"

        # --- Action: ASSIGN ---
        elif action_type == "assign" and target_id is not None:
            matched = False
            for inc in self.incidents:
                if inc.id == target_id and inc.status in ("pending", "triaged") and self.active_agents > 0:
                    if target_id not in self._assigned:
                        self._assigned.add(target_id)
                        self.active_agents -= 1
                        reward += 0.5
                        info["assigned"] = target_id
                        matched = True
                    break
            if not matched:
                reward -= 1.0
                info["error"] = "cannot_assign"

        else:
            reward -= 3.0
            info["error"] = f"unknown_action_{action_type}"

        return self._finalize_step(reward, done, info)

    def _finalize_step(self, reward: float, done: bool, info: dict):
        """Apply ongoing penalties, update SLA risks, check terminal conditions."""

        # Ongoing penalty for unresolved incidents (dense signal)
        for inc in self.incidents:
            if inc.status == "pending":
                reward -= inc.severity * 2.0
            elif inc.status == "triaged":
                reward -= inc.severity * 0.8  # reduced penalty for triaged
            elif inc.status == "escalated":
                reward -= inc.severity * 1.2

            # Update SLA risk
            inc.sla_risk = _compute_sla_risk(inc.deadline, self.time)

        # Dynamic incident spawning in hard mode
        if self._difficulty == "hard" and random.random() < 0.25:
            new_id = max((i.id for i in self.incidents), default=0) + 1
            severity = round(random.uniform(0.6, 1.0), 2)
            self.incidents.append(
                Incident(
                    id=new_id,
                    type=random.choice(["server", "security", "network", "database"]),
                    severity=severity,
                    deadline=random.randint(2, 4),
                    status="pending",
                    priority=_compute_priority(severity),
                    sla_risk=_compute_sla_risk(3, self.time),
                )
            )

        # Terminal: all resolved
        if all(i.status == "resolved" for i in self.incidents):
            done = True
            reward += 5.0  # episode completion bonus

        # Terminal: system health depleted
        if self.system_health <= 0:
            self.system_health = 0.0
            done = True
            reward -= 10.0

        # Free up agents from resolved incidents
        resolved_ids = {i.id for i in self.incidents if i.status == "resolved"}
        freed = self._assigned & resolved_ids
        self.active_agents += len(freed)
        self._assigned -= freed

        return self.state(), round(reward, 2), done, info

    def state(self) -> State:
        """Return current environment state."""
        return State(
            episode_id=self._episode_id,
            time=self.time,
            system_health=self.system_health,
            active_agents=self.active_agents,
            incidents=list(self.incidents),
            resolved_count=self.resolved_count,
            sla_breaches=self.sla_breaches,
        )