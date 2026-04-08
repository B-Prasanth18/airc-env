"""
environment.py - AIRC (AI Incident Response Commander) OpenEnv environment.

Real-world task: An AI agent must triage and resolve IT incidents under time pressure,
prioritizing by severity and deadline, while system health degrades over time.
"""

import random
from .models import Incident, State


class AIRCEnv:
    """
    AI CrisisOps Commander environment.

    Action space:  "resolve <id>"  where id is an integer incident ID
    Observation:   State with time, system_health, list of Incidents
    Reward:        Shaped per step based on resolution quality and urgency
    Episode ends:  All incidents resolved OR system health reaches 0
    """

    def __init__(self):
        self.time = 0
        self.system_health = 1.0
        self.incidents: list = []

    def reset(self, difficulty: str = "easy") -> State:
        """Reset environment. Returns initial State observation."""
        self.time = 0
        self.system_health = 1.0

        count = {"easy": 2, "medium": 4, "hard": 6}.get(difficulty, 2)

        self.incidents = [
            Incident(
                id=i,
                type=random.choice(["server", "security", "network"]),
                severity=round(random.uniform(0.4, 1.0), 2),
                status="pending",
                deadline=random.randint(2, 6),
            )
            for i in range(1, count + 1)
        ]

        return self.state()

    def step(self, action: str):
        """
        Execute action. Returns (State, reward, done, info).
        Action format: "resolve <id>"
        """
        reward = 0.0
        done = False

        self.time += 1
        self.system_health = round(self.system_health - 0.05, 4)

        parts = action.strip().split()

        if len(parts) == 2 and parts[0] == "resolve":
            try:
                target_id = int(parts[1])
            except ValueError:
                return self.state(), -5.0, False, {"error": "invalid_id"}

            matched = False
            for incident in self.incidents:
                if incident.id == target_id and incident.status == "pending":
                    # Reward proportional to severity
                    reward += incident.severity * 10

                    # Urgency bonus/penalty
                    if self.time <= incident.deadline:
                        reward += 5.0  # resolved before deadline
                    else:
                        reward -= 3.0  # resolved after deadline

                    incident.status = "resolved"
                    matched = True
                    break

            if not matched:
                reward -= 5.0  # invalid target (already resolved or wrong id)
        else:
            reward -= 2.0  # malformed action

        # Ongoing penalty for unresolved incidents
        for incident in self.incidents:
            if incident.status == "pending":
                reward -= incident.severity * 2

        # Dynamic new incidents in hard mode (>=4 incidents)
        if len(self.incidents) >= 4 and random.random() < 0.3:
            new_id = max(i.id for i in self.incidents) + 1
            self.incidents.append(
                Incident(
                    id=new_id,
                    type=random.choice(["server", "security", "network"]),
                    severity=round(random.uniform(0.5, 1.0), 2),
                    status="pending",
                    deadline=random.randint(2, 5),
                )
            )

        # Terminal conditions
        if all(i.status == "resolved" for i in self.incidents):
            done = True

        if self.system_health <= 0:
            self.system_health = 0.0
            done = True
            reward -= 10.0

        return self.state(), round(reward, 2), done, {}

    def state(self) -> State:
        """Return current state observation."""
        return State(
            time=self.time,
            system_health=self.system_health,
            incidents=list(self.incidents),
        )