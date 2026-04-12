"""
server/app.py - FastAPI server for AIRC OpenEnv environment.
Exposes /reset, /step, /state, /tasks, /grade, /health on port 7860.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from env.environment import AIRCEnv
from env.grader import grade_easy, grade_medium, grade_hard, compute_score

app = FastAPI(
    title="AIRC OpenEnv — AI CrisisOps Commander",
    description="Real-world IT incident response environment for RL agent training.",
    version="1.0.0",
)

env = AIRCEnv()

TASKS = {
    "airc_easy": {
        "name": "airc_easy",
        "description": "2 incidents (server P2 + network P3). Generous deadlines. "
                       "Agent learns basic triage and resolution prioritization.",
        "difficulty": "easy",
        "grader": grade_easy,
        "max_steps": 10,
        "incidents": 2,
    },
    "airc_medium": {
        "name": "airc_medium",
        "description": "4 incidents (server P1, security P1, network P2, database P1). "
                       "Mixed severity, tighter deadlines. Agent must prioritize by SLA risk.",
        "difficulty": "medium",
        "grader": grade_medium,
        "max_steps": 10,
        "incidents": 4,
    },
    "airc_hard": {
        "name": "airc_hard",
        "description": "6 incidents (all P1/P2), 30% chance of new incident each step. "
                       "Tight deadlines. Tests frontier model capability under extreme pressure.",
        "difficulty": "hard",
        "grader": grade_hard,
        "max_steps": 10,
        "incidents": "6+",
    },
}


# =====================================================
# Request Models
# =====================================================
class ActionRequest(BaseModel):
    action: str


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    task: str = "airc_easy"


# =====================================================
# Endpoints
# =====================================================

@app.get("/tasks")
def get_tasks():
    """Return all available tasks with grader metadata."""
    return {
        "tasks": [
            {
                "name": t["name"],
                "description": t["description"],
                "difficulty": t["difficulty"],
                "has_grader": True,
                "max_steps": t["max_steps"],
                "incidents": t["incidents"],
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment and return initial observation."""
    if req is None:
        req = ResetRequest()

    task_difficulty_map = {
        "airc_easy":   "easy",
        "airc_medium": "medium",
        "airc_hard":   "hard",
    }
    difficulty = task_difficulty_map.get(req.task, req.difficulty)
    state = env.reset(difficulty)

    return {
        "observation": state.dict(),
        "reward": 0.0,
        "done": False,
        "info": {"task": req.task, "difficulty": difficulty}
    }


@app.post("/step")
def step(req: ActionRequest):
    """
    Execute one action.

    Supported actions:
        resolve <id>   - Resolve a pending/triaged/escalated incident
        triage <id>    - Investigate incident (reduces ongoing penalty)
        escalate <id>  - Escalate P1 incident to extend SLA deadline
        assign <id>    - Assign an agent to speed resolution
    """
    state, reward, done, info = env.step(req.action)
    return {
        "observation": state.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def get_state():
    """Return current environment state."""
    return env.state().dict()


@app.post("/grade")
def grade(task: str = "airc_easy"):
    """Grade the current episode for a given task. Returns score in [0.0, 1.0]."""
    task_info = TASKS.get(task, TASKS["airc_easy"])
    score = task_info["grader"](env)
    return {
        "task": task,
        "score": round(score, 4),
        "success": score >= 0.3,
        "resolved": env.resolved_count,
        "sla_breaches": env.sla_breaches,
        "system_health": env.system_health,
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "env": "airc_env", "version": "1.0.0"}


@app.get("/")
def root():
    """Root endpoint with environment info."""
    return {
        "name": "AIRC OpenEnv",
        "description": "AI CrisisOps Commander — Real-world IT incident response environment",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grade", "/health"],
        "tasks": list(TASKS.keys()),
    }


# =====================================================
# Entry point
# =====================================================
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()