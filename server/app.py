"""
server/app.py - FastAPI server for AIRC OpenEnv environment.
Exposes /reset, /step, /state, /tasks endpoints on port 7860.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from env.environment import AIRCEnv
from env.grader import grade_easy, grade_medium, grade_hard, compute_score

app = FastAPI(title="AIRC OpenEnv", version="1.0.0")
env = AIRCEnv()

# Task registry with graders
TASKS = {
    "airc_easy": {
        "name": "airc_easy",
        "description": "2 incidents, moderate severity, generous deadlines.",
        "difficulty": "easy",
        "grader": grade_easy,
    },
    "airc_medium": {
        "name": "airc_medium",
        "description": "4 incidents with mixed severity and tighter deadlines.",
        "difficulty": "medium",
        "grader": grade_medium,
    },
    "airc_hard": {
        "name": "airc_hard",
        "description": "6 incidents with dynamic spawning, tight deadlines.",
        "difficulty": "hard",
        "grader": grade_hard,
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
    """Return list of tasks with grader info."""
    return {
        "tasks": [
            {
                "name": t["name"],
                "description": t["description"],
                "difficulty": t["difficulty"],
                "has_grader": True,
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment and return initial observation."""
    if req is None:
        req = ResetRequest()

    # Map task name to difficulty
    task_difficulty_map = {
        "airc_easy": "easy",
        "airc_medium": "medium",
        "airc_hard": "hard",
    }
    difficulty = task_difficulty_map.get(req.task, req.difficulty)
    state = env.reset(difficulty)

    return {
        "observation": state.dict(),
        "reward": 0.0,
        "done": False,
        "info": {"task": req.task}
    }


@app.post("/step")
def step(req: ActionRequest):
    """Execute one action and return (observation, reward, done, info)."""
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
    """Grade the current episode for a given task."""
    grader_fn = TASKS.get(task, TASKS["airc_easy"])["grader"]
    score = grader_fn(env)
    return {
        "task": task,
        "score": score,
        "success": score >= 0.3,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================================
# Entry point
# =====================================================
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()