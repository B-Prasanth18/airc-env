"""
server/app.py - FastAPI server for AIRC OpenEnv environment.
Exposes /reset, /step, /state endpoints on port 7860.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from env.environment import AIRCEnv

app = FastAPI(title="AIRC OpenEnv", version="1.0.0")
env = AIRCEnv()


# =====================================================
# Request Models
# =====================================================
class ActionRequest(BaseModel):
    action: str


class ResetRequest(BaseModel):
    difficulty: str = "easy"


# =====================================================
# Endpoints
# =====================================================

@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment and return initial observation."""
    difficulty = req.difficulty if req else "easy"
    state = env.reset(difficulty)
    return {
        "observation": state.dict(),
        "reward": 0.0,
        "done": False,
        "info": {}
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


@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================================
# Entry point - required by openenv validate
# =====================================================
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()