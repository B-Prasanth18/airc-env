---
title: AIRC Env
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

# 🚨 AI CrisisOps Commander (AIRC)

An **OpenEnv-compliant** environment simulating real-world IT incident response. An AI agent must triage and resolve incidents under time pressure, balancing severity, deadlines, and system health.

## 🌍 Real-World Task

Models actual SRE/DevOps incident management workflows:
- Incidents arrive with type (server/security/network), severity (0.4–1.0), and deadline
- System health degrades 0.05 per step — agent must act fast
- Hard mode spawns new incidents dynamically mid-episode

## 📐 Action & Observation Spaces

**Action:** `resolve <id>` — string command to resolve a specific incident by ID

**Observation (State):**
```json
{
  "time": 3,
  "system_health": 0.85,
  "incidents": [
    {"id": 1, "type": "server", "severity": 0.87, "deadline": 4, "status": "pending"},
    {"id": 2, "type": "security", "severity": 0.62, "deadline": 6, "status": "resolved"}
  ]
}
```

**Reward shaping:**
- `+severity * 10` for resolving an incident
- `+5` if resolved before deadline, `-3` if after
- `-severity * 2` per step per unresolved incident (urgency pressure)
- `-10` if system health reaches 0
- `-5` for invalid action (wrong ID or already resolved)

## 🎯 Tasks

| Task | Difficulty | Incidents | Description |
|------|-----------|-----------|-------------|
| `airc_easy` | Easy | 2 | Gentle intro, generous deadlines |
| `airc_medium` | Medium | 4 | Mixed severity, tighter deadlines |
| `airc_hard` | Hard | 6+ | Dynamic spawning, tight deadlines |

Scores are normalized to **[0.0, 1.0]** by the grader.

## 🚀 Setup & Running

### Prerequisites
```bash
pip install openenv-core>=0.2.0
```

### Run server locally
```bash
pip install -r requirements.txt
pip install -e .
python -m server.app
```

### Docker
```bash
docker build -t airc-env .
docker run -p 7860:7860 airc-env
```

### Validate
```bash
openenv validate
```

### Run inference
```bash
export API_BASE_URL=<your-endpoint>
export API_KEY=<your-key>
export MODEL_NAME=<your-model>
export TASK_NAME=airc_easy
python inference.py
```

## 📁 Project Structure

```
.
├── inference.py          # Baseline inference script (root)
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container config
├── pyproject.toml        # Package config
├── requirements.txt      # Dependencies
├── env/
│   ├── __init__.py
│   ├── environment.py    # AIRCEnv class
│   ├── models.py         # Pydantic models (Incident, State)
│   └── grader.py         # compute_score() function
└── server/
    ├── __init__.py
    └── app.py            # FastAPI server (reset/step/state)
```
