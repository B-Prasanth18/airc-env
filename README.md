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

> **Real-world SRE incident response environment for training AI agents on enterprise reliability engineering.**

An **OpenEnv-compliant** environment where an AI agent acts as an on-call Site Reliability Engineer (SRE), triaging and resolving IT incidents under time pressure. Models actual incident management workflows used at scale by engineering teams.

---

## 🌍 Why This Environment?

Modern SRE teams face alert fatigue — dozens of incidents per hour, each with different severity, SLA deadlines, and resolution complexity. Training AI agents to assist in incident triage is a genuine open problem with immediate enterprise value.

AIRC simulates this with:
- **Priority classification** (P1/P2/P3) based on severity
- **SLA deadline tracking** with breach detection
- **System health degradation** (0.05 per step) — the clock is always ticking
- **Dynamic incident spawning** in hard mode (new incidents arrive mid-episode)
- **Dense reward shaping** — signal at every step, not just terminal

---

## 📐 Action Space

| Action | Format | Description |
|--------|--------|-------------|
| Resolve | `resolve <id>` | Fully resolve a pending/triaged/escalated incident |
| Triage | `triage <id>` | Investigate incident — reduces ongoing penalty by 60% |
| Escalate | `escalate <id>` | Escalate P1 incident — extends SLA deadline by 2 steps |
| Assign | `assign <id>` | Assign agent to incident — grants resolution bonus |

---

## 👁️ Observation Space

```json
{
  "episode_id": "a3f9b2c1",
  "time": 3,
  "system_health": 0.85,
  "active_agents": 2,
  "resolved_count": 1,
  "sla_breaches": 0,
  "incidents": [
    {
      "id": 1,
      "type": "security",
      "severity": 0.97,
      "deadline": 3,
      "status": "escalated",
      "priority": "P1",
      "sla_risk": 0.8
    },
    {
      "id": 2,
      "type": "server",
      "severity": 0.72,
      "deadline": 5,
      "status": "resolved",
      "priority": "P2",
      "sla_risk": 0.0
    }
  ]
}
```

---

## 🏆 Reward Function

Dense reward signal at every step (not just terminal):

| Event | Reward |
|-------|--------|
| Resolve incident on time | `+severity × 10 + 5` |
| Resolve incident late (SLA breach) | `+severity × 10 - 3` |
| Resolve with assigned agent | `+2 bonus` |
| Successful triage | `+2` |
| Successful P1 escalation | `+1` |
| Per step per pending incident | `-severity × 2` |
| Per step per triaged incident | `-severity × 0.8` |
| System health reaches 0 | `-10` |
| Episode completion (all resolved) | `+5` |
| Invalid action | `-3` |

---

## 🎯 Tasks

| Task | Incidents | Difficulty | Description |
|------|-----------|------------|-------------|
| `airc_easy` | 2 | Easy | Server P2 + Network P3, generous deadlines |
| `airc_medium` | 4 | Medium | Mixed P1/P2, tighter deadlines, SLA pressure |
| `airc_hard` | 6+ | Hard | All P1/P2, dynamic spawning, extreme pressure |

Scores normalized to **[0.0, 1.0]** by per-task graders.

### Grading Criteria

**Easy:** Resolution rate (70%) + Health (20%) + SLA compliance (10%)

**Medium:** Severity-weighted resolution (60%) + Health (25%) + SLA (15%)

**Hard:** Severity-weighted resolution (55%) + Health (30%) + SLA (15%)

---

## 🚀 Quick Start

### Run locally
```bash
pip install -r requirements.txt
pip install -e .
python -m server.app
# Server running at http://localhost:7860
```

### Docker
```bash
docker build -t airc-env .
docker run -p 7860:7860 airc-env
```

### Validate
```bash
openenv validate
# [OK] airc_env: Ready for multi-mode deployment
```

### Run inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export API_KEY=your_hf_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Example interaction
```bash
# Reset for medium task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "airc_medium"}'

# Triage the most critical incident
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "triage 1"}'

# Escalate a P1 incident with tight deadline
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "escalate 2"}'

# Resolve it
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "resolve 1"}'

# Get final grade
curl http://localhost:7860/grade?task=airc_medium
```

---

## 📁 Project Structure

```
.
├── inference.py          # Baseline LLM inference script (root)
├── openenv.yaml          # OpenEnv metadata and task definitions
├── Dockerfile            # Container configuration
├── pyproject.toml        # Package configuration
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── env/
│   ├── __init__.py
│   ├── environment.py    # AIRCEnv — core environment logic
│   ├── models.py         # Pydantic models (Incident, State)
│   └── grader.py         # Per-task deterministic graders
└── server/
    ├── __init__.py
    └── app.py            # FastAPI server (reset/step/state/tasks/grade)
```

---

## 🔗 Links

- **HuggingFace Space**: https://huggingface.co/spaces/Prasanth11b/airc-env
- **GitHub**: https://github.com/B-Prasanth18/airc-env
