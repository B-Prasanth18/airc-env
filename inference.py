"""
inference.py - AIRC Environment Baseline Inference Script
Must be placed at repo ROOT.
Uses API_BASE_URL, API_KEY, MODEL_NAME from environment variables (injected by judges).
"""

import os
import sys
from openai import OpenAI

# =====================================================
# ENV VARS - injected by judges, DO NOT hardcode
# =====================================================
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# =====================================================
# OpenAI client using judge-provided proxy
# =====================================================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

MAX_STEPS = 10


# =====================================================
# LOGGING - exact format required by spec
# =====================================================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# =====================================================
# LLM ACTION - ALWAYS calls through proxy
# =====================================================
def choose_action(state) -> str:
    """Call LLM through judge proxy to decide incident action."""
    pending = [i for i in state.incidents if i.status != "resolved"]
    incidents_info = "\n".join(
        f"  - id={i.id}, type={i.type}, severity={i.severity:.2f}, "
        f"priority={i.priority}, deadline={i.deadline}, "
        f"sla_risk={i.sla_risk:.2f}, status={i.status}"
        for i in state.incidents
    )

    prompt = f"""You are an expert SRE (Site Reliability Engineer) managing IT incidents.

Current State:
- Time step: {state.time}
- System health: {state.system_health:.2f}
- Active agents available: {state.active_agents}
- Resolved: {state.resolved_count} incidents
- SLA breaches so far: {state.sla_breaches}

All Incidents:
{incidents_info}

Available Actions:
  resolve <id>   - Fully resolve a pending/triaged/escalated incident
  triage <id>    - Investigate a pending incident (reduces ongoing penalty)
  escalate <id>  - Escalate a P1 incident to extend its deadline by 2 steps
  assign <id>    - Assign an agent to an incident (bonus reward on resolution)

Strategy Guide:
  1. ALWAYS prioritize P1 incidents with high sla_risk (close to 1.0)
  2. Use triage on incidents you can't resolve immediately
  3. Use escalate on P1 incidents with deadline <= 2
  4. Use assign when active_agents > 0 and a critical incident needs attention
  5. Resolve triaged incidents next (they have reduced penalty)

Respond with ONLY one action in exact format: <action_type> <id>
Example: resolve 2"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert SRE incident commander. "
                           "Respond with exactly one action: resolve/triage/escalate/assign followed by incident ID."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=20,
    )

    action = response.choices[0].message.content.strip().lower()

    # Validate action format
    valid_actions = ("resolve", "triage", "escalate", "assign")
    parts = action.split()

    if len(parts) == 2 and parts[0] in valid_actions:
        try:
            int(parts[1])
            return action
        except ValueError:
            pass

    # Fallback: resolve highest severity pending incident
    import re
    nums = re.findall(r'\d+', action)
    action_word = next((a for a in valid_actions if a in action), "resolve")

    pending_incidents = [i for i in state.incidents if i.status not in ("resolved",)]
    if pending_incidents and nums:
        return f"{action_word} {nums[0]}"

    if pending_incidents:
        # Pick highest SLA risk incident
        urgent = max(pending_incidents, key=lambda x: (x.sla_risk, x.severity))
        return f"resolve {urgent.id}"

    return "resolve 1"


# =====================================================
# RUN ONE TASK
# =====================================================
def run_task(task_name: str, difficulty: str):
    from env.environment import AIRCEnv
    from env.grader import compute_score

    env = AIRCEnv()
    rewards = []
    steps = 0
    score = 0.0
    success = False

    log_start(task=task_name, env="airc_env", model=MODEL_NAME)

    try:
        state = env.reset(difficulty)

        for step in range(1, MAX_STEPS + 1):
            if all(i.status == "resolved" for i in state.incidents):
                break

            error_msg = None
            try:
                action = choose_action(state)
            except Exception as e:
                error_msg = str(e)
                pending = [i for i in state.incidents if i.status != "resolved"]
                if pending:
                    urgent = max(pending, key=lambda x: (x.sla_risk, x.severity))
                    action = f"resolve {urgent.id}"
                else:
                    action = "resolve 1"

            state, reward, done, _ = env.step(action)
            steps += 1
            rewards.append(reward)

            log_step(step=step, action=action, reward=reward, done=done, error=error_msg)

            if done:
                break

        score = float(min(max(compute_score(env), 0.0), 1.0))
        success = score >= 0.3

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True, file=sys.stderr)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    task_name = os.environ.get("TASK_NAME", "")

    difficulty_map = {
        "airc_easy":   "easy",
        "airc_medium": "medium",
        "airc_hard":   "hard",
    }

    if task_name in difficulty_map:
        run_task(task_name, difficulty_map[task_name])
    else:
        # Run all 3 tasks when no specific TASK_NAME is set
        run_task("airc_easy", "easy")
        run_task("airc_medium", "medium")
        run_task("airc_hard", "hard")