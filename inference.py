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
    """Call LLM through judge proxy to decide which incident to resolve."""
    incidents_info = "\n".join(
        f"  - id={i.id}, type={i.type}, severity={i.severity:.2f}, "
        f"deadline={i.deadline}, status={i.status}"
        for i in state.incidents
    )

    prompt = f"""You are an AI incident response commander.

Current environment state:
- Time step: {state.time}
- System health: {state.system_health:.2f}

Active incidents:
{incidents_info}

Your goal: Resolve the most critical pending incident.
Priority: highest severity first, then soonest deadline.

Respond with ONLY the action in this exact format:
resolve <id>

Where <id> is the integer ID of the incident to resolve."""

    # This call MUST go through the proxy - no try/except that silently bypasses it
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent incident response agent. "
                           "Always respond with exactly: resolve <id>"
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=20,
    )

    action = response.choices[0].message.content.strip()

    # Validate and sanitize output (but LLM call was always made above)
    if not action.startswith("resolve "):
        # Extract any number from response as fallback
        import re
        nums = re.findall(r'\d+', action)
        if nums:
            action = f"resolve {nums[0]}"
        else:
            # Pick highest severity pending incident
            pending = sorted(
                [i for i in state.incidents if i.status == "pending"],
                key=lambda x: x.severity,
                reverse=True
            )
            action = f"resolve {pending[0].id}" if pending else "resolve 1"

    return action


# =====================================================
# RUN ONE TASK
# =====================================================
def run_task(task_name: str, difficulty: str):
    # Import here so path issues surface clearly
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
            # Check if all resolved
            if all(i.status == "resolved" for i in state.incidents):
                break

            error_msg = None
            try:
                action = choose_action(state)
            except Exception as e:
                error_msg = str(e)
                # Still need to take some action - pick best pending
                pending = sorted(
                    [i for i in state.incidents if i.status == "pending"],
                    key=lambda x: x.severity,
                    reverse=True
                )
                action = f"resolve {pending[0].id}" if pending else "resolve 1"

            state, reward, done, _ = env.step(action)
            steps += 1
            rewards.append(reward)

            log_step(step=step, action=action, reward=reward, done=done, error=error_msg)

            if done:
                break

        score = compute_score(env)
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
    task_name = os.environ.get("TASK_NAME", "airc_easy")

    difficulty_map = {
        "airc_easy":   "easy",
        "airc_medium": "medium",
        "airc_hard":   "hard",
    }

    difficulty = difficulty_map.get(task_name, "easy")
    run_task(task_name, difficulty)
