import asyncio
import os
import json
import textwrap
import time
from typing import List, Optional

from openai import OpenAI
from pydantic import ValidationError

from client import DataJanitorEnv
from models import DataJanitorAction

# ==========================================
# MANDATORY HACKATHON ENVIRONMENT VARIABLES
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

# Task Config
TASK_NAME = os.getenv("TASK_NAME", "hard")
BENCHMARK = os.getenv("BENCHMARK", "data_janitor")
MAX_STEPS = 40
SUCCESS_SCORE_THRESHOLD = 0.85 

# ==========================================
# SYSTEM PROMPT (Fixed for strict JSON literals)
# ==========================================
SYSTEM_PROMPT = textwrap.dedent("""
    You are an elite Autonomous Data Engineer.
    Goal: Transform messy datasets into ML-Ready pipelines to maximize the final ML score (0.0 to 1.0).
    
    HEURISTICS & ALLOWED STRATEGIES:
    1. MISSING DATA: 'fill_missing' (strategies: 'mean', 'median', 'mode', 'constant').
    2. SKEWNESS: 'transform_distribution' (strategies: 'log1p', 'sqrt' ONLY. Never use 'log').
    3. OUTLIERS: 'handle_outliers' (strategies: 'clip_percentile', 'drop_zscore'). NEVER use on binary (0/1) columns like 'hypertension' or 'heart_disease'.
    4. ENCODING: 'encode_categorical' (strategies: 'one_hot', 'ordinal', 'target_encode').
    5. SCALING: 'scale_feature' (strategies: 'standard', 'minmax', 'robust').
    6. NEVER alter the target column.
    7. FATAL ERROR: DO NOT repeat an action on the same column. If you just applied an action to a column, YOU MUST pick a different column or a different action next.
    8. JUNK/IDs: You MUST use 'drop_column' on any high-cardinality string identifiers (like Names, IDs, or raw text) before submitting.
    
    You must output EXACTLY ONE JSON object matching one of these structures. Output raw JSON only. Do not include markdown formatting or ```json blocks:
    {"command": {"action_type": "drop_column", "column_name": "..."}}
    {"command": {"action_type": "fill_missing", "column_name": "...", "strategy": "median", "constant_value": "..."}} 
    {"command": {"action_type": "handle_outliers", "column_name": "...", "strategy": "clip_percentile", "lower_percentile": 0.05, "upper_percentile": 0.95}} 
    {"command": {"action_type": "transform_distribution", "column_name": "...", "strategy": "log1p"}} 
    {"command": {"action_type": "encode_categorical", "column_name": "...", "strategy": "one_hot"}} 
    {"command": {"action_type": "scale_feature", "column_name": "...", "strategy": "standard"}}
    {"command": {"action_type": "submit", "notes": "..."}}
""").strip()

# ==========================================
# STRICT LOGGING FUNCTIONS (Do Not Modify)
# ==========================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace('\n', ' ') if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace('\n', ' ').strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ==========================================
# AGENT LOGIC WITH BACKOFF
# ==========================================
def get_model_message(client: OpenAI, obs_str: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current Observation:\n{obs_str}\n\nNext action (JSON format):"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            # Remove markdown codeblocks if the LLM adds them despite instructions
            raw_content = (completion.choices[0].message.content or "").strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            return raw_content.strip()
        
        except Exception as exc:
            # Handle rate limiting gracefully
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                wait_time = 4 * (attempt + 1)
                print(f"[DEBUG] API Rate Limited. Retrying in {wait_time}s...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"[DEBUG] Model request failed: {exc}", flush=True)
                return '{"command": {"action_type": "submit", "notes": "API Error"}}'
                
    print("[DEBUG] Max API retries reached.", flush=True)
    return '{"command": {"action_type": "submit", "notes": "Rate limit fallback"}}'

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main() -> None:
    # Hackathon requirement: Initialize via official OpenAI client using mandatory vars
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Allow evaluator to inject Docker container, fallback to local URL
    if IMAGE_NAME:
        print(f"[DEBUG] Booting environment from Docker image: {IMAGE_NAME}", flush=True)
        env = await DataJanitorEnv.from_docker_image(IMAGE_NAME)
    else:
        print(f"[DEBUG] Booting environment from local URL", flush=True)
        env = DataJanitorEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Mandatory START log
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_str = result.observation.model_dump_json(indent=2)
            
            # API Request
            agent_reply = await asyncio.to_thread(get_model_message, client, obs_str)
            error_msg = None
            
            # Baseline delay to prevent immediate rate limit trips
            await asyncio.sleep(2)
            
            try:
                action_dict = json.loads(agent_reply)
                action = DataJanitorAction(**action_dict)
                result = await env.step(action)
            except (json.JSONDecodeError, ValidationError) as e:
                error_msg = f"Agent Output Error: {str(e)}"
                print(f"[DEBUG] {error_msg}", flush=True)
                action = DataJanitorAction(command={"action_type": "submit", "notes": "Format error fallback"})
                result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            
            # Ensure action logs on a single line
            flat_action = json.dumps(action.model_dump())
            
            # Mandatory STEP log
            log_step(step=step, action=flat_action, reward=reward, done=done, error=error_msg)

        # Final Evaluation
        score = result.observation.final_score
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        # Mandatory END log
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())