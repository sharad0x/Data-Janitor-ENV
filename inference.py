import asyncio
import os
import json
import textwrap
import time
from typing import List, Optional

from openai import OpenAI
from pydantic import ValidationError

from dotenv import load_dotenv
load_dotenv()

from client import DataJanitorEnv
from models import DataJanitorAction

# ==========================================
# MANDATORY HACKATHON ENVIRONMENT VARIABLES
# ==========================================
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

SPACE_URL = os.getenv("SPACE_URL") 
BENCHMARK = os.getenv("BENCHMARK", "data_janitor")
MAX_STEPS = 40
SUCCESS_SCORE_THRESHOLD = 0.85 

# The 3 tasks the validator demands we loop through
TASKS_TO_RUN = ["easy", "medium", "hard"]

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
    9. CRITICAL ANTI-LOOPING RULE: If you see "FATAL PENALTY" in your feedback, it means you are looping. You MUST instantly change your strategy and target a different column. Do not touch that column again.
       
    You must output EXACTLY ONE JSON object matching one of these structures. Output raw JSON only. Do not include markdown formatting or ```json blocks:
    {"command": {"action_type": "drop_column", "column_name": "..."}}
    {"command": {"action_type": "fill_missing", "column_name": "...", "strategy": "median", "constant_value": "..."}} 
    {"command": {"action_type": "handle_outliers", "column_name": "...", "strategy": "clip_percentile", "lower_percentile": 0.05, "upper_percentile": 0.95}} 
    {"command": {"action_type": "transform_distribution", "column_name": "...", "strategy": "log1p"}} 
    {"command": {"action_type": "encode_categorical", "column_name": "...", "strategy": "one_hot"}} 
    {"command": {"action_type": "scale_feature", "column_name": "...", "strategy": "standard"}}
    {"command": {"action_type": "submit", "notes": "..."}}
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
            raw_content = (completion.choices[0].message.content or "").strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            return raw_content.strip()
        except Exception as exc:
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                time.sleep(4 * (attempt + 1))
            else:
                return '{"command": {"action_type": "submit", "notes": "API Error"}}'
    return '{"command": {"action_type": "submit", "notes": "Rate limit fallback"}}'

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if SPACE_URL:
        env = DataJanitorEnv(base_url=SPACE_URL)
    elif IMAGE_NAME:
        env = await DataJanitorEnv.from_docker_image(IMAGE_NAME)
    else:
        env = DataJanitorEnv(base_url="http://127.0.0.1:8000")

    for current_task in TASKS_TO_RUN:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await env.reset()
            
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_str = result.observation.model_dump_json(indent=2)
                agent_reply = await asyncio.to_thread(get_model_message, client, obs_str)
                
                await asyncio.sleep(2)
                
                error_msg = None
                try:
                    action_dict = json.loads(agent_reply)
                    action = DataJanitorAction(**action_dict)
                    result = await env.step(action)
                except (json.JSONDecodeError, ValidationError) as e:
                    error_msg = f"Agent Output Error: {str(e)}".replace('\n', ' ')
                    action = DataJanitorAction(command={"action_type": "submit", "notes": "Format error fallback"})
                    result = await env.step(action)

                done = result.done

                # HACKATHON PROTECTION: 
                # The environment sends intermediate rewards for the UI, 
                # but the validator expects 0.0 until the episode ends.
                display_reward = result.reward if done else 0.0

                rewards.append(display_reward)
                steps_taken = step
                
                flat_action = json.dumps(action.model_dump())
                log_step(step=step, action=flat_action, reward=display_reward, done=done, error=error_msg)

            score = result.observation.final_score
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            # DO NOT close env here! Just log the end of the current task.
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    # Close env only after all 3 tasks are complete
    try:
        await env.close()
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(main())