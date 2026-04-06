
import os
import sys
import json
import requests
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

hf_token=os.getenv("HF_TOKEN")
# ── Environment variables ────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "nvidia/nemotron-3-super-120b-a12b")
HF_TOKEN     = os.environ.get("HF_TOKEN",     hf_token)
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None

# ── Valid actions ────────────────────────────────────────────────────────────
VALID_ACTIONS = {
    "ASK_VITALS", "ASK_SYMPTOMS", "ASK_HISTORY", "ASK_PAIN",
    "ASK_DURATION", "ASK_CONTEXT", "REQUEST_IMAGE",
    "TRIAGE_1", "TRIAGE_2", "TRIAGE_3", "TRIAGE_4", "TRIAGE_5",
}

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Emergency Department Triage Nurse. Your goal is to assign the correct ESI level (1–5) while maximizing clinical safety and information efficiency.

### OPERATIONAL RULES:
1. OUTPUT FORMAT: You must reason internally about the case, but the VERY LAST LINE of your response must be EXACTLY one action name from the VALID ACTIONS list.
2. VALID ACTIONS: [ASK_VITALS, ASK_SYMPTOMS, ASK_HISTORY, ASK_PAIN, ASK_DURATION, ASK_CONTEXT, REQUEST_IMAGE, TRIAGE_1, TRIAGE_2, TRIAGE_3, TRIAGE_4, TRIAGE_5]
3. EFFICIENCY: Do not repeat questions. If information is already in the "Gathered Info" section, move to the next logical inquiry or triage.
4. MANDATORY INQUIRY: Always ASK_VITALS if they are missing. In Task 2 and 3, you MUST ask at least 2 discriminating questions before triaging to pass the reasoning cap.

### CLINICAL STRATEGY BY TASK:
- TASK 1 (Clear Cases): Identify the core physiological threat and triage immediately once vitals are known.
- TASK 2 (Ambiguity): Use ASK_PAIN, ASK_DURATION, or ASK_CONTEXT to differentiate between similar presentations (e.g., Cardiac vs. Musculoskeletal).
- TASK 3 (Deceptive/Hard): ACTIVATE HIGH SUSPICION. If a patient has a "benign" complaint but high-risk factors (age, diabetes, history), you MUST use ASK_CONTEXT or REQUEST_IMAGE. 
  *CRITICAL:* Assigning TRIAGE 3, 4, or 5 to a life-threatening (ESI 1) case results in a score of 0.0. When in doubt, escalate to TRIAGE_1 or TRIAGE_2.

### DECEPTIVE PATTERN RECOGNITION:
- Elderly + Back Pain = Possible Aortic Dissection (Action: ASK_CONTEXT / ASK_VITALS)
- Diabetic + Stomach Bug = Possible DKA (Action: ASK_CONTEXT / ASK_VITALS)
- Female + Anxiety/Indigestion = Possible NSTEMI (Action: ASK_PAIN / ASK_VITALS)
- Leg Swelling + Oral Contraceptives = Possible DVT/PE (Action: ASK_DURATION)

Reason briefly, then output the action name on the final line."""


# ── Logging functions (mandatory stdout format) ──────────────────────────────
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=medtriage-rl model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatter ────────────────────────────────────────────────────
def format_observation(obs: dict) -> str:
    lines = [
        f"Patient: {obs['patient_meta']['age']}yo "
        f"{obs['patient_meta']['gender']}",
        f"Complaint: {obs['chief_complaint']}",
        f"Step: {obs['step']} of {obs['max_steps']}",
    ]
    if obs.get("image_available"):
        lines.append("Image: available")
    if obs.get("revealed_info"):
        lines.append("Information gathered:")
        for k, v in obs["revealed_info"].items():
            lines.append(f"  {k}: {v}")
        lines.append(
            "Do not repeat: " + ", ".join(obs["revealed_info"].keys())
        )
    else:
        lines.append("Information gathered: none yet")
    lines.append("\nAction:")
    return "\n".join(lines)


# ── Heuristic fallback agent ─────────────────────────────────────────────────
def get_heuristic_action(observation: dict) -> str:
    revealed = observation.get("revealed_info", {})
    step     = observation["step"]
    chief    = observation["chief_complaint"].lower()

    emergency_keywords = [
        "seizure", "chest pain", "crushing", "unresponsive",
        "not breathing", "unconscious", "cardiac arrest",
        "anaphylaxis", "bee sting", "throat tight",
        "active labour", "contractions",
    ]
    if any(kw in chief for kw in emergency_keywords):
        if "ASK_VITALS"   not in revealed: return "ASK_VITALS"
        if "ASK_SYMPTOMS" not in revealed: return "ASK_SYMPTOMS"
        return "TRIAGE_1"

    if "ASK_VITALS"   not in revealed: return "ASK_VITALS"
    if "ASK_SYMPTOMS" not in revealed and step < 3: return "ASK_SYMPTOMS"
    if "ASK_HISTORY"  not in revealed and step < 4: return "ASK_HISTORY"

    vitals = revealed.get("ASK_VITALS", "").lower()
    critical = ["bp 8","bp 9","spo2 9","spo2 8","hr 11","hr 12","hr 13",
                "hypotensive","unresponsive","seizing","kussmaul",
                "diaphoretic","clammy","pale"]
    if any(s in vitals for s in critical): return "TRIAGE_1"

    urgent = ["tachycardic","febrile","39.","38.","hr 10"]
    if any(s in vitals for s in urgent): return "TRIAGE_2"

    minor = ["sprain","twisted","minor cut","laceration",
             "sore throat","runny"]
    if any(kw in chief for kw in minor): return "TRIAGE_4"

    return "TRIAGE_3"

HAS_WARNED_FALLBACK=False
# ── LLM action selector ──────────────────────────────────────────────────────
def get_llm_action(observation: dict, history: List[str] = None) -> str:
    if client is None:
        if not HAS_WARNED_FALLBACK:
            print("\n[WARNING] HF_TOKEN not found. Switching to HEURISTIC Model.",file=sys.stderr)
            HAS_WARNED_FALLBACK=True
        return get_heuristic_action(observation)

    obs_text = format_observation(observation)
    if history:
        obs_text += "\nActions already taken: " + ", ".join(history)
        obs_text += "\nDo not repeat these."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": obs_text},
            ],
            max_tokens=512,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip().upper()

        # Layer 1: exact match
        if raw in VALID_ACTIONS:
            return raw

        # Layer 2: last valid action mentioned (handles reasoning models)
        found = []
        for action in VALID_ACTIONS:
            idx = raw.rfind(action)
            if idx != -1:
                found.append((idx, action))
        if found:
            return sorted(found, reverse=True)[0][1]

        # Layer 3: heuristic fallback
        return get_heuristic_action(observation)

    except Exception:
        if not HAS_WARNED_FALLBACK:
            print(f"\n[WARNING] API Error: {e}. Switching to HEURISTIC model.", file=sys.stderr)
            HAS_WARNED_FALLBACK=True
        return get_heuristic_action(observation)


# ── Single episode runner ────────────────────────────────────────────────────
def run_episode(task_id: str, env_url: str) -> dict:
    """
    Run one episode. Returns dict with score, steps, rewards, success.
    Emits [START], [STEP]*, [END] to stdout.
    """
    log_start(task=task_id, model=MODEL_NAME)

    rewards     : List[float] = []
    steps_taken : int         = 0
    score       : float       = 0.0
    success     : bool        = False
    history     : List[str]   = []

    try:
        # Reset
        reset_resp = requests.post(
            f"{env_url}/reset",
            params={"task_id": task_id},
            timeout=30,
        )
        reset_resp.raise_for_status()
        observation = reset_resp.json()

        done = False

        while not done:
            action = get_llm_action(observation, history)
            history.append(action)

            try:
                step_resp = requests.post(
                    f"{env_url}/step",
                    params={"task_id": task_id},
                    json={"action": action},
                    timeout=30,
                )
                step_resp.raise_for_status()
                result = step_resp.json()
            except Exception as e:
                log_step(
                    step=steps_taken + 1,
                    action=action,
                    reward=0.0,
                    done=True,
                    error=str(e),
                )
                break

            reward      = result["reward"].get("total", 0.0)
            done        = result["done"]
            observation = result["observation"]
            steps_taken += 1
            rewards.append(reward)

            error = None
            if done:
                grader_score = result["reward"].get("grader_score")
                if grader_score is not None:
                    score   = float(grader_score)
                    success = score > 0.0

            log_step(
                step=steps_taken,
                action=action,
                reward=reward,
                done=done,
                error=error,
            )

    except Exception as e:
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return {"score": 0.0, "steps": steps_taken,
                "rewards": rewards, "success": False}

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "score":   score,
        "steps":   steps_taken,
        "rewards": rewards,
        "success": success,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_results = {}

    for task_id in ["task_1", "task_2", "task_3"]:
        task_scores = []
        for episode in range(3):
            result = run_episode(task_id, ENV_URL)
            task_scores.append(result["score"])

        mean = round(sum(task_scores) / len(task_scores), 4)
        all_results[task_id] = {"scores": task_scores, "mean": mean}

    # Summary table to stderr so it doesn't interfere with stdout parsing
    print("\n" + "=" * 60, file=sys.stderr)
    print("RESULTS SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for task_id, data in all_results.items():
        scores_str = ", ".join(f"{s:.4f}" for s in data["scores"])
        print(f"{task_id}: [{scores_str}]  mean={data['mean']:.4f}",
              file=sys.stderr)
    overall = sum(d["mean"] for d in all_results.values()) / len(all_results)
    print(f"Overall mean: {overall:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_passed = all(d["mean"] > 0.0 for d in all_results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()