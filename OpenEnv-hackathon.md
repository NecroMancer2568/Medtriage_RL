# MedTriage-RL — Claude Code Implementation Plan
## Complete context for building the OpenEnv medical triage environment

---

## 1. PROJECT OVERVIEW

### What you are building
A complete OpenEnv-compatible reinforcement learning environment called **MedTriage-RL**. This is a medical triage simulation where an AI agent plays the role of an emergency department triage nurse. The agent observes patient presentations, asks follow-up questions, and assigns an ESI (Emergency Severity Index) urgency level from 1 (Immediate) to 5 (Non-urgent).

### Why it exists
Built for the Meta × PyTorch OpenEnv AI Hackathon (Scaler School of Technology, Bangalore, April 25–26 2026). The environment must be deployed to Hugging Face Spaces and evaluated by judges who run a standard LLM (e.g. Nemotron 3 Super) against it via a baseline inference script.

### What judges evaluate
| Criterion | Weight | What it means |
|---|---|---|
| Real-world utility | 30% | Models something humans genuinely do |
| Task & grader quality | 25% | 3 tasks, deterministic graders, 0.0–1.0 scores |
| Environment design | 20% | Dense reward (not sparse), clean state management |
| Code quality & spec | 15% | OpenEnv spec compliance, Docker works, clean code |
| Creativity & novelty | 10% | Novel domain not seen in OpenEnv catalog before |

### Pass/fail gate — disqualified if any fail
- HF Space deploys and responds to `reset()` with HTTP 200
- `openenv validate` passes
- `docker build && docker run` works cleanly
- `inference.py` runs without error and produces reproducible scores
- 3+ tasks with graders that return scores in 0.0–1.0

---

## 2. HARDWARE & CONSTRAINTS

- Developer machine: MacBook M2 Air, 8GB RAM — CPU only, no GPU
- Unsloth does NOT support Apple Silicon MPS for training yet
- No model training required — judges bring their own LLM
- Inference script must run on 2 vCPU / 8GB RAM in under 20 minutes
- All dependencies must be pinned in requirements.txt

---

## 3. EXACT PROJECT FILE STRUCTURE

```
medtriage-env/                    ← root, this is the HF Space repo
├── inference.py                  ← REQUIRED: exactly this name, in root
├── openenv.yaml                  ← OpenEnv metadata, tags: [openenv]
├── Dockerfile                    ← must build and run cleanly
├── README.md                     ← env description, spaces, tasks, scores
├── requirements.txt              ← all pinned versions
├── src/
│   ├── server.py                 ← FastAPI app, /reset /step /state
│   ├── env.py                    ← TriageEnv class + ALL Pydantic models
│   ├── graders.py                ← Task1Grader, Task2Grader, Task3Grader
│   ├── reward.py                 ← dense reward function (step-level signal)
│   ├── patient_sim.py            ← patient loader, hidden state manager
│   └── patients/
│       ├── easy_cases.json       ← 7 easy patient cases (ALREADY WRITTEN)
│       ├── medium_cases.json     ← 7 medium patient cases (ALREADY WRITTEN)
│       └── hard_cases.json       ← 7 hard patient cases (ALREADY WRITTEN)
└── tests/
    ├── test_graders.py           ← proves graders deterministic + 0.0–1.0
    └── test_env.py               ← validates spec compliance
```

---

## 4. PATIENT DATA — ALREADY WRITTEN, DO NOT REGENERATE

The patient case JSON files are complete and validated. **Do not modify their structure.** Every case has this exact schema:

```json
{
  "id": "hard_001",
  "difficulty": "easy|medium|hard",
  "chief_complaint": "string — what the patient says walking in",
  "patient_meta": { "age": 19, "gender": "male" },
  "hidden_diagnosis": "string — true diagnosis, never shown to agent",
  "true_esi": 1,
  "image_available": true,
  "image_description": "string | null — clinical text returned on REQUEST_IMAGE",
  "red_flags": ["list of clinical red flags present"],
  "discriminating_questions": ["ASK_HISTORY", "ASK_SYMPTOMS"],
  "deceptive_element": "string — only in hard cases, explains the trap",
  "fork_condition": "string — only in medium cases, explains the decision fork",
  "answer_map": {
    "ASK_VITALS": "string — full vitals text revealed when agent asks",
    "ASK_SYMPTOMS": "string",
    "ASK_HISTORY": "string",
    "ASK_PAIN": "string",
    "ASK_DURATION": "string",
    "ASK_CONTEXT": "string",
    "REQUEST_IMAGE": "string | null — same as image_description"
  },
  "grader_notes": "string — explains scoring intent for this case"
}
```

**Case counts:** 7 easy, 7 medium, 7 hard = 21 total
**ESI distribution easy:** 1, 1, 1, 2, 4, 4, 5
**ESI distribution medium:** all 1 and 2
**ESI distribution hard:** all ESI 1 (deceptive presentations)
**Image cases:** 3 easy, 4 medium, 2 hard

---

## 5. THE 12 VALID ACTIONS

```python
VALID_ACTIONS = Literal[
    "ASK_VITALS",      # reveals HR, BP, RR, SpO2, Temp
    "ASK_SYMPTOMS",    # reveals associated symptoms
    "ASK_HISTORY",     # reveals past medical history, medications
    "ASK_PAIN",        # reveals pain score, quality, radiation
    "ASK_DURATION",    # reveals onset timing and progression
    "ASK_CONTEXT",     # reveals situational context, exposures
    "REQUEST_IMAGE",   # returns clinical image description (text)
    "TRIAGE_1",        # terminal: assign ESI 1 — Immediate
    "TRIAGE_2",        # terminal: assign ESI 2 — Emergent
    "TRIAGE_3",        # terminal: assign ESI 3 — Urgent
    "TRIAGE_4",        # terminal: assign ESI 4 — Less urgent
    "TRIAGE_5",        # terminal: assign ESI 5 — Non-urgent
]
```

**Rules:**
- `ASK_*` and `REQUEST_IMAGE` are information-gathering — they consume one step and return data
- `TRIAGE_*` are terminal — they end the episode immediately
- Max steps per episode: 6 (configurable)
- If max steps reached without TRIAGE action: episode ends, reward = -2.0

---

## 6. PYDANTIC MODELS — FULL SPECIFICATION

### 6.1 TriageAction

```python
class TriageAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: VALID_ACTIONS = Field(
        description="One of 12 valid action strings. TRIAGE_* ends the episode."
    )
    image_data: Optional[str] = Field(
        default=None,
        description=(
            "Reserved for future multimodal support. "
            "Accepted but NOT processed in v1. "
            "Environment always returns the case-specific "
            "pre-defined clinical description regardless of this field."
        )
    )
```

**Key design decisions:**
- `extra="forbid"` — unknown fields return 422, not silently ignored
- `image_data` is accepted but always ignored — prevents crashes if judge tests endpoint directly with image payloads
- This is documented explicitly so judges understand the contract

### 6.2 TriageObservation

```python
class TriageObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str = Field(
        description="Patient's presenting complaint. Always visible from step 0."
    )
    patient_meta: dict = Field(
        description="Age and gender. Always visible from step 0."
    )
    revealed_info: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Accumulates across steps. Keys are action names "
            "(e.g. 'ASK_VITALS'), values are the revealed text. "
            "Information once revealed stays visible forever."
        )
    )
    step: int = Field(
        description="Current step count. Starts at 0, max 6."
    )
    max_steps: int = Field(
        default=6,
        description="Maximum steps before forced episode termination."
    )
    image_available: bool = Field(
        description="Whether this case has an image that can be requested."
    )
    task_id: str = Field(
        description="Which task this episode belongs to: task_1, task_2, or task_3."
    )
    done: bool = Field(
        default=False,
        description="True after a TRIAGE_* action or step exhaustion."
    )
```

**Key design decision — revealed_info is additive:**
Once information is revealed it stays in the observation forever. This makes episodes clean, reproducible, and prevents the agent from being confused about what it already knows.

### 6.3 TriageReward

```python
class TriageReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float = Field(
        description="Sum of all components. This is what RL algorithms use."
    )
    components: dict[str, float] = Field(
        description=(
            "Breakdown of reward by component. "
            "Keys: relevance, redundancy, safety, accuracy, efficiency. "
            "Enables interpretable reward analysis."
        )
    )
    is_terminal: bool = Field(
        description="True if this reward is for a TRIAGE_* action."
    )
    grader_score: Optional[float] = Field(
        default=None,
        description=(
            "Final grader score 0.0-1.0. "
            "Only populated on terminal step (TRIAGE_* action). "
            "This is what judges read for scoring."
        )
    )
    explanation: str = Field(
        description="Human-readable explanation of this step's reward."
    )
```

### 6.4 StepResult (returned by /step endpoint)

```python
class StepResult(BaseModel):
    observation: TriageObservation
    reward: TriageReward
    done: bool
    info: dict[str, Any] = Field(
        description=(
            "Always included. Contains: action_received, "
            "image_data_received (bool), image_processing status, "
            "episode_id, case_id, current_task."
        )
    )
```

**The info dict must always contain:**
```python
info = {
    "action_received": action.action,
    "image_data_received": action.image_data is not None,
    "image_processing": (
        "standardised_clinical_description_returned"
        if action.action == "REQUEST_IMAGE" and patient["image_available"]
        else "image_not_available_for_this_case"
        if action.action == "REQUEST_IMAGE"
        else "not_applicable"
    ),
    "episode_id": str(uuid),
    "case_id": patient["id"],
    "current_task": task_id,
    "step": current_step,
    "note": (
        "REQUEST_IMAGE returns pre-defined clinical descriptions. "
        "External image_data payloads are accepted but not processed. "
        "This ensures deterministic reproducible grading."
        if action.action == "REQUEST_IMAGE" else ""
    )
}
```

---

## 7. TRIAGE ENV CLASS — FULL SPECIFICATION

```python
class TriageEnv:
    def __init__(self, task_id: Literal["task_1", "task_2", "task_3"]):
        # task_1 = easy cases, task_2 = medium, task_3 = hard
        self.task_id = task_id
        self.cases = self._load_cases(task_id)
        self._current_patient = None
        self._revealed_info = {}
        self._step_count = 0
        self._episode_id = None
        self._actions_taken = []
        self._done = False
        self.max_steps = 6

    def reset(self) -> TriageObservation:
        # Pick random patient from correct difficulty tier
        # Reset ALL hidden state — critical for reproducibility
        # Return observation with only chief_complaint + patient_meta visible

    def step(self, action: TriageAction) -> StepResult:
        # 1. Validate not already done
        # 2. If ASK_* or REQUEST_IMAGE: reveal info, compute step reward
        # 3. If TRIAGE_*: compute grader score, set done=True
        # 4. If step >= max_steps and not done: force terminate, penalty reward
        # 5. Always return StepResult with full info dict

    def state(self) -> TriageObservation:
        # Return current observation without advancing anything
        # Pure read — no side effects
```

**Hidden state the agent never sees:**
- `patient["hidden_diagnosis"]` — true diagnosis
- `patient["true_esi"]` — correct triage level
- `patient["red_flags"]` — list of present red flags
- `patient["answer_map"]` — full answer lookup table

**State the agent accumulates:**
- `revealed_info` — grows with each ASK_* call

---

## 8. THE THREE GRADERS — EXACT SCORING FORMULAS

### Task 1 Grader (easy cases)

```python
def grade(patient, actions_taken, assigned_esi) -> float:
    # Step 1: accuracy
    esi_diff = abs(assigned_esi - patient["true_esi"])
    if esi_diff == 0:   accuracy = 1.0
    elif esi_diff == 1: accuracy = 0.5
    else:               accuracy = 0.0

    # Step 2: efficiency bonus
    ask_count = sum(1 for a in actions_taken if a.startswith("ASK_") or a == "REQUEST_IMAGE")
    if ask_count <= 2:  efficiency = 1.0
    elif ask_count <= 4: efficiency = 0.85
    else:               efficiency = 0.7

    return round(accuracy * efficiency, 4)
```

**Expected frontier model score: 0.75–0.90**

### Task 2 Grader (medium cases)

```python
def grade(patient, actions_taken, assigned_esi) -> float:
    # Step 1: accuracy
    esi_diff = abs(assigned_esi - patient["true_esi"])
    if esi_diff == 0:   accuracy = 1.0
    elif esi_diff == 1: accuracy = 0.5
    else:               accuracy = 0.0

    # Step 2: question relevance — did agent ask discriminating questions?
    asked = set(actions_taken)
    disc_qs = set(patient["discriminating_questions"])
    relevance = len(asked & disc_qs) / len(disc_qs) if disc_qs else 1.0

    # Step 3: redundancy penalty — repeated actions
    redundant = len(actions_taken) - len(set(actions_taken))
    redundancy_penalty = min(redundant * 0.1, 0.3)

    # Step 4: red flag penalty — missed critical question when red flag present
    red_flag_penalty = 0.0
    if patient["red_flags"] and disc_qs.isdisjoint(asked):
        red_flag_penalty = 0.3  # missed all discriminating questions

    score = (accuracy * relevance) - redundancy_penalty - red_flag_penalty
    return round(max(0.0, min(1.0, score)), 4)
```

**Expected frontier model score: 0.45–0.65**

### Task 3 Grader (hard/deceptive cases)

```python
def grade(patient, actions_taken, assigned_esi) -> float:
    # SAFETY GATE — hard zero for critical undertriage
    true_esi = patient["true_esi"]
    # All hard cases are ESI 1 — assigning ESI 4 or 5 = dangerous miss
    if true_esi <= 2 and assigned_esi >= 4:
        return 0.0  # safety gate: catastrophic undertriage

    # Step 1: accuracy
    esi_diff = abs(assigned_esi - true_esi)
    if esi_diff == 0:   accuracy = 1.0
    elif esi_diff == 1: accuracy = 0.5
    else:               accuracy = 0.0

    # Step 2: depth bonus — did agent probe beyond the deceptive surface?
    asked = set(actions_taken)
    disc_qs = set(patient["discriminating_questions"])
    probed_correctly = bool(asked & disc_qs)
    depth_bonus = 0.2 if probed_correctly else 0.0

    # Step 3: if correct ESI but never probed — cap at 0.5 (lucky guess)
    if accuracy == 1.0 and not probed_correctly:
        return round(0.5, 4)

    score = accuracy + depth_bonus
    return round(max(0.0, min(1.0, score)), 4)
```

**Expected frontier model score: 0.20–0.40**
**Key property: all hard cases are ESI 1. Any ESI 4/5 assignment = automatic 0.0**

---

## 9. DENSE REWARD FUNCTION — STEP-BY-STEP SIGNAL

This is the 20% environment design criterion. Reward must provide signal at EVERY step, not just at episode end.

```python
def compute_step_reward(patient, action, revealed_info, step, max_steps) -> TriageReward:

    components = {}

    if action.action.startswith("TRIAGE_"):
        # Terminal action — reward is grader score
        assigned_esi = int(action.action[-1])
        grader_score = run_grader(patient, history, assigned_esi)
        components["accuracy"] = grader_score
        total = grader_score
        explanation = f"Episode ended. Grader score: {grader_score}"

    elif action.action in ("ASK_VITALS","ASK_SYMPTOMS","ASK_HISTORY",
                           "ASK_PAIN","ASK_DURATION","ASK_CONTEXT","REQUEST_IMAGE"):
        # Information-gathering action
        disc_qs = set(patient["discriminating_questions"])

        # Relevance: was this a discriminating question?
        relevance = 0.3 if action.action in disc_qs else 0.0
        components["relevance"] = relevance

        # Redundancy: already asked this?
        redundancy = -0.1 if action.action in revealed_info else 0.0
        components["redundancy"] = redundancy

        # Urgency penalty: wasting steps near the end
        steps_left = max_steps - step
        urgency_penalty = -0.15 if steps_left <= 2 else 0.0
        components["urgency"] = urgency_penalty

        total = relevance + redundancy + urgency_penalty
        explanation = (
            f"Asked {action.action}. "
            f"{'Discriminating question — good signal.' if relevance > 0 else 'Non-discriminating question.'}"
            f"{' (Already asked — redundant.)' if redundancy < 0 else ''}"
            f"{' (Warning: only {steps_left} steps remaining.)' if urgency_penalty < 0 else ''}"
        )
        grader_score = None

    elif step >= max_steps:
        # Timeout — no triage action taken
        components["timeout"] = -2.0
        total = -2.0
        explanation = "Max steps reached without triage decision."
        grader_score = 0.0

    return TriageReward(
        total=round(total, 4),
        components=components,
        is_terminal=action.action.startswith("TRIAGE_"),
        grader_score=grader_score,
        explanation=explanation
    )
```

---

## 10. SERVER — FastAPI ENDPOINTS

Keep this file thin. All logic lives in `env.py`, `graders.py`, `reward.py`.

```python
# server.py — ~60 lines total

app = FastAPI(title="MedTriage-RL", version="1.0.0")

# Three env instances, one per task
envs = {
    "task_1": TriageEnv("task_1"),
    "task_2": TriageEnv("task_2"),
    "task_3": TriageEnv("task_3"),
}

@app.post("/reset")
def reset(task_id: str = "task_1") -> TriageObservation:
    # Validate task_id, call env.reset(), return observation

@app.post("/step")
def step(action: TriageAction, task_id: str = "task_1") -> StepResult:
    # Call env.step(action), return StepResult

@app.get("/state")
def state(task_id: str = "task_1") -> TriageObservation:
    # Call env.state(), return current observation — no side effects

@app.get("/health")
def health():
    return {"status": "ok", "tasks": ["task_1", "task_2", "task_3"]}

@app.get("/tasks")
def list_tasks():
    # Return task descriptions, difficulty levels, expected scores
```

**Important:** The server maintains one env instance per task. Each `/reset` call picks a new random patient from that task's case pool.

---

## 11. OPENENV.YAML — EXACT STRUCTURE

```yaml
name: medtriage-rl
version: "1.0.0"
description: >
  Medical emergency department triage simulation.
  An AI agent acts as a triage nurse: observing patient
  presentations, asking targeted clinical questions, and
  assigning ESI urgency levels (1-5). Three difficulty tiers
  test basic triage, ambiguous presentations, and deceptive
  cases that superficially appear benign.

tags:
  - openenv
  - healthcare
  - medical
  - triage
  - reinforcement-learning

tasks:
  - id: task_1
    name: "Single-symptom triage"
    difficulty: easy
    description: "Clear unambiguous presentations. Test basic ESI knowledge."
    expected_score_range: [0.75, 0.90]

  - id: task_2
    name: "Ambiguous multi-symptom triage"
    difficulty: medium
    description: "Two-fork diagnoses. Agent must ask discriminating questions."
    expected_score_range: [0.45, 0.65]

  - id: task_3
    name: "Deceptive presentation triage"
    difficulty: hard
    description: "Critical cases with benign-sounding complaints. Safety gate active."
    expected_score_range: [0.20, 0.40]

action_space:
  type: discrete
  actions:
    - ASK_VITALS
    - ASK_SYMPTOMS
    - ASK_HISTORY
    - ASK_PAIN
    - ASK_DURATION
    - ASK_CONTEXT
    - REQUEST_IMAGE
    - TRIAGE_1
    - TRIAGE_2
    - TRIAGE_3
    - TRIAGE_4
    - TRIAGE_5

observation_space:
  type: dict
  fields:
    chief_complaint: string
    patient_meta: object
    revealed_info: object
    step: integer
    max_steps: integer
    image_available: boolean
    task_id: string
    done: boolean

reward:
  type: dense
  range: [-2.0, 1.2]
  terminal: grader_score_0_to_1
  partial: step_level_signal

endpoints:
  reset: POST /reset
  step: POST /step
  state: GET /state
  health: GET /health
```

---

## 12. DOCKERFILE — EXACT REQUIREMENTS

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

**HuggingFace Spaces requires port 7860.**
**requirements.txt must pin ALL versions — example:**
```
fastapi==0.111.0
uvicorn==0.30.0
pydantic==2.7.1
```

---

## 13. INFERENCE.PY — EXACT SPECIFICATION

This is a pass/fail gate item. Must be named exactly `inference.py` in root.

```python
"""
inference.py — MedTriage-RL baseline inference script.
Uses OpenAI client against the deployed environment.
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Produces reproducible scores on all 3 tasks.
Must complete in under 20 minutes on 2 vCPU / 8GB RAM.
"""

import os
from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an emergency department triage nurse.
You will receive a patient observation. Choose exactly ONE action from:
ASK_VITALS, ASK_SYMPTOMS, ASK_HISTORY, ASK_PAIN, ASK_DURATION,
ASK_CONTEXT, REQUEST_IMAGE, TRIAGE_1, TRIAGE_2, TRIAGE_3, TRIAGE_4, TRIAGE_5

TRIAGE_1 = Immediate (life-threatening)
TRIAGE_2 = Emergent (high risk)
TRIAGE_3 = Urgent (stable but needs care)
TRIAGE_4 = Less urgent (one resource needed)
TRIAGE_5 = Non-urgent (can wait)

Respond with ONLY the action name. Nothing else."""

def run_episode(task_id: str, env_url: str) -> float:
    # POST /reset, run agent loop until done, return grader_score

def main():
    results = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        scores = [run_episode(task_id, ENV_URL) for _ in range(3)]
        results[task_id] = {
            "scores": scores,
            "mean": round(sum(scores)/len(scores), 4)
        }
    # Print results table
    # Exit 0 on success, 1 on any failure

if __name__ == "__main__":
    main()
```

**Run 3 episodes per task (9 total). Must finish in under 20 minutes.**

---

## 14. SOLVED DESIGN DECISIONS — DO NOT REVISIT

These were explicitly debated and resolved. Do not change them.

### REQUEST_IMAGE design
**Decision:** `REQUEST_IMAGE` returns a text string (structured clinical description), never a real image.
**Reason:** Ensures deterministic reproducible grading (required by spec). Judges cannot inject external images through the standard evaluation pipeline. The `image_data` field on `TriageAction` is accepted but always ignored.
**Implementation:** When `REQUEST_IMAGE` is called, look up `patient["answer_map"]["REQUEST_IMAGE"]`. If `image_available` is false, return `"No visual evidence available for this case."` The `info` dict always explains this behavior.

### Extra fields on TriageAction
**Decision:** `model_config = ConfigDict(extra="forbid")`
**Reason:** If a judge manually tests `/step` with an unknown field, they get a clear 422 validation error rather than silent failure. Documents the API contract explicitly.

### No model training
**Decision:** The environment does not include any ML training code.
**Reason:** The hackathon evaluates the environment, not trained agents. Judges bring their own LLM. No GPU needed.

### Patient data as static JSON
**Decision:** 21 hand-crafted patient cases in JSON files. No external datasets.
**Reason:** No access credentials needed, works offline, deterministic, fast to load, fully controlled quality.

### Dense reward not sparse
**Decision:** Reward signal at every step via `reward.py`, not just at episode end.
**Reason:** Explicitly required by the 20% "environment design" scoring criterion. The spec says "rewards partial progress toward task completion."

### Revealed info is additive and permanent
**Decision:** Once `ASK_VITALS` is called, vitals appear in every subsequent observation.
**Reason:** Makes episodes reproducible and prevents agent confusion. Clean state management.

### Three separate env instances (one per task)
**Decision:** `envs = {"task_1": TriageEnv("task_1"), ...}`
**Reason:** Each task has its own random state and case pool. Calling `/reset?task_id=task_2` doesn't affect task_1's state.

---

## 15. TESTS — WHAT MUST PASS

### test_graders.py
For each grader, test these exact three scenarios:

**Task 1:**
- Correct ESI, 2 questions asked → score ≈ 1.0
- Off by 1 ESI, 3 questions → score ≈ 0.425
- Off by 2+ ESI → score = 0.0

**Task 2:**
- Correct ESI + asked all discriminating questions → score ≈ 0.9
- Correct ESI + skipped discriminating questions → score ≈ 0.5
- Wrong ESI → score ≈ 0.0

**Task 3:**
- ESI 1 assigned + probed correctly → score ≈ 1.0
- ESI 1 assigned + no probing (lucky guess) → score = 0.5
- ESI 4 assigned on a critical (true ESI 1) case → score = 0.0 (safety gate)

### test_env.py
- `reset()` returns valid `TriageObservation` with step=0
- `state()` after reset equals `reset()` output
- `step()` with invalid action raises 422
- `step()` with `TRIAGE_1` sets `done=True`
- `step()` with `REQUEST_IMAGE` on image_available=False returns safe string
- `step()` with `image_data` payload does not crash
- `revealed_info` grows correctly across steps
- After `done=True`, further `step()` calls raise appropriate error

---

## 16. README — REQUIRED SECTIONS

Must include all of these or judges dock points:

1. Environment description and motivation (why medical triage, why India context)
2. Action space — list all 12 actions with descriptions
3. Observation space — list all fields with types
4. Task descriptions — all 3 with difficulty and what makes each hard
5. Reward function — explain dense shaping
6. Setup and usage — `docker build`, `docker run`, curl examples
7. Baseline scores — the actual numbers from running `inference.py`
8. Note on REQUEST_IMAGE design decision

---

## 17. BUILD ORDER — DO NOT DEVIATE

Build in this exact order. Each layer depends on the one before.

```
1. src/patients/*.json     — ALREADY DONE, do not touch
2. src/env.py              — Pydantic models first, then TriageEnv class
3. src/graders.py          — Task1Grader, Task2Grader, Task3Grader
4. src/reward.py           — dense step-level reward function
5. src/patient_sim.py      — patient loader, case selection by difficulty
6. src/server.py           — FastAPI wrapper, thin layer only
7. openenv.yaml            — metadata file
8. Dockerfile              — containerisation
9. tests/                  — write and run tests
10. inference.py           — baseline script, last before README
11. README.md              — write after you have real baseline scores
```

**Critical checkpoint at step 8:** Run `docker build && docker run`, then
`curl http://localhost:7860/reset?task_id=task_1`
Must return valid JSON before proceeding to inference.py.

---

## 18. QUICK REFERENCE — KEY NUMBERS

| Parameter | Value |
|---|---|
| Max steps per episode | 6 |
| Number of cases per tier | 7 |
| Total patient cases | 21 |
| Valid action count | 12 |
| Reward range | -2.0 to 1.2 |
| Grader score range | 0.0 to 1.0 |
| HF Spaces port | 7860 |
| Max inference.py runtime | 20 minutes |
| Episodes per task in baseline | 3 |
| Inference machine spec | 2 vCPU, 8GB RAM |

