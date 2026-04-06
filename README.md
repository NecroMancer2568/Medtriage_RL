---
title: MedTriage RL
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---
## Note* A gradio UI is also available for this environment. You can access it at [https://mayank200062006-medtriage-openenv.hf.space/ui/]
# MedTriage-RL
Emergency Department Triage Reinforcement Learning Environment.
# MedTriage-RL

An OpenEnv-compatible reinforcement learning environment simulating 
emergency department triage. An AI agent acts as a triage nurse: 
observing patient presentations, asking targeted clinical questions, 
and assigning ESI urgency levels (1–5).

## Environment Description

### Why Medical Triage?/ Motivation behind MedTriage-RL

Emergency department triage is a high-stakes real-world task where 
errors cause patient harm. India's EDs face chronic understaffing 
and triage errors. This environment trains and evaluates agents on 
realistic clinical decision-making across three difficulty tiers, 
including deceptive presentations that challenge even frontier models.
The safety gate in Task 3 models real clinical risk — catastrophic 
undertriage scores zero regardless of other performance.

This environment tests three core capabilities:
1. **Pattern recognition** — identifying classic presentations
2. **Information gathering** — knowing which questions reveal critical information
3. **Safety awareness** — recognizing when benign-sounding complaints hide emergencies

### Three Difficulty Tiers

| Task | Difficulty | Description | Expected Score |
|------|------------|-------------|----------------|
| task_1 | Easy | Clear, unambiguous presentations | 0.75–0.90 |
| task_2 | Medium | Ambiguous cases requiring discriminating questions | 0.45–0.65 |
| task_3 | Hard | Deceptive presentations hiding critical conditions | 0.20–0.40 |

### Task 1 — Single-symptom triage (easy)
Clear unambiguous presentations. Tests basic ESI knowledge and 
efficiency. Agents should identify the correct urgency level from 
standard clinical signs within 2–3 questions.
Expected frontier model score: 0.75–0.90

### Task 2 — Ambiguous multi-symptom triage (medium)
Two-fork diagnoses where the correct ESI depends on one or two 
discriminating questions. Examples: thunderclap headache vs migraine, 
ectopic pregnancy vs appendicitis, DVT with PE vs simple leg swelling.
Agents must ask the right clarifying questions to disambiguate.
Expected frontier model score: 0.45–0.65

### Task 3 — Deceptive presentation triage (hard)
Life-threatening cases with benign-sounding chief complaints. 
Examples: DKA presenting as stomach bug, aortic dissection as back 
strain, female NSTEMI as anxiety. Safety gate active — assigning 
ESI 3, 4, or 5 to a true ESI 1 patient scores 0.0 regardless of 
other factors. Agents must probe beyond the chief complaint.
Expected frontier model score: 0.20–0.40

## Action space

| Action | Description |
|---|---|
| ASK_VITALS | Measure HR, BP, RR, SpO2, temperature |
| ASK_SYMPTOMS | Ask about associated symptoms |
| ASK_HISTORY | Past medical history, medications, allergies |
| ASK_PAIN | Pain severity, character, location, radiation |
| ASK_DURATION | Onset timing and progression |
| ASK_CONTEXT | Situational context, exposures, recent events |
| REQUEST_IMAGE | Clinical image description of affected area |
| TRIAGE_1 | Immediate — life-threatening |
| TRIAGE_2 | Emergent — seen within 15 minutes |
| TRIAGE_3 | Urgent — seen within 60 minutes |
| TRIAGE_4 | Less urgent — one resource, low risk |
| TRIAGE_5 | Non-urgent — no resources needed |

## Observation space

| Field | Type | Description |
|---|---|---|
| chief_complaint | string | Patient's presenting complaint |
| patient_meta | dict | Age and gender |
| revealed_info | dict | Accumulated answers to ASK_* actions |
| step | int | Current step (0–6) |
| max_steps | int | Maximum steps (6) |
| image_available | bool | Whether a clinical image exists |
| task_id | string | task_1, task_2, or task_3 |
| done | bool | True after TRIAGE_* or timeout |

## Reward function

MedTriage-RL uses a dense reward function that provides signal at 
every step of the episode, not just at termination. This ensures 
agents receive meaningful gradients throughout the trajectory rather 
than waiting for a sparse end-of-episode signal.

### Step-level rewards (information-gathering actions)

Every time the agent calls an ASK_* action or REQUEST_IMAGE, the 
following components are computed and summed:

#### Relevance reward
Measures whether the agent asked a clinically meaningful question 
for this specific patient case.

| Condition | Reward |
|---|---|
| Action is a discriminating question for this case | +0.30 |
| Action is not a discriminating question for this case | +0.00 |

Discriminating questions are case-specific — they are the questions 
that unlock the information needed to correctly resolve the clinical 
uncertainty. For example, in an ambiguous headache case (Task 2), 
ASK_PAIN unlocks the thunderclap character that distinguishes 
subarachnoid haemorrhage from migraine. In a deceptive DKA case 
(Task 3), ASK_CONTEXT reveals the missed insulin dose. Asking 
ASK_VITALS on a case where vitals are not discriminating scores 
+0.00 — not penalised, but not rewarded.

#### Redundancy penalty
Penalises the agent for repeating a question it has already asked 
in the current episode.

| Condition | Reward |
|---|---|
| Action has already been taken this episode | −0.10 |
| Action is new | +0.00 |

Information revealed by an ASK_* action is permanently visible in 
the observation from that step forward. Asking the same question 
twice wastes a step and receives this penalty.

#### Urgency penalty
Penalises the agent for continuing to ask questions when it is 
running out of steps — it should commit to a triage decision.

| Steps remaining | Reward |
|---|---|
| 3 or more steps remaining | +0.00 |
| 2 or fewer steps remaining | −0.15 |

This penalty fires on every information-gathering action taken 
when the agent has 2 or fewer steps left. It encourages the agent 
to triage efficiently rather than exhausting the step budget.

#### Combined step reward formula

### Terminal Reward

On `TRIAGE_*` action, the grader computes a score from 0.0 to 1.0 based on:
- **Accuracy** — how close to the true ESI level
- **Efficiency** — question count (task_1)
- **Relevance** — asked discriminating questions (task_2)
- **Safety gate** — hard zero for catastrophic undertriage (task_3)

### Timeout Penalty

If max_steps reached without a `TRIAGE_*` action: reward = -2.0, grader_score = 0.0

## Setup and usage

### Docker
docker build -t medtriage-rl .
docker run -p 7860:7860 medtriage-rl

### Local
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 7860

### API endpoints
POST /reset?task_id=task_1   — start new episode
POST /step?task_id=task_1    — execute action
GET  /state?task_id=task_1   — current observation
GET  /health                 — health check
GET  /docs                   — interactive API documentation

### Example
curl -X POST http://localhost:7860/reset?task_id=task_1
curl -X POST http://localhost:7860/step?task_id=task_1 \
     -H "Content-Type: application/json" \
     -d '{"action": "ASK_VITALS"}'

### inference.py
Requires environment variables:
  API_BASE_URL — LLM API endpoint
  MODEL_NAME   — model identifier  
  HF_TOKEN     — API key
  ENV_URL      — environment URL (default: http://localhost:7860)

python3 inference.py


## Baseline scores

Model: nvidia/nemotron-3-super-120b-a12b

| Task | Episodes | Mean | Target |
|---|---|---|---|
| Task 1 (easy) | [0.85, 0.43, 0.85] | 0.71 | 0.75–0.90 |
| Task 2 (medium) | [0.60, 0.60, 0.40] | 0.53 | 0.45–0.65 |
| Task 3 (hard) | [0.50, 0.30, 0.30] | 0.37 | 0.20–0.40 |
| Overall | | 0.54 | |

### Note on Task 3 scores
Scores of 0.30 reflect the lucky guess cap in Task3Grader. The model 
correctly identifies critical patients from vital sign abnormalities 
but does not ask the case-specific discriminating questions 
(ASK_CONTEXT, ASK_DURATION, REQUEST_IMAGE) needed to demonstrate 
understanding of the deceptive presentation. A score of 0.70 requires 
asking at least 2 discriminating questions before triaging. This cap 
is intentional — it distinguishes mechanically correct triage from 
clinically reasoned triage on life-threatening cases.

### Note on REQUEST_IMAGE
REQUEST_IMAGE returns structured clinical descriptions generated by 
medical domain experts. External image data in the action payload is 
accepted but not processed — the environment always returns the 
case-specific pre-defined description. This ensures deterministic 
reproducible grading across all evaluation runs.

## Project Structure

```
medtriage-env/
├── inference.py          # Baseline inference script
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container configuration
├── requirements.txt      # Pinned dependencies
├── src/
│   ├── server.py         # FastAPI endpoints
│   ├── env.py            # TriageEnv + Pydantic models
│   ├── graders.py        # Task graders (0.0-1.0)
│   ├── reward.py         # Dense reward function
│   ├── patient_sim.py    # Patient case loader
│   └── patients/         # 21 patient cases (7 per tier)
└── tests/
    ├── test_graders.py   # Grader determinism tests
    └── test_env.py       # Spec compliance tests
```
## NOTE* For evaluating the model Grader I used the eval.py script it mimics the behaviour of 4 different AGENTS, AGENT 1:- A random agent, AGENT 2:- ALWAYS ESI 3, AGENT 3:- ALWAYS ESI 1, AGENT 4:- Keyword Heuristic. This way i was able to remodel my graders reward logic to be more robust and aligned with the task requirements.
