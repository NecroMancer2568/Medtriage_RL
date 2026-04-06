"""
server.py — FastAPI server for MedTriage-RL environment.
Thin wrapper around TriageEnv. All logic lives in env.py, graders.py, reward.py.
"""

from fastapi import FastAPI, HTTPException, Query

from src.env import TriageAction, TriageEnv, TriageObservation, StepResult


app = FastAPI(
    title="MedTriage-RL",
    version="1.0.0",
    description=(
        "Medical emergency department triage simulation. "
        "An AI agent acts as a triage nurse: observing patient presentations, "
        "asking targeted clinical questions, and assigning ESI urgency levels (1-5)."
    ),
)

# Three env instances, one per task
envs = {
    "task_1": TriageEnv("task_1"),
    "task_2": TriageEnv("task_2"),
    "task_3": TriageEnv("task_3"),
}

@app.get("/")
def root():
    return {
        "name": "MedTriage-RL",
        "version": "1.0.0",
        "description": "Medical triage RL environment. Use /reset, /step, /state endpoints.",
        "docs": "/docs",
        "tasks": ["task_1", "task_2", "task_3"],
        "status": "ready"
    }
@app.post("/reset", response_model=TriageObservation)
def reset(task_id: str = Query(default="task_1", description="Task ID: task_1, task_2, or task_3")):
    """
    Reset the environment and start a new episode.
    Returns the initial observation with chief_complaint and patient_meta visible.
    """
    if task_id not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id: {task_id}. Must be one of: task_1, task_2, task_3"
        )
    
    return envs[task_id].reset()


@app.post("/step", response_model=StepResult)
def step(
    action: TriageAction,
    task_id: str = Query(default="task_1", description="Task ID: task_1, task_2, or task_3")
):
    """
    Execute an action in the environment.
    
    Actions:
    - ASK_VITALS, ASK_SYMPTOMS, ASK_HISTORY, ASK_PAIN, ASK_DURATION, ASK_CONTEXT, REQUEST_IMAGE: 
      Information-gathering actions that reveal patient data.
    - TRIAGE_1 to TRIAGE_5: Terminal actions that end the episode with an ESI assignment.
    """
    if task_id not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id: {task_id}. Must be one of: task_1, task_2, task_3"
        )
    
    env = envs[task_id]
    
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=TriageObservation)
def state(task_id: str = Query(default="task_1", description="Task ID: task_1, task_2, or task_3")):
    """
    Get the current observation without advancing the environment.
    Pure read — no side effects.
    """
    if task_id not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id: {task_id}. Must be one of: task_1, task_2, task_3"
        )
    
    env = envs[task_id]
    
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "tasks": ["task_1", "task_2", "task_3"]}


@app.get("/tasks")
def list_tasks():
    """List available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "task_1",
                "name": "Single-symptom triage",
                "difficulty": "easy",
                "description": "Clear unambiguous presentations. Test basic ESI knowledge.",
                "expected_score_range": [0.75, 0.90],
                "case_count": 7,
            },
            {
                "id": "task_2",
                "name": "Ambiguous multi-symptom triage",
                "difficulty": "medium",
                "description": "Two-fork diagnoses. Agent must ask discriminating questions.",
                "expected_score_range": [0.45, 0.65],
                "case_count": 7,
            },
            {
                "id": "task_3",
                "name": "Deceptive presentation triage",
                "difficulty": "hard",
                "description": "Critical cases with benign-sounding complaints. Safety gate active.",
                "expected_score_range": [0.20, 0.40],
                "case_count": 7,
            },
        ]
    }
