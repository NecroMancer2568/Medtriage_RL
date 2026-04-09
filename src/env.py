"""
env.py — MedTriage-RL environment and Pydantic models.
Contains the TriageEnv class and all data models for the OpenEnv environment.
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# Valid actions for the triage environment
VALID_ACTIONS = Literal[
    "ASK_VITALS",
    "ASK_SYMPTOMS",
    "ASK_HISTORY",
    "ASK_PAIN",
    "ASK_DURATION",
    "ASK_CONTEXT",
    "REQUEST_IMAGE",
    "TRIAGE_1",
    "TRIAGE_2",
    "TRIAGE_3",
    "TRIAGE_4",
    "TRIAGE_5",
]

# Information-gathering actions
ASK_ACTIONS = {
    "ASK_VITALS",
    "ASK_SYMPTOMS",
    "ASK_HISTORY",
    "ASK_PAIN",
    "ASK_DURATION",
    "ASK_CONTEXT",
    "REQUEST_IMAGE",
}

# Terminal actions
TRIAGE_ACTIONS = {"TRIAGE_1", "TRIAGE_2", "TRIAGE_3", "TRIAGE_4", "TRIAGE_5"}


class TriageAction(BaseModel):
    """Action submitted to the environment."""
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


class TriageObservation(BaseModel):
    """Observation returned from reset() and step()."""
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


class TriageReward(BaseModel):
    """Reward structure with breakdown components."""
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


class StepResult(BaseModel):
    """Result returned by the step() method."""
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


class TriageEnv:
    """OpenEnv-compatible medical triage environment."""

    def __init__(self, task_id: Literal["task_1", "task_2", "task_3"]):
        self.task_id = task_id
        self.cases = self._load_cases(task_id)
        self._current_patient: Optional[dict] = None
        self._revealed_info: dict[str, str] = {}
        self._step_count: int = 0
        self._episode_id: Optional[str] = None
        self._actions_taken: list[str] = []
        self._done: bool = False
        self.max_steps: int = 6

    def _load_cases(self, task_id: str) -> list[dict]:
        """Load patient cases based on task difficulty."""
        patients_dir = Path(__file__).parent / "patients"
        
        difficulty_map = {
            "task_1": "easy_cases.json",
            "task_2": "medium_cases.json",
            "task_3": "hard_cases.json",
        }
        
        filename = difficulty_map.get(task_id)
        if not filename:
            raise ValueError(f"Invalid task_id: {task_id}")
        
        filepath = patients_dir / filename
        with open(filepath, "r") as f:
            return json.load(f)

    def reset(self) -> TriageObservation:
        """Reset environment and start a new episode with a random patient."""
        # Pick random patient from the case pool
        self._current_patient = random.choice(self.cases)
        
        # Reset all hidden state
        self._revealed_info = {}
        self._step_count = 0
        self._episode_id = str(uuid.uuid4())
        self._actions_taken = []
        self._done = False
        
        return TriageObservation(
            chief_complaint=self._current_patient["chief_complaint"],
            patient_meta=self._current_patient["patient_meta"],
            revealed_info={},
            step=0,
            max_steps=self.max_steps,
            image_available=self._current_patient["image_available"],
            task_id=self.task_id,
            done=False,
        )

    def step(self, action: TriageAction) -> StepResult:
        """Execute an action and return the result."""
        # Import here to avoid circular imports
        from src.graders import get_grader
        from src.reward import compute_step_reward
        
        if self._current_patient is None:
            raise RuntimeError("Must call reset() before step()")
        
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        action_str = action.action
        self._actions_taken.append(action_str)
        
        # Handle information-gathering actions
        if action_str in ASK_ACTIONS:
            # Check redundancy BEFORE updating revealed_info
            is_redundant = action_str in self._revealed_info
            
            # Reveal information if not already revealed
            if not is_redundant:
                answer = self._current_patient["answer_map"].get(action_str)
                if action_str == "REQUEST_IMAGE":
                    if self._current_patient["image_available"]:
                        self._revealed_info[action_str] = (
                            answer or self._current_patient.get("image_description", "")
                        )
                    else:
                        self._revealed_info[action_str] = (
                            "No visual evidence available for this case."
                        )
                else:
                    self._revealed_info[action_str] = answer or ""
            
            self._step_count += 1
            
            # Check for step exhaustion
            if self._step_count >= self.max_steps:
                self._done = True
                reward = TriageReward(
                    total=-2.0,
                    components={"timeout": -2.0},
                    is_terminal=True,
                    grader_score=0.001,
                    explanation="Max steps reached without triage decision."
                )
            else:
                reward = compute_step_reward(
                    patient=self._current_patient,
                    action=action,
                    revealed_info=self._revealed_info,
                    step=self._step_count,
                    max_steps=self.max_steps,
                    is_redundant=is_redundant,
                )
        
        # Handle terminal triage actions
        elif action_str in TRIAGE_ACTIONS:
            assigned_esi = int(action_str[-1])
            self._done = True
            
            grader = get_grader(self.task_id)
            grader_score = grader.grade(
                patient=self._current_patient,
                actions_taken=self._actions_taken[:-1],  # Exclude the TRIAGE action itself
                assigned_esi=assigned_esi,
            )
            
            reward = TriageReward(
                total=grader_score,
                components={"accuracy": grader_score},
                is_terminal=True,
                grader_score=grader_score,
                explanation=f"Episode ended. Assigned ESI {assigned_esi}. Grader score: {grader_score}"
            )
        
        else:
            raise ValueError(f"Invalid action: {action_str}")
        
        # Build observation
        observation = TriageObservation(
            chief_complaint=self._current_patient["chief_complaint"],
            patient_meta=self._current_patient["patient_meta"],
            revealed_info=dict(self._revealed_info),
            step=self._step_count,
            max_steps=self.max_steps,
            image_available=self._current_patient["image_available"],
            task_id=self.task_id,
            done=self._done,
        )
        
        # Build info dict
        info = {
            "action_received": action_str,
            "image_data_received": action.image_data is not None,
            "image_processing": self._get_image_processing_status(action),
            "episode_id": self._episode_id,
            "case_id": self._current_patient["id"],
            "true_esi": self._current_patient["true_esi"],
            "current_task": self.task_id,
            "step": self._step_count,
            "note": self._get_image_note(action),
        }
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info=info,
        )

    def _get_image_processing_status(self, action: TriageAction) -> str:
        """Get image processing status for info dict."""
        if action.action != "REQUEST_IMAGE":
            return "not_applicable"
        if self._current_patient and self._current_patient["image_available"]:
            return "standardised_clinical_description_returned"
        return "image_not_available_for_this_case"

    def _get_image_note(self, action: TriageAction) -> str:
        """Get note about image processing for info dict."""
        if action.action == "REQUEST_IMAGE":
            return (
                "REQUEST_IMAGE returns pre-defined clinical descriptions. "
                "External image_data payloads are accepted but not processed. "
                "This ensures deterministic reproducible grading."
            )
        return ""

    def state(self) -> TriageObservation:
        """Return current observation without advancing anything."""
        if self._current_patient is None:
            raise RuntimeError("Must call reset() before state()")
        
        return TriageObservation(
            chief_complaint=self._current_patient["chief_complaint"],
            patient_meta=self._current_patient["patient_meta"],
            revealed_info=dict(self._revealed_info),
            step=self._step_count,
            max_steps=self.max_steps,
            image_available=self._current_patient["image_available"],
            task_id=self.task_id,
            done=self._done,
        )
