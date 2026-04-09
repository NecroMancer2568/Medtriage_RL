"""
graders.py — Task-specific graders for MedTriage-RL.
Each grader implements a deterministic scoring formula.
All scores are clamped to the open interval (0.001, 0.999)
because the judging system rejects exactly 0.0 and 1.0.
"""

from abc import ABC, abstractmethod
from typing import Literal


def _clamp(score: float) -> float:
    """
    Clamp score to open interval (0.001, 0.999).
    The judging system requires scores strictly between 0 and 1.
    0.0 and 1.0 are both rejected by the validator.
    """
    return round(max(0.001, min(0.999, score)), 4)


class BaseGrader(ABC):
    """Abstract base class for task graders."""
    
    @abstractmethod
    def grade(self, patient: dict, actions_taken: list[str], assigned_esi: int) -> float:
        """
        Grade an episode.
        
        Args:
            patient: Full patient case dict including hidden_diagnosis, true_esi, etc.
            actions_taken: List of actions taken before the TRIAGE action.
            assigned_esi: The ESI level assigned by the agent (1-5).
            
        Returns:
            Score from 0.001 to 0.999 (open interval).
        """
        pass


class Task1Grader(BaseGrader):
    """
    Task 1 Grader for easy cases.
    Evaluates accuracy, reasoning quality, and efficiency.
    Expected frontier model score: 0.75-0.90
    
    Scoring logic:
    - Wrong by 2+ ESI levels: always ~0.0
    - Correct/off-by-1 WITHOUT asking any discriminating question:
      capped at accuracy * 0.4 (lucky guess penalty)
    - Correct/off-by-1 WITH at least 1 discriminating question:
      accuracy * efficiency_bonus (full scoring)
    
    This ensures random agents that accidentally guess correctly
    cannot score above 0.4, while agents that show clinical
    reasoning are rewarded with full efficiency credit.
    """

    def grade(self, patient: dict, actions_taken: list[str], assigned_esi: int) -> float:
        # Step 1: accuracy
        esi_diff = abs(assigned_esi - patient["true_esi"])
        if esi_diff == 0:
            accuracy = 1.0
        elif esi_diff == 1:
            accuracy = 0.5
        else:
            return _clamp(0.0)  # wrong by 2+, efficiency is irrelevant

        # Step 2: reasoning check — did agent ask anything clinically relevant?
        asked = set(actions_taken)
        disc_qs = set(patient.get("discriminating_questions", []))
        asked_something_relevant = bool(asked & disc_qs)

        if not asked_something_relevant:
            # Correct answer but zero clinical reasoning demonstrated.
            # Cap hard at 0.4 to punish lucky guesses.
            return _clamp(accuracy * 0.4)

        # Step 3: efficiency bonus — only applies when reasoning was shown
        ask_count = sum(
            1 for a in actions_taken
            if a.startswith("ASK_") or a == "REQUEST_IMAGE"
        )
        if ask_count <= 2:
            efficiency = 1.0
        elif ask_count <= 4:
            efficiency = 0.85
        else:
            efficiency = 0.7

        return _clamp(accuracy * efficiency)

class Task2Grader(BaseGrader):
    """
    Task 2 Grader for medium (ambiguous) cases.
    Evaluates accuracy, question relevance, redundancy, and red flag detection.
    Expected frontier model score: 0.45-0.65
    """
    
    def grade(self, patient: dict, actions_taken: list[str], assigned_esi: int) -> float:
        # Step 1: accuracy
        esi_diff = abs(assigned_esi - patient["true_esi"])
        if esi_diff == 0:
            accuracy = 1.0
        elif esi_diff == 1:
            accuracy = 0.5
        else:
            accuracy = 0.0

        # Step 2: question relevance with tiered credit
        asked = set(actions_taken)
        disc_qs = set(patient.get("discriminating_questions", []))

        if not disc_qs:
            relevance = 1.0
        else:
            matched = len(asked & disc_qs)
            total = len(disc_qs)
            if matched == 0:
                relevance = 0.0
            elif matched == total:
                relevance = 1.0
            else:
                relevance = 0.6 + (0.4 * (matched - 1) / (total - 1)) if total > 1 else 0.6

        # Step 3: redundancy penalty
        redundant = len(actions_taken) - len(set(actions_taken))
        redundancy_penalty = min(redundant * 0.1, 0.3)

        # Step 4: red flag penalty
        red_flag_penalty = 0.0
        if patient.get("red_flags") and disc_qs.isdisjoint(asked):
            red_flag_penalty = 0.3

        score = (accuracy * relevance) - redundancy_penalty - red_flag_penalty
        return _clamp(score)


class Task3Grader(BaseGrader):
    """
    Task 3 Grader for hard (deceptive) cases.
    Includes safety gate for catastrophic undertriage.
    Expected frontier model score: 0.20-0.40
    """
    
    def grade(self, patient: dict, actions_taken: list[str], assigned_esi: int) -> float:
        true_esi = patient["true_esi"]
        
        # SAFETY GATE — hard penalty for critical undertriage
        if true_esi <= 2 and assigned_esi >= 3:
            return _clamp(0.0)  # safety gate: catastrophic undertriage
        
        # Step 1: accuracy
        esi_diff = abs(assigned_esi - true_esi)
        if esi_diff == 0:
            accuracy = 1.0
        elif esi_diff == 1:
            accuracy = 0.5
        else:
            accuracy = 0.0
        
        # Step 2: depth bonus — did agent probe beyond the deceptive surface?
        asked = set(actions_taken)
        disc_qs = set(patient.get("discriminating_questions", []))
        probed_deeply = len(asked & disc_qs) >= 2
        probed_at_all = bool(asked & disc_qs)
        depth_bonus = 0.2 if probed_deeply else 0.0

        if accuracy == 1.0 and not probed_deeply:
            return _clamp(0.3)  # lucky guess cap
        
        score = accuracy + depth_bonus
        return _clamp(score)


def get_grader(task_id: Literal["task_1", "task_2", "task_3"]) -> BaseGrader:
    """Factory function to get the appropriate grader for a task."""
    graders = {
        "task_1": Task1Grader(),
        "task_2": Task2Grader(),
        "task_3": Task3Grader(),
    }
    
    grader = graders.get(task_id)
    if grader is None:
        raise ValueError(f"Invalid task_id: {task_id}")
    
    return grader
