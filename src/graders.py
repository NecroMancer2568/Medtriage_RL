"""
graders.py — Task-specific graders for MedTriage-RL.
Each grader implements a deterministic scoring formula returning 0.0-1.0.
"""

from abc import ABC, abstractmethod
from typing import Literal


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
            Score from 0.0 to 1.0.
        """
        pass


class Task1Grader(BaseGrader):
    """
    Task 1 Grader for easy cases.
    Evaluates accuracy, reasoning quality, and efficiency.
    Expected frontier model score: 0.75-0.90
    
    Scoring logic:
    - Wrong by 2+ ESI levels: always 0.0
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
            return 0.001  # wrong by 2+, efficiency is irrelevant

        # Step 2: reasoning check — did agent ask anything clinically relevant?
        # An agent that never asked a discriminating question showed no reasoning.
        asked = set(actions_taken)
        disc_qs = set(patient.get("discriminating_questions", []))
        asked_something_relevant = bool(asked & disc_qs)

        if not asked_something_relevant:
            # Correct answer but zero clinical reasoning demonstrated.
            # Cap hard at 0.4 to punish lucky guesses.
            return max(0.001, min(0.999, round(accuracy * 0.4, 4)))

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

        return max(0.001, min(0.999, round(accuracy * efficiency, 4)))

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
        # Asking 1 of 2 discriminating questions = partial but meaningful
        # Asking both = full credit
        # Asking neither when red flags present = heavy penalty
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
                # Partial credit: first discriminating question gives 0.6,
                # each additional gives proportional bonus up to 1.0
                # This rewards asking one good question even if not both
                relevance = 0.6 + (0.4 * (matched - 1) / (total - 1)) if total > 1 else 0.6

        # Step 3: redundancy penalty
        redundant = len(actions_taken) - len(set(actions_taken))
        redundancy_penalty = min(redundant * 0.1, 0.3)

        # Step 4: red flag penalty — only fires when NO discriminating
        # questions were asked at all, not for partial coverage
        red_flag_penalty = 0.0
        if patient.get("red_flags") and disc_qs.isdisjoint(asked):
            red_flag_penalty = 0.3

        score = (accuracy * relevance) - redundancy_penalty - red_flag_penalty
        return max(0.001, min(0.999, round(score, 4)))


class Task3Grader(BaseGrader):
    """
    Task 3 Grader for hard (deceptive) cases.
    Includes safety gate for catastrophic undertriage.
    Expected frontier model score: 0.20-0.40
    """
    
    def grade(self, patient: dict, actions_taken: list[str], assigned_esi: int) -> float:
        true_esi = patient["true_esi"]
        
        # SAFETY GATE — hard zero for critical undertriage
        # All hard cases are ESI 1 — assigning ESI 3 or 4 or 5 = dangerous miss
        if true_esi <= 2 and assigned_esi >= 3:
            return 0.001  # safety gate: catastrophic undertriage
        
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
        # Require at least 2 discriminating questions for full probing credit
        # One question (usually ASK_VITALS) is not enough to defeat a deceptive case
        probed_deeply = len(asked & disc_qs) >= 2
        probed_at_all = bool(asked & disc_qs)
        depth_bonus = 0.2 if probed_deeply else 0.0

        if accuracy == 1.0 and not probed_deeply:
            return round(0.3, 4)
        
        score = accuracy + depth_bonus
        return max(0.001, min(0.999, round(score, 4)))


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
