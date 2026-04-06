"""
patient_sim.py — Patient loader and case selection for MedTriage-RL.
Handles loading patient cases from JSON files and selecting by difficulty.
"""

import json
import random
from pathlib import Path
from typing import Literal, Optional


class PatientSimulator:
    """Manages patient case loading and selection."""
    
    def __init__(self):
        self._cases_cache: dict[str, list[dict]] = {}
        self._patients_dir = Path(__file__).parent / "patients"
    
    def _load_cases_for_difficulty(self, difficulty: str) -> list[dict]:
        """Load cases for a specific difficulty level."""
        if difficulty in self._cases_cache:
            return self._cases_cache[difficulty]
        
        filename_map = {
            "easy": "easy_cases.json",
            "medium": "medium_cases.json",
            "hard": "hard_cases.json",
        }
        
        filename = filename_map.get(difficulty)
        if not filename:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        
        filepath = self._patients_dir / filename
        with open(filepath, "r") as f:
            cases = json.load(f)
        
        self._cases_cache[difficulty] = cases
        return cases
    
    def load_cases_for_task(
        self, task_id: Literal["task_1", "task_2", "task_3"]
    ) -> list[dict]:
        """Load cases for a specific task."""
        task_to_difficulty = {
            "task_1": "easy",
            "task_2": "medium",
            "task_3": "hard",
        }
        
        difficulty = task_to_difficulty.get(task_id)
        if not difficulty:
            raise ValueError(f"Invalid task_id: {task_id}")
        
        return self._load_cases_for_difficulty(difficulty)
    
    def get_random_case(
        self, task_id: Literal["task_1", "task_2", "task_3"]
    ) -> dict:
        """Get a random case from the specified task's case pool."""
        cases = self.load_cases_for_task(task_id)
        return random.choice(cases)
    
    def get_case_by_id(self, case_id: str) -> Optional[dict]:
        """Get a specific case by its ID."""
        # Determine difficulty from case_id prefix
        if case_id.startswith("easy_"):
            difficulty = "easy"
        elif case_id.startswith("medium_"):
            difficulty = "medium"
        elif case_id.startswith("hard_"):
            difficulty = "hard"
        else:
            return None
        
        cases = self._load_cases_for_difficulty(difficulty)
        for case in cases:
            if case["id"] == case_id:
                return case
        
        return None
    
    def get_all_cases(self) -> list[dict]:
        """Get all cases across all difficulties."""
        all_cases = []
        for difficulty in ["easy", "medium", "hard"]:
            all_cases.extend(self._load_cases_for_difficulty(difficulty))
        return all_cases
    
    def get_case_count(self, task_id: Optional[str] = None) -> int:
        """Get the number of cases, optionally filtered by task."""
        if task_id:
            return len(self.load_cases_for_task(task_id))
        return len(self.get_all_cases())


# Singleton instance for convenience
_simulator: Optional[PatientSimulator] = None


def get_patient_simulator() -> PatientSimulator:
    """Get the singleton PatientSimulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = PatientSimulator()
    return _simulator
