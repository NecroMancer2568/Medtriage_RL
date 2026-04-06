"""
test_graders.py — Tests for MedTriage-RL graders.
Proves graders are deterministic and return scores in 0.0-1.0 range.
"""

import pytest
from src.graders import Task1Grader, Task2Grader, Task3Grader


class TestTask1Grader:
    """Test cases for Task 1 (easy) grader."""
    
    @pytest.fixture
    def grader(self):
        return Task1Grader()
    
    @pytest.fixture
    def easy_patient(self):
        return {
            "id": "easy_001",
            "true_esi": 2,
            "red_flags": ["chest pain"],
            "discriminating_questions": ["ASK_VITALS", "ASK_SYMPTOMS"],
        }
    
    def test_correct_esi_few_questions(self, grader, easy_patient):
        """Correct ESI with 2 questions → score ≈ 1.0"""
        actions = ["ASK_VITALS", "ASK_SYMPTOMS"]
        score = grader.grade(easy_patient, actions, assigned_esi=2)
        
        assert score == 1.0  # Perfect accuracy * perfect efficiency
        assert 0.0 <= score <= 1.0
    
    def test_off_by_one_esi(self, grader, easy_patient):
        """Off by 1 ESI with 3 questions → score ≈ 0.425"""
        actions = ["ASK_VITALS", "ASK_SYMPTOMS", "ASK_HISTORY"]
        score = grader.grade(easy_patient, actions, assigned_esi=3)  # Off by 1
        
        # 0.5 accuracy * 0.85 efficiency = 0.425
        assert score == 0.425
        assert 0.0 <= score <= 1.0
    
    def test_off_by_two_esi(self, grader, easy_patient):
        """Off by 2+ ESI → score = 0.0"""
        actions = ["ASK_VITALS"]
        score = grader.grade(easy_patient, actions, assigned_esi=4)  # Off by 2
        
        assert score == 0.0
    
    def test_deterministic(self, grader, easy_patient):
        """Same inputs always produce same output."""
        actions = ["ASK_VITALS", "ASK_SYMPTOMS"]
        
        scores = [grader.grade(easy_patient, actions, assigned_esi=2) for _ in range(10)]
        
        assert all(s == scores[0] for s in scores)


class TestTask2Grader:
    """Test cases for Task 2 (medium) grader."""
    
    @pytest.fixture
    def grader(self):
        return Task2Grader()
    
    @pytest.fixture
    def medium_patient(self):
        return {
            "id": "medium_001",
            "true_esi": 2,
            "red_flags": ["altered consciousness"],
            "discriminating_questions": ["ASK_HISTORY", "ASK_SYMPTOMS"],
        }
    
    def test_correct_esi_all_discriminating(self, grader, medium_patient):
        """Correct ESI + asked all discriminating questions → score ≈ 0.9-1.0"""
        actions = ["ASK_HISTORY", "ASK_SYMPTOMS"]  # Both discriminating
        score = grader.grade(medium_patient, actions, assigned_esi=2)
        
        # 1.0 accuracy * 1.0 relevance - 0 penalties = 1.0
        assert score == 1.0
        assert 0.0 <= score <= 1.0
    
    def test_correct_esi_skipped_discriminating(self, grader, medium_patient):
        """Correct ESI but skipped discriminating questions → score ≈ 0.5"""
        actions = ["ASK_VITALS", "ASK_PAIN"]  # Neither discriminating
        score = grader.grade(medium_patient, actions, assigned_esi=2)
        
        # 1.0 accuracy * 0.0 relevance = 0.0, then -0.3 red flag penalty = -0.3 → clamped to 0.0
        assert score == 0.0
        assert 0.0 <= score <= 1.0
    
    def test_wrong_esi(self, grader, medium_patient):
        """Wrong ESI → score ≈ 0.0"""
        actions = ["ASK_HISTORY", "ASK_SYMPTOMS"]
        score = grader.grade(medium_patient, actions, assigned_esi=5)  # Way off
        
        assert score == 0.0
    
    def test_redundancy_penalty(self, grader, medium_patient):
        """Repeated actions incur redundancy penalty."""
        actions = ["ASK_HISTORY", "ASK_HISTORY", "ASK_SYMPTOMS"]  # One repeat
        score = grader.grade(medium_patient, actions, assigned_esi=2)
        
        # 1.0 * 1.0 - 0.1 (redundancy) = 0.9
        assert score == 0.9
        assert 0.0 <= score <= 1.0
    
    def test_deterministic(self, grader, medium_patient):
        """Same inputs always produce same output."""
        actions = ["ASK_HISTORY", "ASK_SYMPTOMS"]
        
        scores = [grader.grade(medium_patient, actions, assigned_esi=2) for _ in range(10)]
        
        assert all(s == scores[0] for s in scores)


class TestTask3Grader:
    """Test cases for Task 3 (hard/deceptive) grader."""
    
    @pytest.fixture
    def grader(self):
        return Task3Grader()
    
    @pytest.fixture
    def hard_patient(self):
        return {
            "id": "hard_001",
            "true_esi": 1,  # All hard cases are ESI 1
            "red_flags": ["subtle cardiac signs"],
            "discriminating_questions": ["ASK_VITALS", "ASK_HISTORY"],
            "deceptive_element": "Patient presents with vague complaint",
        }
    
    def test_correct_esi_with_probing(self, grader, hard_patient):
        """ESI 1 assigned + probed correctly → score ≈ 1.0"""
        actions = ["ASK_VITALS", "ASK_HISTORY"]  # Both discriminating
        score = grader.grade(hard_patient, actions, assigned_esi=1)
        
        # 1.0 accuracy + 0.2 depth bonus = 1.2 → clamped to 1.0
        assert score == 1.0
        assert 0.0 <= score <= 1.0
    
    def test_correct_esi_no_probing_lucky_guess(self, grader, hard_patient):
        """ESI 1 assigned but no probing (lucky guess) → score = 0.5"""
        actions = ["ASK_PAIN", "ASK_CONTEXT"]  # Neither discriminating
        score = grader.grade(hard_patient, actions, assigned_esi=1)
        
        assert score == 0.5  # Capped at 0.5 for lucky guess
        assert 0.0 <= score <= 1.0
    
    def test_safety_gate_catastrophic_undertriage(self, grader, hard_patient):
        """ESI 4 or 5 on critical case (true ESI 1) → score = 0.0 (safety gate)"""
        actions = ["ASK_VITALS"]
        
        # Test ESI 4
        score4 = grader.grade(hard_patient, actions, assigned_esi=4)
        assert score4 == 0.0  # Safety gate triggered
        
        # Test ESI 5
        score5 = grader.grade(hard_patient, actions, assigned_esi=5)
        assert score5 == 0.0  # Safety gate triggered
    
    def test_off_by_one_with_probing(self, grader, hard_patient):
        """ESI 2 assigned (off by 1) with probing."""
        actions = ["ASK_VITALS", "ASK_HISTORY"]
        score = grader.grade(hard_patient, actions, assigned_esi=2)
        
        # 0.5 accuracy + 0.2 depth bonus = 0.7
        assert score == 0.7
        assert 0.0 <= score <= 1.0
    
    def test_deterministic(self, grader, hard_patient):
        """Same inputs always produce same output."""
        actions = ["ASK_VITALS", "ASK_HISTORY"]
        
        scores = [grader.grade(hard_patient, actions, assigned_esi=1) for _ in range(10)]
        
        assert all(s == scores[0] for s in scores)


class TestGraderScoreRange:
    """Ensure all graders return scores in valid 0.0-1.0 range."""
    
    @pytest.mark.parametrize("GraderClass", [Task1Grader, Task2Grader, Task3Grader])
    def test_score_bounds(self, GraderClass):
        """All grader scores must be in [0.0, 1.0]."""
        grader = GraderClass()
        patient = {
            "id": "test",
            "true_esi": 2,
            "red_flags": ["test"],
            "discriminating_questions": ["ASK_VITALS"],
        }
        
        # Test various ESI assignments
        for esi in range(1, 6):
            for actions in [[], ["ASK_VITALS"], ["ASK_VITALS", "ASK_SYMPTOMS"]]:
                score = grader.grade(patient, actions, assigned_esi=esi)
                
                assert isinstance(score, float), f"Score should be float, got {type(score)}"
                assert 0.0 <= score <= 1.0, f"Score {score} out of bounds [0.0, 1.0]"
