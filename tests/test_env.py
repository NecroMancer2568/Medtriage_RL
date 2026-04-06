"""
test_env.py — Tests for MedTriage-RL environment.
Validates OpenEnv spec compliance.
"""

import pytest
from src.env import TriageEnv, TriageAction, TriageObservation, StepResult


class TestTriageEnvBasics:
    """Basic environment tests."""
    
    @pytest.fixture
    def env(self):
        return TriageEnv("task_1")
    
    def test_reset_returns_valid_observation(self, env):
        """reset() returns valid TriageObservation with step=0."""
        obs = env.reset()
        
        assert isinstance(obs, TriageObservation)
        assert obs.step == 0
        assert obs.done is False
        assert obs.chief_complaint != ""
        assert "age" in obs.patient_meta
        assert "gender" in obs.patient_meta
        assert obs.revealed_info == {}
        assert obs.task_id == "task_1"
        assert obs.max_steps == 6
    
    def test_state_after_reset_equals_reset_output(self, env):
        """state() after reset equals reset() output."""
        reset_obs = env.reset()
        state_obs = env.state()
        
        assert reset_obs.chief_complaint == state_obs.chief_complaint
        assert reset_obs.patient_meta == state_obs.patient_meta
        assert reset_obs.step == state_obs.step
        assert reset_obs.done == state_obs.done
        assert reset_obs.revealed_info == state_obs.revealed_info
    
    def test_state_before_reset_raises(self):
        """state() before reset() raises RuntimeError."""
        env = TriageEnv("task_1")
        
        with pytest.raises(RuntimeError, match="Must call reset"):
            env.state()
    
    def test_step_before_reset_raises(self):
        """step() before reset() raises RuntimeError."""
        env = TriageEnv("task_1")
        action = TriageAction(action="ASK_VITALS")
        
        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step(action)


class TestTriageEnvActions:
    """Test action handling."""
    
    @pytest.fixture
    def env(self):
        env = TriageEnv("task_1")
        env.reset()
        return env
    
    def test_ask_vitals_reveals_info(self, env):
        """ASK_VITALS reveals vitals information."""
        action = TriageAction(action="ASK_VITALS")
        result = env.step(action)
        
        assert isinstance(result, StepResult)
        assert "ASK_VITALS" in result.observation.revealed_info
        assert result.done is False
        assert result.observation.step == 1
    
    def test_triage_action_ends_episode(self, env):
        """TRIAGE_* action sets done=True."""
        action = TriageAction(action="TRIAGE_1")
        result = env.step(action)
        
        assert result.done is True
        assert result.observation.done is True
        assert result.reward.is_terminal is True
        assert result.reward.grader_score is not None
        assert 0.0 <= result.reward.grader_score <= 1.0
    
    def test_step_after_done_raises(self, env):
        """step() after done=True raises RuntimeError."""
        # End the episode
        env.step(TriageAction(action="TRIAGE_3"))
        
        # Try to step again
        with pytest.raises(RuntimeError, match="Episode is done"):
            env.step(TriageAction(action="ASK_VITALS"))
    
    def test_revealed_info_grows_across_steps(self, env):
        """revealed_info accumulates across steps."""
        env.step(TriageAction(action="ASK_VITALS"))
        result = env.step(TriageAction(action="ASK_SYMPTOMS"))
        
        assert "ASK_VITALS" in result.observation.revealed_info
        assert "ASK_SYMPTOMS" in result.observation.revealed_info
        assert len(result.observation.revealed_info) == 2


class TestTriageEnvImageHandling:
    """Test REQUEST_IMAGE handling."""
    
    def test_request_image_when_available(self):
        """REQUEST_IMAGE on image_available=True returns description."""
        # We need to find a case with image_available=True
        env = TriageEnv("task_1")
        
        # Reset until we get a case with image
        for _ in range(20):
            obs = env.reset()
            if obs.image_available:
                break
        
        if not obs.image_available:
            pytest.skip("No image-available case found in test iterations")
        
        result = env.step(TriageAction(action="REQUEST_IMAGE"))
        
        assert "REQUEST_IMAGE" in result.observation.revealed_info
        assert result.observation.revealed_info["REQUEST_IMAGE"] != ""
        assert "No visual evidence" not in result.observation.revealed_info["REQUEST_IMAGE"]
    
    def test_request_image_when_not_available(self):
        """REQUEST_IMAGE on image_available=False returns safe string."""
        env = TriageEnv("task_1")
        
        # Reset until we get a case without image
        for _ in range(20):
            obs = env.reset()
            if not obs.image_available:
                break
        
        if obs.image_available:
            pytest.skip("No non-image case found in test iterations")
        
        result = env.step(TriageAction(action="REQUEST_IMAGE"))
        
        assert "REQUEST_IMAGE" in result.observation.revealed_info
        assert "No visual evidence" in result.observation.revealed_info["REQUEST_IMAGE"]
    
    def test_image_data_payload_does_not_crash(self):
        """step() with image_data payload does not crash."""
        env = TriageEnv("task_1")
        env.reset()
        
        # Send action with image_data
        action = TriageAction(action="REQUEST_IMAGE", image_data="base64encodeddata...")
        result = env.step(action)
        
        # Should not crash, and info should note image_data was received
        assert result.info["image_data_received"] is True
        assert result.info["action_received"] == "REQUEST_IMAGE"


class TestTriageEnvInfoDict:
    """Test info dict contents."""
    
    @pytest.fixture
    def env(self):
        env = TriageEnv("task_1")
        env.reset()
        return env
    
    def test_info_dict_contents(self, env):
        """Info dict contains required fields."""
        result = env.step(TriageAction(action="ASK_VITALS"))
        
        assert "action_received" in result.info
        assert "image_data_received" in result.info
        assert "image_processing" in result.info
        assert "episode_id" in result.info
        assert "case_id" in result.info
        assert "current_task" in result.info
        assert "step" in result.info
        
        assert result.info["action_received"] == "ASK_VITALS"
        assert result.info["image_data_received"] is False
        assert result.info["current_task"] == "task_1"


class TestTriageEnvTimeout:
    """Test step exhaustion handling."""
    
    def test_max_steps_exhaustion(self):
        """Episode ends with penalty when max_steps reached without TRIAGE."""
        env = TriageEnv("task_1")
        env.max_steps = 3  # Reduce for faster test
        env.reset()
        
        # Take 3 ASK actions
        env.step(TriageAction(action="ASK_VITALS"))
        env.step(TriageAction(action="ASK_SYMPTOMS"))
        result = env.step(TriageAction(action="ASK_HISTORY"))
        
        assert result.done is True
        assert result.reward.total == -2.0
        assert result.reward.grader_score == 0.0
        assert "timeout" in result.reward.components


class TestTriageEnvTasks:
    """Test different task configurations."""
    
    @pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
    def test_all_tasks_work(self, task_id):
        """All three tasks can be created and run."""
        env = TriageEnv(task_id)
        obs = env.reset()
        
        assert obs.task_id == task_id
        assert len(env.cases) == 7  # 7 cases per difficulty
    
    def test_invalid_task_raises(self):
        """Invalid task_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid task_id"):
            TriageEnv("task_99")


class TestPydanticValidation:
    """Test Pydantic model validation."""
    
    def test_invalid_action_rejected(self):
        """Invalid action string is rejected by Pydantic."""
        with pytest.raises(Exception):  # ValidationError
            TriageAction(action="INVALID_ACTION")
    
    def test_extra_fields_forbidden(self):
        """Extra fields in TriageAction are rejected."""
        with pytest.raises(Exception):  # ValidationError
            TriageAction(action="ASK_VITALS", unknown_field="test")
