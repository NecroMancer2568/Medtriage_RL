"""
reward.py — Dense step-level reward function for MedTriage-RL.
Provides signal at every step, not just at episode end.
"""

from src.env import TriageAction, TriageReward


def compute_step_reward(
    patient: dict,
    action: TriageAction,
    revealed_info: dict[str, str],
    step: int,
    max_steps: int,
    is_redundant: bool = False,
) -> TriageReward:
    """
    Compute dense reward for a single step.
    
    This function is called for information-gathering actions (ASK_* and REQUEST_IMAGE).
    Terminal actions (TRIAGE_*) are handled directly in env.step() using the grader.
    
    Args:
        patient: Current patient case dict.
        action: The action taken.
        revealed_info: Currently revealed information (after this action).
        step: Current step number (after action).
        max_steps: Maximum steps allowed.
        is_redundant: Whether this action was already taken in this episode.
        
    Returns:
        TriageReward with component breakdown.
    """
    action_str = action.action
    components = {}
    explanations = []
    
    disc_qs = set(patient.get("discriminating_questions", []))
    
    # Relevance: was this a discriminating question?
    relevance = 0.3 if action_str in disc_qs else 0.0
    components["relevance"] = relevance
    if relevance > 0:
        explanations.append("Discriminating question — good signal.")
    else:
        explanations.append("Non-discriminating question.")
    
    # Redundancy penalty
    redundancy = -0.1 if is_redundant else 0.0
    components["redundancy"] = redundancy
    if redundancy < 0:
        explanations.append("(Already asked — redundant.)")
    
    # Urgency penalty: wasting steps near the end
    steps_left = max_steps - step
    urgency_penalty = -0.15 if steps_left <= 2 else 0.0
    components["urgency"] = urgency_penalty
    if urgency_penalty < 0:
        explanations.append(f"(Warning: only {steps_left} steps remaining.)")
    
    total = relevance + redundancy + urgency_penalty
    
    explanation = f"Asked {action_str}. " + " ".join(explanations)
    
    return TriageReward(
        total=round(total, 4),
        components=components,
        is_terminal=False,
        grader_score=None,
        explanation=explanation,
    )
