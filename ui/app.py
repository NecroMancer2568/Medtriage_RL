"""
ui/app.py — Gradio interface for MedTriage-RL.
Runs alongside the FastAPI server. Lets users interact with
the triage environment through a visual interface.
"""

import gradio as gr
import requests
import json

ENV_URL = "http://localhost:7860"

TASK_DESCRIPTIONS = {
    "task_1": "Easy — Clear unambiguous presentations. Basic ESI knowledge.",
    "task_2": "Medium — Ambiguous cases. Must ask discriminating questions.",
    "task_3": "Hard — Deceptive presentations. Safety gate active.",
}

ACTIONS = [
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

ACTION_DESCRIPTIONS = {
    "ASK_VITALS":    "Measure HR, BP, RR, SpO2, temperature",
    "ASK_SYMPTOMS":  "Ask about associated symptoms",
    "ASK_HISTORY":   "Past medical history, medications, allergies",
    "ASK_PAIN":      "Pain severity, character, location, radiation",
    "ASK_DURATION":  "Onset timing and how symptoms progressed",
    "ASK_CONTEXT":   "Situational context, exposures, recent events",
    "REQUEST_IMAGE": "Ask patient to show the affected area visually",
    "TRIAGE_1":      "IMMEDIATE — life-threatening, resuscitation now",
    "TRIAGE_2":      "EMERGENT — seen within 15 minutes",
    "TRIAGE_3":      "URGENT — seen within 60 minutes",
    "TRIAGE_4":      "LESS URGENT — one resource, low risk",
    "TRIAGE_5":      "NON-URGENT — no resources needed",
}


def reset_episode(task_id: str):
    """Reset environment and start a new episode."""
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=10,
        )
        resp.raise_for_status()
        obs = resp.json()

        patient_info = (
            f"**Age:** {obs['patient_meta']['age']} years old  \n"
            f"**Gender:** {obs['patient_meta']['gender'].capitalize()}  \n"
            f"**Chief Complaint:** {obs['chief_complaint']}  \n"
            f"**Image Available:** {'Yes' if obs['image_available'] else 'No'}  \n"
            f"**Step:** {obs['step']} / {obs['max_steps']}"
        )

        history = (
            f"### Episode started — {TASK_DESCRIPTIONS[task_id]}\n\n"
            f"**Patient presenting:**\n{obs['chief_complaint']}\n\n"
            f"Gather information and assign a triage level."
        )

        return (
            patient_info,   # patient_box
            history,        # history_box
            "",             # revealed_box
            "",             # reward_box
            False,          # done state
            obs,            # observation state
            0.0,            # cumulative reward
        )

    except Exception as e:
        error_msg = f"Error connecting to environment: {e}"
        return error_msg, error_msg, "", "", False, {}, 0.0


def take_action(action: str, obs_state: dict, history: str,
                cumulative_reward: float, done: bool, task_id: str):
    """Execute one action against the environment."""

    if done:
        return (
            obs_state.get("chief_complaint", ""),
            history + "\n\n**Episode is over. Click Reset to start a new episode.**",
            "",
            "",
            done,
            obs_state,
            cumulative_reward,
        )

    if not obs_state:
        return (
            "",
            "**Please reset the environment first.**",
            "",
            "",
            False,
            {},
            0.0,
        )

    try:
        resp = requests.post(
            f"{ENV_URL}/step",
            params={"task_id": task_id},
            json={"action": action},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()

        new_obs    = result["observation"]
        reward     = result["reward"]
        is_done    = result["done"]
        step_total = reward["total"]
        explanation = reward["explanation"]
        grader_score = reward.get("grader_score")

        cumulative_reward += step_total

        # Build patient info panel
        patient_info = (
            f"**Age:** {new_obs['patient_meta']['age']} years old  \n"
            f"**Gender:** {new_obs['patient_meta']['gender'].capitalize()}  \n"
            f"**Chief Complaint:** {new_obs['chief_complaint']}  \n"
            f"**Image Available:** {'Yes' if new_obs['image_available'] else 'No'}  \n"
            f"**Step:** {new_obs['step']} / {new_obs['max_steps']}"
        )

        # Build revealed info panel
        revealed_lines = []
        for key, value in new_obs.get("revealed_info", {}).items():
            revealed_lines.append(f"**{key}:**\n{value}\n")
        revealed_text = "\n---\n".join(revealed_lines) if revealed_lines else "Nothing revealed yet."

        # Build reward panel
        reward_color = "green" if step_total > 0 else ("red" if step_total < 0 else "gray")
        reward_text = (
            f"**Step reward:** {step_total:+.2f}  \n"
            f"**Cumulative reward:** {cumulative_reward:.2f}  \n"
            f"**Explanation:** {explanation}"
        )

        # Build history entry
        action_line = (
            f"\n\n**Step {new_obs['step']} — {action}**  \n"
            f"_{ACTION_DESCRIPTIONS.get(action, '')}_  \n"
            f"Reward: {step_total:+.2f}"
        )

        if is_done:
            if grader_score is not None:
                final_grade = grader_score
                if final_grade >= 0.8:
                    grade_label = "Excellent"
                elif final_grade >= 0.6:
                    grade_label = "Good"
                elif final_grade >= 0.4:
                    grade_label = "Partial"
                elif final_grade > 0.0:
                    grade_label = "Poor"
                else:
                    grade_label = "Failed"

                action_line += (
                    f"\n\n---\n### Episode complete\n"
                    f"**Grader score: {final_grade:.3f} — {grade_label}**  \n"
                    f"Total steps: {new_obs['step']}  \n"
                    f"Total reward: {cumulative_reward:.2f}"
                )

                reward_text += (
                    f"\n\n**FINAL GRADER SCORE: {final_grade:.3f}**  \n"
                    f"**Grade: {grade_label}**"
                )
            else:
                action_line += (
                    "\n\n---\n### Episode timed out\n"
                    "Max steps reached without triage decision."
                )

        new_history = history + action_line

        return (
            patient_info,
            new_history,
            revealed_text,
            reward_text,
            is_done,
            new_obs,
            cumulative_reward,
        )

    except Exception as e:
        error_line = f"\n\n**Error on action {action}:** {e}"
        return (
            obs_state.get("chief_complaint", ""),
            history + error_line,
            "",
            "",
            done,
            obs_state,
            cumulative_reward,
        )


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="MedTriage-RL",
    theme=gr.themes.Soft(),
) as demo:

    # Hidden state
    obs_state         = gr.State({})
    done_state        = gr.State(False)
    cumulative_reward = gr.State(0.0)

    # Header
    gr.Markdown("""
    # MedTriage-RL
    **Medical Emergency Department Triage — OpenEnv RL Environment**

    An AI triage nurse simulation. Observe the patient, gather information,
    and assign the correct ESI urgency level. Your grader score reflects
    clinical reasoning quality, not just the final answer.
    """)

    with gr.Row():
        with gr.Column(scale=1):

            # Task selector and reset
            gr.Markdown("### Setup")
            task_selector = gr.Dropdown(
                choices=["task_1", "task_2", "task_3"],
                value="task_1",
                label="Task difficulty",
                info="Easy → Medium → Hard (deceptive presentations)",
            )
            reset_btn = gr.Button(
                "Reset / New Patient",
                variant="primary",
                size="lg",
            )

            # Patient info
            gr.Markdown("### Current patient")
            patient_box = gr.Markdown("Click **Reset** to start.")

            # Reward panel
            gr.Markdown("### Reward")
            reward_box = gr.Markdown("No actions taken yet.")

        with gr.Column(scale=2):

            # Action selector
            gr.Markdown("### Choose your action")
            with gr.Row():
                action_selector = gr.Dropdown(
                    choices=ACTIONS,
                    value="ASK_VITALS",
                    label="Action",
                    scale=2,
                )
                action_btn = gr.Button(
                    "Take action",
                    variant="secondary",
                    size="lg",
                    scale=1,
                )

            # Action reference
            with gr.Accordion("Action reference", open=False):
                action_ref = "\n".join(
                    f"- **{k}** — {v}"
                    for k, v in ACTION_DESCRIPTIONS.items()
                )
                gr.Markdown(action_ref)

            # Revealed info
            gr.Markdown("### Information gathered")
            revealed_box = gr.Markdown("Nothing revealed yet.")

    # Full-width history
    gr.Markdown("### Episode history")
    history_box = gr.Markdown("No episode started.")

    # ── Event handlers ───────────────────────────────────────────────────────

    reset_btn.click(
        fn=reset_episode,
        inputs=[task_selector],
        outputs=[
            patient_box,
            history_box,
            revealed_box,
            reward_box,
            done_state,
            obs_state,
            cumulative_reward,
        ],
    )

    action_btn.click(
        fn=take_action,
        inputs=[
            action_selector,
            obs_state,
            history_box,
            cumulative_reward,
            done_state,
            task_selector,
        ],
        outputs=[
            patient_box,
            history_box,
            revealed_box,
            reward_box,
            done_state,
            obs_state,
            cumulative_reward,
        ],
    )

    # ESI reference
    with gr.Accordion("ESI level reference", open=False):
        gr.Markdown("""
        | Level | Name | Seen within | Example |
        |---|---|---|---|
        | ESI 1 | Immediate | Now | Cardiac arrest, active seizure |
        | ESI 2 | Emergent | 15 minutes | Chest pain with diaphoresis |
        | ESI 3 | Urgent | 60 minutes | Abdominal pain needing labs |
        | ESI 4 | Less urgent | 2 hours | Ankle sprain needing X-ray |
        | ESI 5 | Non-urgent | Hours | Sore throat, minor cold |
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True,
    )