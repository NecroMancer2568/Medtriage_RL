"""
evaluate_graders.py — Complete grader evaluation suite for MedTriage-RL.

Runs 4 agent types against all 21 patient cases directly (no server needed).
Produces a detailed report showing grader calibration, safety gate behaviour,
score distributions, and monotonicity checks.

Run from project root:
    python eval/evaluate_graders.py

Requirements: your src/ directory must be on the path.
"""

import json
import random
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Callable

# ── Path setup ──────────────────────────────────────────────────────────────
# Adjust this if your project root is different
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graders import Task1Grader, Task2Grader, Task3Grader, get_grader
from src.env import TriageEnv, TriageAction

# ── Load all patient cases directly ─────────────────────────────────────────
PATIENTS_DIR = PROJECT_ROOT / "src" / "patients"

def load_cases(difficulty: str) -> list[dict]:
    filepath = PATIENTS_DIR / f"{difficulty}_cases.json"
    with open(filepath) as f:
        return json.load(f)

EASY_CASES   = load_cases("easy")
MEDIUM_CASES = load_cases("medium")
HARD_CASES   = load_cases("hard")

TASK_MAP = {
    "task_1": ("easy",   EASY_CASES,   Task1Grader()),
    "task_2": ("medium", MEDIUM_CASES, Task2Grader()),
    "task_3": ("hard",   HARD_CASES,   Task3Grader()),
}

ALL_ASK_ACTIONS = [
    "ASK_VITALS", "ASK_SYMPTOMS", "ASK_HISTORY",
    "ASK_PAIN", "ASK_DURATION", "ASK_CONTEXT", "REQUEST_IMAGE"
]
TRIAGE_ACTIONS = ["TRIAGE_1", "TRIAGE_2", "TRIAGE_3", "TRIAGE_4", "TRIAGE_5"]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — AGENT DEFINITIONS
# Each agent is a function: (patient_case, step, revealed_info) → action_str
# ════════════════════════════════════════════════════════════════════════════

def agent_random(patient: dict, step: int, revealed: dict) -> str:
    """
    Agent 1 — Pure random.
    Picks uniformly from all 12 valid actions.
    Expected score floor: 0.08–0.15 on Task 1.
    If it scores above 0.25 consistently → grader too generous.
    Model equivalent: no model needed, pure Python.
    """
    all_actions = ALL_ASK_ACTIONS + TRIAGE_ACTIONS
    return random.choice(all_actions)


def agent_always_esi3(patient: dict, step: int, revealed: dict) -> str:
    """
    Agent 2 — Anchoring agent. Always asks 2 questions then assigns ESI 3.
    ESI 3 is the most common real ED triage level — lazy but not random.
    Tests: safety gate (Task 3 all ESI 1 → ESI 3 = 0.0 for most cases).
    Model equivalent: your existing heuristic in inference.py, simplified.
    """
    if "ASK_VITALS" not in revealed:
        return "ASK_VITALS"
    if "ASK_SYMPTOMS" not in revealed:
        return "ASK_SYMPTOMS"
    return "TRIAGE_3"


def agent_always_esi1(patient: dict, step: int, revealed: dict) -> str:
    """
    Agent 3 — Overtriage agent. Immediately assigns ESI 1 after 1 question.
    Tests: whether grader penalises overtriage on easy low-acuity cases.
    On Task 3 it accidentally gets all cases right (all true ESI 1) 
    but without probing → should score 0.5 (lucky guess cap).
    Model equivalent: no model needed.
    """
    if "ASK_VITALS" not in revealed:
        return "ASK_VITALS"
    return "TRIAGE_1"


def agent_keyword_heuristic(patient: dict, step: int, revealed: dict) -> str:
    """
    Agent 4 — Keyword heuristic. Mirrors your existing inference.py heuristic.
    This is the best non-LLM agent. Uses chief complaint keywords + vitals text.
    Tests: whether environment logic works end-to-end before you add an LLM.
    Model equivalent: your get_heuristic_action() in inference.py.
    """
    chief = patient["chief_complaint"].lower()

    # Immediate emergency keywords → ask vitals then ESI 1
    # Immediate emergency keywords → ask vitals + symptoms then ESI 1
    # Two questions before triaging ensures at least one discriminating
    # question is captured for scoring purposes
    emergency_keywords = [
        "seizure", "chest pain", "crushing", "unresponsive",
        "not breathing", "unconscious", "cardiac arrest",
        "anaphylaxis", "anaphylactic", "bee sting", "throat tight",
        "active labour", "contractions"
    ]
    if any(kw in chief for kw in emergency_keywords):
        if "ASK_VITALS" not in revealed:
            return "ASK_VITALS"
        if "ASK_SYMPTOMS" not in revealed:
            return "ASK_SYMPTOMS"
        return "TRIAGE_1"

    # Standard sequence: vitals → symptoms → history → decide
    if "ASK_VITALS" not in revealed:
        return "ASK_VITALS"
    if "ASK_SYMPTOMS" not in revealed and step < 3:
        return "ASK_SYMPTOMS"
    if "ASK_HISTORY" not in revealed and step < 4:
        return "ASK_HISTORY"

    # Read vitals to decide
    vitals = revealed.get("ASK_VITALS", "")
    symptoms = revealed.get("ASK_SYMPTOMS", "")

    critical_signs = [
        "bp 8", "bp 9", "spo2 9", "spo2 8", "hr 11", "hr 12", "hr 13",
        "hypotensive", "unresponsive", "seizing", "kussmaul",
        "diaphoretic", "clammy", "pale"
    ]
    if any(sign in vitals.lower() for sign in critical_signs):
        return "TRIAGE_1"

    urgent_signs = ["tachycardic", "febrile", "39.", "38.", "hr 10"]
    if any(sign in vitals.lower() for sign in urgent_signs):
        return "TRIAGE_2"

    minor_keywords = ["sprain", "twisted", "minor cut", "laceration", "sore throat", "runny"]
    if any(kw in chief for kw in minor_keywords):
        return "TRIAGE_4"

    return "TRIAGE_3"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EPISODE RUNNER (no server, direct grader calls)
# ════════════════════════════════════════════════════════════════════════════

def run_episode_direct(
    patient: dict,
    grader,
    agent_fn: Callable,
    max_steps: int = 6,
    seed: int = None,
) -> dict:
    """
    Run one episode directly against the grader (no HTTP server needed).
    Returns a dict with full episode details for analysis.
    """
    if seed is not None:
        random.seed(seed)

    revealed = {}
    actions_taken = []
    step = 0
    step_rewards = []

    while step < max_steps:
        action = agent_fn(patient, step, revealed)

        if action in TRIAGE_ACTIONS:
            # Terminal action
            assigned_esi = int(action[-1])
            grader_score = grader.grade(
                patient=patient,
                actions_taken=actions_taken,
                assigned_esi=assigned_esi,
            )
            actions_taken.append(action)
            return {
                "case_id": patient["id"],
                "true_esi": patient["true_esi"],
                "assigned_esi": assigned_esi,
                "grader_score": grader_score,
                "actions_taken": actions_taken,
                "step_rewards": step_rewards,
                "steps_used": step,
                "terminated_by": "triage_action",
                "probed_discriminating": bool(
                    set(actions_taken) & set(patient.get("discriminating_questions", []))
                ),
                "image_requested": "REQUEST_IMAGE" in actions_taken,
            }

        elif action in ALL_ASK_ACTIONS:
            # Information gathering — reveal answer
            is_redundant = action in revealed
            if not is_redundant:
                if action == "REQUEST_IMAGE":
                    if patient.get("image_available"):
                        revealed[action] = patient["answer_map"].get(
                            "REQUEST_IMAGE", patient.get("image_description", "")
                        )
                    else:
                        revealed[action] = "No visual evidence available for this case."
                else:
                    revealed[action] = patient["answer_map"].get(action, "")

            actions_taken.append(action)
            step += 1

            # Compute step reward (mirrors reward.py logic)
            disc_qs = set(patient.get("discriminating_questions", []))
            relevance = 0.3 if action in disc_qs else 0.0
            redundancy = -0.1 if is_redundant else 0.0
            steps_left = max_steps - step
            urgency = -0.15 if steps_left <= 2 else 0.0
            step_rewards.append(round(relevance + redundancy + urgency, 4))

        else:
            # Invalid action — treat as timeout
            break

    # Timeout — no TRIAGE action taken
    return {
        "case_id": patient["id"],
        "true_esi": patient["true_esi"],
        "assigned_esi": None,
        "grader_score": 0.0,
        "actions_taken": actions_taken,
        "step_rewards": step_rewards,
        "steps_used": step,
        "terminated_by": "timeout",
        "probed_discriminating": False,
        "image_requested": "REQUEST_IMAGE" in actions_taken,
    }


def run_agent_on_task(
    agent_fn: Callable,
    task_id: str,
    episodes_per_case: int = 5,
) -> list[dict]:
    """
    Run an agent against every case in a task, multiple episodes each.
    Returns list of episode result dicts.
    """
    _, cases, grader = TASK_MAP[task_id]
    results = []

    for case in cases:
        for ep in range(episodes_per_case):
            result = run_episode_direct(
                patient=case,
                grader=grader,
                agent_fn=agent_fn,
                seed=ep * 100,  # reproducible but varied
            )
            result["task_id"] = task_id
            result["episode"] = ep
            results.append(result)

    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ANALYSIS FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def score_distribution(scores: list[float]) -> dict:
    """Compute distribution statistics for a list of scores."""
    if not scores:
        return {}
    scores_sorted = sorted(scores)
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = variance ** 0.5

    # Bucket into deciles
    buckets = defaultdict(int)
    for s in scores:
        bucket = min(int(s * 10), 9)  # 0–9
        buckets[bucket] += 1

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "median": round(scores_sorted[n // 2], 4),
        "count": n,
        "zero_rate": round(scores.count(0.0) / n, 3),
        "perfect_rate": round(scores.count(1.0) / n, 3),
        "distribution_buckets": {
            f"{i*10}-{i*10+9}%": buckets[i] for i in range(10)
        }
    }


def print_section(title: str):
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)


def print_subsection(title: str):
    print(f"\n  ── {title} ──")


def bar(value: float, width: int = 30) -> str:
    """ASCII progress bar for score visualization."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THE 6 CALIBRATION CHECKS
# ════════════════════════════════════════════════════════════════════════════

def check_1_score_floor(results_by_agent: dict):
    """Check 1: Random agent floor should be 0.08–0.15 on Task 1."""
    print_section("CHECK 1 — Score floor (random agent)")
    print("  Random agent has no medical knowledge.")
    print("  Expected Task 1 mean: 0.08–0.15")
    print("  If higher → grader too generous")
    print("  If always 0.0 → grader has an unfair cliff\n")

    random_results = results_by_agent["random"]
    for task_id in ["task_1", "task_2", "task_3"]:
        task_scores = [r["grader_score"] for r in random_results if r["task_id"] == task_id]
        dist = score_distribution(task_scores)
        status = ""
        if task_id == "task_1":
            if 0.05 <= dist["mean"] <= 0.20:
                status = "PASS"
            elif dist["mean"] > 0.20:
                status = "WARN — grader may be too generous"
            else:
                status = "WARN — floor too low, check grader cliffs"
        print(f"  {task_id}:  mean={dist['mean']:.4f}  "
              f"zero_rate={dist['zero_rate']:.2f}  "
              f"perfect_rate={dist['perfect_rate']:.2f}  "
              f"{status}")


def check_2_safety_gate(results_by_agent: dict):
    """Check 2: ESI 3 agent on Task 3 must trigger safety gate on all ESI-1 cases."""
    print_section("CHECK 2 — Safety gate (anchoring agent on Task 3)")
    print("  Anchoring agent assigns ESI 3 on everything.")
    print("  ALL hard cases are true ESI 1.")
    print("  ESI 3 vs ESI 1 = 2-level gap → safety gate fires → score MUST be 0.0\n")

    esi3_results = results_by_agent["esi3"]
    task3_results = [r for r in esi3_results if r["task_id"] == "task_3"]

    all_zero = True
    for r in task3_results:
        esi_gap = abs((r["assigned_esi"] or 3) - r["true_esi"])
        gate_fired = r["grader_score"] == 0.0
        if r["assigned_esi"] and esi_gap >= 2 and not gate_fired:
            all_zero = False
            print(f"  FAIL: {r['case_id']} assigned ESI {r['assigned_esi']} "
                  f"vs true ESI {r['true_esi']} — score was {r['grader_score']}, expected 0.0")

    scores = [r["grader_score"] for r in task3_results]
    print(f"  Task 3 scores (ESI-3 agent): {[round(s,4) for s in set(scores)]}")
    if all_zero:
        print("  PASS — safety gate fires correctly on all ESI-1 cases")
    else:
        print("  FAIL — safety gate has a bug, fix Task3Grader immediately")


def check_3_overtriage_penalty(results_by_agent: dict):
    """Check 3: Overtriage agent should be penalised on low-acuity easy cases."""
    print_section("CHECK 3 — Overtriage penalty (always-ESI-1 agent on Task 1)")
    print("  Agent assigns ESI 1 on everything after 1 question.")
    print("  Easy cases include ESI 4 (ankle sprain) and ESI 5 (sore throat).")
    print("  ESI 1 vs ESI 4 = 3-level gap → score must be 0.0\n")

    esi1_results = results_by_agent["esi1"]
    task1_results = [r for r in esi1_results if r["task_id"] == "task_1"]

    case_scores = {}
    for r in task1_results:
        cid = r["case_id"]
        if cid not in case_scores:
            case_scores[cid] = []
        case_scores[cid].append(r["grader_score"])

    for case_id, scores in sorted(case_scores.items()):
        mean_score = sum(scores) / len(scores)
        print(f"  {case_id}: mean={mean_score:.4f}  "
              f"(assigned ESI 1 vs true ESI {task1_results[0]['true_esi']})")

    # Task 3 — overtriage agent accidentally correct (all ESI 1)
    task3_results = [r for r in esi1_results if r["task_id"] == "task_3"]
    task3_scores = [r["grader_score"] for r in task3_results]
    task3_mean = sum(task3_scores) / len(task3_scores) if task3_scores else 0
    print(f"\n  Task 3 mean (all true ESI 1, assigned ESI 1, no probing): {task3_mean:.4f}")
    print("  Expected: 0.5 (correct ESI but lucky guess — no probing)")
    if 0.45 <= task3_mean <= 0.55:
        print("  PASS — lucky guess cap working correctly")
    else:
        print(f"  WARN — expected ~0.5, got {task3_mean:.4f}. Check Task3Grader lucky guess logic.")


def check_4_determinism(results_by_agent: dict):
    """Check 4: Same actions on same case must always produce same score."""
    print_section("CHECK 4 — Determinism (same input → same output always)")
    print("  Running each case 5 times with same seed.")
    print("  All 5 scores per case must be identical.\n")

    graders = {
        "task_1": Task1Grader(),
        "task_2": Task2Grader(),
        "task_3": Task3Grader(),
    }
    failures = 0

    for task_id, (_, cases, grader) in TASK_MAP.items():
        for case in cases:
            # Fixed trajectory: ask 2 discriminating questions then TRIAGE_1
            disc_qs = case.get("discriminating_questions", ["ASK_VITALS"])
            fixed_actions = disc_qs[:2] if len(disc_qs) >= 2 else disc_qs

            scores = []
            for _ in range(5):
                s = grader.grade(
                    patient=case,
                    actions_taken=list(fixed_actions),
                    assigned_esi=case["true_esi"],
                )
                scores.append(s)

            if len(set(scores)) != 1:
                print(f"  FAIL: {case['id']} produced different scores: {scores}")
                failures += 1

    if failures == 0:
        print("  PASS — all 21 cases produce identical scores across 5 runs")
    else:
        print(f"  FAIL — {failures} cases are non-deterministic")


def check_5_monotonicity(results_by_agent: dict):
    """Check 5: On Task 1, correct+efficient > correct+wasteful > off-by-1 > off-by-2."""
    print_section("CHECK 5 — Monotonicity (Task 1 score ordering)")
    print("  For every easy case, verify score ordering:")
    print("  correct+2q > correct+5q > off-by-1 > off-by-2+\n")

    grader = Task1Grader()
    failures = 0

    for case in EASY_CASES:
        true_esi = case["true_esi"]

        # Always test in the direction that produces a genuinely wrong answer.
        # For ESI 1-3: go less urgent (higher number = less urgent).
        # For ESI 4-5: go more urgent (lower number = more urgent).
        # This prevents the min(5, esi+1) edge case where ESI 5 + 1 = 5
        # which is still the correct answer and scores 1.0 incorrectly.
        if true_esi <= 3:
            esi_off_1 = true_esi + 1
            esi_off_2 = min(5, true_esi + 2)
        else:
            esi_off_1 = true_esi - 1
            esi_off_2 = max(1, true_esi - 2)

        # Perfect trajectory: ask the actual discriminating questions for this case
        # This ensures the check tests realistic optimal behaviour, not a hardcoded
        # question sequence that may not be relevant to this specific patient.
        disc_qs = case.get("discriminating_questions", ["ASK_VITALS"])
        perfect_actions = disc_qs[:2] if len(disc_qs) >= 2 else disc_qs

        # Wasteful trajectory: ask all 5 standard questions (always includes disc_qs)
        wasteful_actions = list(dict.fromkeys(
            list(perfect_actions) + ["ASK_VITALS", "ASK_SYMPTOMS", "ASK_HISTORY"]
        ))

        s_perfect  = grader.grade(case, list(perfect_actions), true_esi)
        s_wasteful = grader.grade(case, wasteful_actions, true_esi)
        s_off1     = grader.grade(case, list(perfect_actions), esi_off_1)
        s_off2     = grader.grade(case, disc_qs[:1], esi_off_2)
        ordered = s_perfect >= s_wasteful >= s_off1 >= s_off2
        status = "PASS" if ordered else "FAIL"

        print(f"  {case['id']}: perfect={s_perfect:.4f}  wasteful={s_wasteful:.4f}  "
              f"off1={s_off1:.4f}  off2={s_off2:.4f}  {status}")

        if not ordered:
            failures += 1

    if failures == 0:
        print("\n  PASS — monotonicity holds across all easy cases")
    else:
        print(f"\n  FAIL — {failures} cases violate monotonicity, fix Task1Grader")


def check_6_boundary_values(results_by_agent: dict):
    """Check 6: Boundary behaviors — immediate triage and timeout."""
    print_section("CHECK 6 — Boundary values")
    print("  Testing: 0 questions then TRIAGE, and 6 questions then timeout\n")

    graders = {
        "task_1": Task1Grader(),
        "task_2": Task2Grader(),
        "task_3": Task3Grader(),
    }

    for task_id, (_, cases, grader) in TASK_MAP.items():
        case = cases[0]  # Test on first case of each task
        true_esi = case["true_esi"]

        # Boundary 1: immediate correct triage (0 questions)
        score_immediate = grader.grade(case, [], true_esi)

        # Boundary 2: immediate wrong triage
        wrong_esi = 5 if true_esi <= 2 else 1
        score_wrong = grader.grade(case, [], wrong_esi)

        # Boundary 3: full 6 questions then correct triage
        all_asks = ALL_ASK_ACTIONS[:6]
        score_exhaustive = grader.grade(case, all_asks, true_esi)

        print(f"  {task_id} ({case['id']}):  "
              f"immediate_correct={score_immediate:.4f}  "
              f"immediate_wrong={score_wrong:.4f}  "
              f"exhaustive_correct={score_exhaustive:.4f}")

    # Timeout check via env directly
    print("\n  Timeout check (6 ASK actions, no TRIAGE):")
    for task_id in ["task_1", "task_2", "task_3"]:
        env = TriageEnv(task_id)
        env.max_steps = 3
        env.reset()
        for action_str in ["ASK_VITALS", "ASK_SYMPTOMS", "ASK_HISTORY"]:
            result = env.step(TriageAction(action=action_str))
        assert result.done is True
        assert result.reward.total == -2.0
        assert result.reward.grader_score == 0.0
        print(f"  {task_id}: timeout score={result.reward.total}  "
              f"grader_score={result.reward.grader_score}  PASS")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCORE DISTRIBUTION REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_full_distribution_report(results_by_agent: dict):
    """Print score distributions for all agents across all tasks."""
    print_section("SCORE DISTRIBUTION REPORT")

    agent_labels = {
        "random":   "Agent 1 — Random          ",
        "esi3":     "Agent 2 — Always ESI 3    ",
        "esi1":     "Agent 3 — Always ESI 1    ",
        "heuristic":"Agent 4 — Keyword heuristic",
    }

    expected_ranges = {
        "task_1": (0.75, 0.90),
        "task_2": (0.45, 0.65),
        "task_3": (0.20, 0.40),
    }

    for task_id in ["task_1", "task_2", "task_3"]:
        lo, hi = expected_ranges[task_id]
        print(f"\n  {task_id.upper()}  (target for LLM agent: {lo}–{hi})")
        print(f"  {'Agent':<30} {'Mean':>6}  {'Std':>5}  "
              f"{'Min':>5}  {'Max':>5}  {'0.0%':>5}  {'1.0%':>5}  Bar")
        print("  " + "─" * 80)

        for agent_key, label in agent_labels.items():
            scores = [
                r["grader_score"]
                for r in results_by_agent[agent_key]
                if r["task_id"] == task_id
            ]
            dist = score_distribution(scores)
            b = bar(dist["mean"])
            in_range = ""
            if agent_key == "heuristic":
                if lo <= dist["mean"] <= hi:
                    in_range = " <-- target range"
                elif dist["mean"] > hi:
                    in_range = " <-- above target"
                else:
                    in_range = " <-- below target"

            print(f"  {label}  "
                  f"{dist['mean']:>6.4f}  "
                  f"{dist['std']:>5.4f}  "
                  f"{dist['min']:>5.4f}  "
                  f"{dist['max']:>5.4f}  "
                  f"{dist['zero_rate']:>5.2f}  "
                  f"{dist['perfect_rate']:>5.2f}  "
                  f"|{b}|{in_range}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CASE-BY-CASE BREAKDOWN (hardest cases)
# ════════════════════════════════════════════════════════════════════════════

def print_hard_case_breakdown(results_by_agent: dict):
    """Show per-case scores on Task 3 for the heuristic agent."""
    print_section("HARD CASE BREAKDOWN — Heuristic agent vs Task 3")
    print("  Shows per-case performance. Judges' LLM will do better here.")
    print("  Cases scoring > 0.4 for heuristic agent may not be hard enough.\n")

    heuristic_results = [
        r for r in results_by_agent["heuristic"]
        if r["task_id"] == "task_3"
    ]

    case_agg = defaultdict(list)
    for r in heuristic_results:
        case_agg[r["case_id"]].append(r)

    print(f"  {'Case ID':<15}  {'Mean':>6}  {'Probed%':>7}  "
          f"{'Img%':>5}  {'Actions (sample)':>30}")
    print("  " + "─" * 75)

    for case_id in sorted(case_agg.keys()):
        episodes = case_agg[case_id]
        scores = [e["grader_score"] for e in episodes]
        mean = sum(scores) / len(scores)
        probed_pct = sum(1 for e in episodes if e["probed_discriminating"]) / len(episodes)
        img_pct = sum(1 for e in episodes if e["image_requested"]) / len(episodes)
        sample_actions = episodes[0]["actions_taken"]

        flag = ""
        if mean > 0.4:
            flag = " <-- may not be hard enough"
        elif mean == 0.0:
            flag = " <-- safety gate always fires"

        print(f"  {case_id:<15}  {mean:>6.4f}  {probed_pct:>7.1%}  "
              f"{img_pct:>5.1%}  {str(sample_actions)[:30]}{flag}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FINAL VERDICT
# ════════════════════════════════════════════════════════════════════════════

def print_verdict(results_by_agent: dict):
    """Print final go/no-go verdict for submission."""
    print_section("FINAL VERDICT — Submission readiness")

    issues = []
    warnings = []

    # Check 1: Random floor
    random_t1 = [r["grader_score"] for r in results_by_agent["random"] if r["task_id"] == "task_1"]
    floor = sum(random_t1) / len(random_t1) if random_t1 else 0
    if floor > 0.25:
        issues.append(f"Task 1 random floor too high ({floor:.4f} > 0.25) — grader too generous")
    elif floor < 0.02:
        warnings.append(f"Task 1 random floor very low ({floor:.4f}) — may have harsh cliffs")

    # Check 2: Safety gate
    esi3_t3 = [r["grader_score"] for r in results_by_agent["esi3"] if r["task_id"] == "task_3"
               and r["assigned_esi"] is not None and abs(r["assigned_esi"] - r["true_esi"]) >= 2]
    if any(s > 0 for s in esi3_t3):
        issues.append("Safety gate NOT firing on catastrophic undertriage — critical bug")

    # Check 3: Heuristic in range
    expected = {"task_1": (0.55, 0.95), "task_2": (0.25, 0.75), "task_3": (0.10, 0.55)}
    for task_id, (lo, hi) in expected.items():
        scores = [r["grader_score"] for r in results_by_agent["heuristic"] if r["task_id"] == task_id]
        mean = sum(scores) / len(scores) if scores else 0
        if not (lo <= mean <= hi):
            warnings.append(f"{task_id} heuristic mean {mean:.4f} outside expected {lo}–{hi}")

    print()
    if not issues and not warnings:
        print("  READY TO SUBMIT")
        print("  All calibration checks passed.")
        print("  Grader is well-calibrated for judging evaluation.")
    elif not issues:
        print("  READY WITH WARNINGS")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print("  NOT READY — Fix these issues before submission:")
        for i in issues:
            print(f"  FAIL: {i}")
        for w in warnings:
            print(f"  WARN: {w}")

    print()
    print("  Next step: run inference.py with a real LLM to get final baseline scores.")
    print("  Target ranges:")
    print("    Task 1: 0.75–0.90")
    print("    Task 2: 0.45–0.65")
    print("    Task 3: 0.20–0.40")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    EPISODES_PER_CASE = 7  # 7 cases × 7 episodes = 49 episodes per agent per task

    print("═" * 65)
    print("  MedTriage-RL — Grader Evaluation Suite")
    print(f"  {EPISODES_PER_CASE} episodes × 7 cases × 3 tasks × 4 agents = "
          f"{EPISODES_PER_CASE * 7 * 3 * 4} total episodes")
    print("═" * 65)

    agents = {
        "random":    agent_random,
        "esi3":      agent_always_esi3,
        "esi1":      agent_always_esi1,
        "heuristic": agent_keyword_heuristic,
    }

    results_by_agent = {}
    for agent_name, agent_fn in agents.items():
        print(f"\n  Running {agent_name} agent...", end="", flush=True)
        all_results = []
        for task_id in ["task_1", "task_2", "task_3"]:
            task_results = run_agent_on_task(agent_fn, task_id, EPISODES_PER_CASE)
            all_results.extend(task_results)
        results_by_agent[agent_name] = all_results
        total_episodes = len(all_results)
        mean_score = sum(r["grader_score"] for r in all_results) / total_episodes
        print(f" done. {total_episodes} episodes, overall mean={mean_score:.4f}")

    # Run all 6 checks
    check_1_score_floor(results_by_agent)
    check_2_safety_gate(results_by_agent)
    check_3_overtriage_penalty(results_by_agent)
    check_4_determinism(results_by_agent)
    check_5_monotonicity(results_by_agent)
    check_6_boundary_values(results_by_agent)

    # Full distribution report
    print_full_distribution_report(results_by_agent)

    # Hard case breakdown
    print_hard_case_breakdown(results_by_agent)

    # Final verdict
    print_verdict(results_by_agent)


if __name__ == "__main__":
    main()