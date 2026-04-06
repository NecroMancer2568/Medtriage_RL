import requests

ENV_URL = "http://localhost:7860"
TASK = "task_1"

def run_manual_test(name, action_sequence):
    print(f"\n--- Testing: {name} ---")
    # 1. Reset
    requests.post(f"{ENV_URL}/reset", params={"task_id": "task_1", "case_id": "easy_001"})
    
    # 2. Execute information gathering actions
    for action in action_sequence[:-1]:
        requests.post(f"{ENV_URL}/step", params={"task_id": TASK}, json={"action": action})
        print(f"Action: {action}")
    
    # 3. Execute final Triage action
    final_action = action_sequence[-1]
    resp = requests.post(f"{ENV_URL}/step", params={"task_id": TASK}, json={"action": final_action})
    result = resp.json()
    
    score = result["reward"]["grader_score"]
    print(f"Final Triage: {final_action}")
    print(f"RESULTING SCORE: {score}")
    true_esi = result["info"]["true_esi"] # Or check the patient meta
    print(f"Agent assigned 1 | True ESI was: {true_esi}")
    return score

# --- THE AUDIT SUITE ---

# 1. Test "The 1.0 Perfect Path" (Fast & Accurate)
run_manual_test("Perfect Fast Agent", ["ASK_VITALS", "TRIAGE_1"]) 
# (Assumes case is ESI 1. Adjust triage number based on the case you see in terminal)

# 2. Test "The 0.70 Shotgunner" (Slow & Accurate)
run_manual_test("Shotgunner Agent", [
    "ASK_VITALS", "ASK_SYMPTOMS", "ASK_HISTORY", 
    "ASK_PAIN", "ASK_DURATION", "TRIAGE_1"
])

# 3. Test "The 0.50 Near Miss" (Fast but slightly wrong)
run_manual_test("Near Miss Agent", ["ASK_VITALS", "TRIAGE_2"])