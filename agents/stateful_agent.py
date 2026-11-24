
"""
LESSON 17 - PART 4: AGENT STATE AND WORKING MEMORY

GOAL:
Save agent progress and resume after interruption.

FIRST RUN:
SIMULATE_INTERRUPTION = True
- Generates a plan
- Retrieves claims
- Saves a checkpoint
- Pauses before calculating the total

SECOND RUN:
SIMULATE_INTERRUPTION = False
- Loads the checkpoint
- Skips completed tasks
- Calculates the total
- Generates the summary
- Verifies completion

RUN:
python -m agents.stateful_agent
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from agents.llm_planning_agent import (
    GOAL,
    generate_plan,
    validate_plan,
    find_claims,
    calculate_total,
    generate_summary,
    verify_goal,
)


# --------------------------------------------------
# STEP 1: CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "agent_checkpoint.json"
)

# Set False after your first run.
SIMULATE_INTERRUPTION = True


# --------------------------------------------------
# STEP 2: SAVE AGENT STATE
# --------------------------------------------------

def save_checkpoint(state: dict) -> None:
    """
    Persist the full agent state to disk.

    Write to a temporary file first, then replace
    the checkpoint to reduce partial-file risks.
    """

    temp_path = CHECKPOINT_PATH.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )

    temp_path.replace(CHECKPOINT_PATH)

    print("[CHECKPOINT] Agent state saved")


# --------------------------------------------------
# STEP 3: LOAD PREVIOUS STATE
# --------------------------------------------------

def load_checkpoint() -> dict | None:

    if not CHECKPOINT_PATH.exists():
        return None

    state = json.loads(
        CHECKPOINT_PATH.read_text(
            encoding="utf-8"
        )
    )

    print("[MEMORY] Previous agent state loaded")

    return state


# --------------------------------------------------
# STEP 4: RECORD AGENT EVENTS
# --------------------------------------------------

def record_event(
    state: dict,
    action: str,
    details: str
) -> None:
    """
    Store an event in execution history.

    This is an audit-style history of actions,
    not hidden LLM reasoning.
    """

    state["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "details": details
    })


# --------------------------------------------------
# STEP 5: CREATE NEW AGENT STATE
# --------------------------------------------------

def create_new_state() -> dict:

    plan = generate_plan(GOAL)

    validate_plan(plan)

    print("[PLANNER] Generated plan approved")

    state = {
        "goal": GOAL,
        "status": "in_progress",
        "plan": [
            {
                **step.model_dump(),
                "status": "pending"
            }
            for step in plan.steps
        ],
        "results": {},
        "errors": [],
        "history": []
    }

    record_event(
        state,
        "plan_created",
        "Validated execution plan"
    )

    save_checkpoint(state)

    return state


# --------------------------------------------------
# STEP 6: EXECUTE OR RESUME
# --------------------------------------------------

def run_agent() -> dict:

    state = load_checkpoint()

    if state is None:
        state = create_new_state()
    else:
        # Checkpoints are trusted local exercise data.
        # Validate the stored goal and plan before
        # using them to resume execution.
        if state.get("goal") != GOAL:
            raise ValueError(
                "Checkpoint belongs to another goal"
            )

        from agents.llm_planning_agent import AgentPlan

        restored_plan = AgentPlan.model_validate({
            "steps": [
                {
                    "id": step["id"],
                    "action": step["action"],
                    "depends_on": step["depends_on"]
                }
                for step in state["plan"]
            ]
        })

        validate_plan(restored_plan)

    if state["status"] == "completed":
        print("[AGENT] Goal already completed")
        return state

    completed_ids = {
        step["id"]
        for step in state["plan"]
        if step["status"] == "completed"
    }

    for step in state["plan"]:

        # Skip finished tasks during a resume.
        if step["status"] == "completed":
            print(
                f"[MEMORY] Skipping completed: "
                f"{step['action']}"
            )
            continue

        required = set(step["depends_on"])

        if not required.issubset(completed_ids):
            raise RuntimeError(
                f"Unmet dependencies: {step['action']}"
            )

        action = step["action"]

        print(f"\n[EXECUTOR] Running {action}")

        step["status"] = "in_progress"

        record_event(
            state,
            "task_started",
            action
        )

        save_checkpoint(state)

        # Execute only approved operations.
        if action == "find_claims":

            state["results"]["claims"] = find_claims()

        elif action == "calculate_total":

            state["results"]["total"] = calculate_total(
                state["results"]["claims"]
            )

        elif action == "generate_summary":

            state["results"]["summary"] = generate_summary(
                state["results"]["claims"],
                state["results"]["total"]
            )

        else:
            raise ValueError("Unauthorized action")

        step["status"] = "completed"
        completed_ids.add(step["id"])

        record_event(
            state,
            "task_completed",
            action
        )

        save_checkpoint(state)

        # ------------------------------------------
        # SIMULATE INTERRUPTION AFTER FIRST TASK
        # ------------------------------------------

        if (
            SIMULATE_INTERRUPTION
            and action == "find_claims"
        ):
            state["status"] = "paused"

            record_event(
                state,
                "agent_paused",
                "Simulated interruption"
            )

            save_checkpoint(state)

            print("\n[AGENT] Execution paused")
            return state

    # ----------------------------------------------
    # FINAL GOAL VERIFICATION
    # ----------------------------------------------

    if verify_goal(state):
        state["status"] = "completed"

        record_event(
            state,
            "goal_completed",
            "All verification checks passed"
        )
    else:
        state["status"] = "failed"
        state["errors"].append(
            "Goal verification failed"
        )

    save_checkpoint(state)

    return state


# --------------------------------------------------
# STEP 7: RUN
# --------------------------------------------------

if __name__ == "__main__":

    final_state = run_agent()

    print("\n========== AGENT STATUS ==========")
    print(final_state["status"])

    print("\n========== PLAN ==========")
    print(json.dumps(final_state["plan"], indent=2))

    print("\n========== RESULTS ==========")
    print(json.dumps(final_state["results"], indent=2))

    print("\n========== EXECUTION HISTORY ==========")
    print(json.dumps(final_state["history"], indent=2))
