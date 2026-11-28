
"""
LESSON 19 - PART 9
Multi-specialist orchestration and budgets

RUN:
python -m langgraph_lessons.multi_agent_budget full
python -m langgraph_lessons.multi_agent_budget focused
python -m langgraph_lessons.multi_agent_budget tight

The specialists are deterministic Python workers.
The coordinator uses local Qwen2.
"""

import asyncio
import operator
import sys
import time

from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field
active_workers = 0
peak_workers = 0

start_time = time.perf_counter()

# ---------------------------------------------
# 1. BUSINESS DATA
# ---------------------------------------------

CLAIMS = [
    {"id": "CLM1001", "status": "Under Review", "amount": 4500},
    {"id": "CLM1002", "status": "Approved", "amount": 850},
    {"id": "CLM1003", "status": "Rejected", "amount": 2200},
    {"id": "CLM1004", "status": "Approved", "amount": 3200},
    {"id": "CLM1005", "status": "Approved", "amount": 1800},
]

ROLE_COSTS = {
    "exposure": 2,
    "quality": 1,
    "exceptions": 1,
}

MAX_WORKERS = 3
MAX_CONCURRENT_WORKERS = 2


# ---------------------------------------------
# 2. STRUCTURED PLANNING OUTPUT
# ---------------------------------------------

class SpecialistTask(BaseModel):

    role: Literal[
        "exposure",
        "quality",
        "exceptions"
    ]

    reason: str = Field(min_length=1)


class SpecialistPlan(BaseModel):

    tasks: list[SpecialistTask] = Field(
        min_length=1,
        max_length=MAX_WORKERS
    )


# ---------------------------------------------
# 3. SHARED STATE
# ---------------------------------------------

class TeamState(TypedDict):

    goal: str
    required_roles: list[str]
    budget_limit: int

    plan: list[dict]
    plan_status: str
    error: str

    reserved_units: int

    results: Annotated[list[dict], operator.add]

    report: dict


# ---------------------------------------------
# 4. INPUT FOR EACH SPECIALIST
# ---------------------------------------------

class WorkerState(TypedDict):

    role: str
    task_id: str


# ---------------------------------------------
# 5. QWEN2 COORDINATOR
# ---------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)

planner = model.with_structured_output(
    SpecialistPlan,
    method="json_schema"
)


def create_team_plan(state: TeamState) -> dict:

    print("\n[COORDINATOR] Creating specialist plan")

    try:
        plan = planner.invoke([
            (
                "system",
                "You are a task coordinator. "
                "Available specialists are: "
                "exposure, quality, exceptions. "
                "Exposure calculates claim exposure. "
                "Quality checks data validity. "
                "Exceptions identifies large claims. "
                "Return exactly one task per required role. "
                "Do not add unrequested specialists. "
                "Do not invent roles or permissions. "
                "Return a structured SpecialistPlan."
            ),
            (
                "human",
                f"Goal: {state['goal']}\n"
                f"Required roles: {state['required_roles']}"
            )
        ])

        if not isinstance(plan, SpecialistPlan):
            raise ValueError("Unexpected planner output")

        tasks = [
            task.model_dump()
            for task in plan.tasks
        ]

        print("LLM proposed:", tasks)

        return {
            "plan": tasks,
            "plan_status": "PROPOSED",
            "error": "",
        }

    except Exception as exc:
        return {
            "plan": [],
            "plan_status": "REJECTED",
            "error": f"Planning error: {exc}",
        }


# ---------------------------------------------
# 6. VALIDATE AND RESERVE BUDGET
# ---------------------------------------------

def validate_budget(state: TeamState) -> dict:

    print("\n[COORDINATOR] Validating plan and budget")

    if state["plan_status"] != "PROPOSED":
        return {}

    required = state["required_roles"]
    planned = [
        task["role"] for task in state["plan"]
    ]

    if (
        not 1 <= len(required) <= MAX_WORKERS
        or len(required) != len(set(required))
        or not set(required).issubset(ROLE_COSTS)
    ):
        return {
            "plan_status": "REJECTED",
            "error": "Invalid required specialist roles",
        }

    if (
        len(planned) != len(set(planned))
        or set(planned) != set(required)
        or not set(planned).issubset(ROLE_COSTS)
    ):
        return {
            "plan_status": "REJECTED",
            "error": (
                "Plan does not match required roles. "
                f"Required: {required}; planned: {planned}"
            ),
        }

    total_cost = sum(
        ROLE_COSTS[role]
        for role in planned
    )

    print("Requested units:", total_cost)
    print("Available units:", state["budget_limit"])

    if total_cost > state["budget_limit"]:
        return {
            "plan_status": "REJECTED",
            "error": (
                f"Budget exceeded: need {total_cost}, "
                f"have {state['budget_limit']}"
            ),
        }

    return {
        "plan_status": "APPROVED",
        "reserved_units": total_cost,
        "error": "",
    }


# ---------------------------------------------
# 7. DISPATCH ONLY APPROVED TASKS
# ---------------------------------------------

def dispatch_specialists(state: TeamState):

    if state["plan_status"] != "APPROVED":
        print("[DISPATCH] Blocked by validation")
        return "finalize"

    print(
        "[DISPATCH] Starting",
        len(state["plan"]),
        "specialist tasks"
    )

    return [
        Send(
            "run_specialist",
            {
                "role": task["role"],
                "task_id": f"task-{index + 1}",
            }
        )
        for index, task in enumerate(state["plan"])
    ]


# ---------------------------------------------
# 8. THREE SPECIALIST IMPLEMENTATIONS
# ---------------------------------------------

def exposure_specialist() -> dict:

    approved = sum(
        claim["amount"]
        for claim in CLAIMS
        if claim["status"] == "Approved"
    )

    pending = sum(
        claim["amount"]
        for claim in CLAIMS
        if claim["status"] == "Under Review"
    )

    return {
        "approved_amount": approved,
        "pending_amount": pending,
        "combined_amount": approved + pending,
    }


def quality_specialist() -> dict:

    invalid_records = [
        claim["id"]
        for claim in CLAIMS
        if (
            claim["amount"] < 0
            or not claim["id"]
            or claim["status"] not in {
                "Approved",
                "Under Review",
                "Rejected"
            }
        )
    ]

    return {
        "checked_records": len(CLAIMS),
        "invalid_records": invalid_records,
    }


def exceptions_specialist() -> dict:

    threshold = 3000

    large_claims = [
        claim["id"]
        for claim in CLAIMS
        if claim["amount"] > threshold
    ]

    return {
        "threshold": threshold,
        "large_claim_ids": large_claims,
    }


# ---------------------------------------------
# 9. DYNAMIC SPECIALIST EXECUTOR
# ---------------------------------------------

SPECIALIST_HANDLERS = {
    "exposure": exposure_specialist,
    "quality": quality_specialist,
    "exceptions": exceptions_specialist,
}



async def run_specialist(state: WorkerState) -> dict:

    global active_workers, peak_workers

    role = state["role"]
    task_id = state["task_id"]

    active_workers += 1

    peak_workers = max(
        peak_workers,
        active_workers
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"[{elapsed:.2f}s] START {role} "
        f"| Active workers: {active_workers}"
    )

    try:
        # Longer delay to make concurrency obvious.
        await asyncio.sleep(2)

        output = SPECIALIST_HANDLERS[role]()

        result = {
            "task_id": task_id,
            "role": role,
            "success": True,
            "output": output,
        }

    except Exception as exc:

        result = {
            "task_id": task_id,
            "role": role,
            "success": False,
            "error": str(exc),
        }

    finally:
        active_workers -= 1

        elapsed = time.perf_counter() - start_time

        print(
            f"[{elapsed:.2f}s] END {role} "
            f"| Active workers: {active_workers}"
        )

    return {
        "results": [result]
    }



# ---------------------------------------------
# 10. VERIFY AND FINALIZE
# ---------------------------------------------

def finalize(state: TeamState) -> dict:

    print("\n[COORDINATOR] Finalizing")

    if state["plan_status"] != "APPROVED":
        return {
            "report": {
                "status": "FAILED",
                "reason": state["error"],
                "units_reserved": 0,
            }
        }

    expected = [
        task["role"]
        for task in state["plan"]
    ]

    received = [
        result["role"]
        for result in state["results"]
    ]

    valid_results = (
        len(received) == len(expected)
        and set(received) == set(expected)
        and all(
            result["success"]
            for result in state["results"]
        )
    )

    if not valid_results:
        return {
            "report": {
                "status": "FAILED",
                "reason": "Specialist execution incomplete",
                "units_reserved": state["reserved_units"],
            }
        }

    return {
        "report": {
            "status": "SUCCESS",
            "units_reserved": state["reserved_units"],
            "units_remaining": (
                state["budget_limit"]
                - state["reserved_units"]
            ),
            "specialist_results": sorted(
                state["results"],
                key=lambda r: r["role"]
            ),
        }
    }


# ---------------------------------------------
# 11. BUILD LANGGRAPH
# ---------------------------------------------

builder = StateGraph(TeamState)

builder.add_node("create_team_plan", create_team_plan)
builder.add_node("validate_budget", validate_budget)
builder.add_node("run_specialist", run_specialist)
builder.add_node("finalize", finalize)

builder.add_edge(START, "create_team_plan")
builder.add_edge("create_team_plan", "validate_budget")

builder.add_conditional_edges(
    "validate_budget",
    dispatch_specialists,
    ["run_specialist", "finalize"]
)

builder.add_edge("run_specialist", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()


# ---------------------------------------------
# 12. RUN EXPERIMENTS
# ---------------------------------------------

async def main():

    scenario = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "full"
    )

    scenarios = {
        "full": {
            "required_roles": [
                "exposure",
                "quality",
                "exceptions",
            ],
            "budget_limit": 4,
        },
        "focused": {
            "required_roles": ["exposure"],
            "budget_limit": 2,
        },
        "tight": {
            "required_roles": [
                "exposure",
                "quality",
                "exceptions",
            ],
            "budget_limit": 3,
        },
    }

    if scenario not in scenarios:
        raise SystemExit(
            "Choose: full, focused or tight"
        )

    settings = scenarios[scenario]

    initial_state: TeamState = {
        "goal": (
            "Review insurance claim exposure, "
            "data quality and large-claim exceptions."
        ),
        "required_roles": settings["required_roles"],
        "budget_limit": settings["budget_limit"],
        "plan": [],
        "plan_status": "PENDING",
        "error": "",
        "reserved_units": 0,
        "results": [],
        "report": {},
    }

    print("\nSCENARIO:", scenario)

    result = await graph.ainvoke(
        initial_state,
        config={
            "max_concurrency": MAX_CONCURRENT_WORKERS,
            "recursion_limit": 20,
        }
    )
    print("\nPeak concurrent workers:", peak_workers)

    print("\n========== FINAL REPORT ==========")

    report = result["report"]

    print("Status:", report["status"])

    if report["status"] == "FAILED":
        print("Reason:", report["reason"])
        return

    print("Budget reserved:", report["units_reserved"])
    print("Budget remaining:", report["units_remaining"])

    for item in report["specialist_results"]:
        print(
            item["role"],
            "->",
            item["output"]
        )


if __name__ == "__main__":
    asyncio.run(main())
