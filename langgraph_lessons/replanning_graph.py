
"""
LESSON 19 - PART 8: DYNAMIC REPLANNING
=====================================

Concepts:
- LLM-generated execution plans
- Plan validation
- Feedback-driven replanning
- Dynamic worker dispatch with Send()
- Handling failed worker results
- Bounded planning attempts
- Separating current results from old attempts

RUN:
python -m langgraph_lessons.replanning_graph normal
python -m langgraph_lessons.replanning_graph bad_plan
python -m langgraph_lessons.replanning_graph worker_failure
"""

import operator
import sys

from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# --------------------------------------------------
# STEP 1: BUSINESS DATA AND EXECUTION POLICY
# --------------------------------------------------

CLAIMS = [
    ("CLM1001", "Under Review", 4500.0),
    ("CLM1002", "Approved", 850.0),
    ("CLM1003", "Rejected", 2200.0),
    ("CLM1004", "Approved", 3200.0),
    ("CLM1005", "Approved", 1800.0),
]

ALLOWED_STATUSES = {
    "Approved",
    "Under Review",
    "Rejected",
}

MAX_WORKERS = 3
MAX_PLAN_ATTEMPTS = 3


# --------------------------------------------------
# STEP 2: PYDANTIC PLAN SCHEMA
# --------------------------------------------------

class AnalysisTask(BaseModel):
    status: Literal[
        "Approved",
        "Under Review",
        "Rejected"
    ]
    reason: str = Field(min_length=1)


class AnalysisPlan(BaseModel):
    tasks: list[AnalysisTask] = Field(
        min_length=1,
        max_length=MAX_WORKERS
    )
    explanation: str = Field(min_length=1)


# --------------------------------------------------
# STEP 3: SHARED GRAPH STATE
# --------------------------------------------------

class PlannerState(TypedDict):

    goal: str
    required_statuses: list[str]
    min_amount: float

    plan: list[dict]
    plan_attempts: int
    plan_status: str

    feedback: str

    # Results accumulate across planning attempts.
    findings: Annotated[list[dict], operator.add]

    report: dict

    # These flags are ONLY for lesson experiments.
    scenario: str


# --------------------------------------------------
# STEP 4: TASK-SPECIFIC WORKER STATE
# --------------------------------------------------

class WorkerState(TypedDict):
    status: str
    min_amount: float
    attempt: int
    scenario: str


# --------------------------------------------------
# STEP 5: QWEN2 STRUCTURED PLANNER
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)

planner = model.with_structured_output(
    AnalysisPlan,
    method="json_schema"
)


# --------------------------------------------------
# STEP 6: GENERATE OR REVISE A PLAN
# --------------------------------------------------

def generate_plan(state: PlannerState) -> dict:

    attempt = state["plan_attempts"] + 1

    print(f"\n[NODE] generate_plan - Attempt {attempt}")

    messages = [
        (
            "system",
            "You are an insurance analysis planner. "
            "Return executable tasks only. "
            "Permitted statuses: Approved, "
            "Under Review, Rejected. "
            "The user requirements have already "
            "been validated by the application. "
            "Create exactly one analysis task for "
            "each required status. "
            "Do not add excluded categories. "
            "Use the feedback to correct errors. "
            "You are planning, not executing."
        ),
        (
            "human",
            (
                f"Goal: {state['goal']}\n"
                f"Required statuses: "
                f"{state['required_statuses']}\n"
                f"Minimum amount: "
                f"{state['min_amount']}\n"
                f"Previous feedback: "
                f"{state['feedback'] or 'None'}"
            )
        )
    ]

    try:
        plan = planner.invoke(messages)

        if not isinstance(plan, AnalysisPlan):
            raise ValueError(
                "LLM returned an unexpected plan type"
            )

        tasks = [
            task.model_dump()
            for task in plan.tasks
        ]

        # Controlled demonstration:
        # Deliberately corrupt the first proposed
        # plan so validation and replanning can
        # be observed reliably.
        if (
            state["scenario"] == "bad_plan"
            and attempt == 1
        ):
            tasks = [
                {
                    "status": "Approved",
                    "reason": "Required",
                },
                {
                    "status": "Under Review",
                    "reason": "Required",
                },
                {
                    "status": "Rejected",
                    "reason": "Incorrect extra task",
                },
            ]

            print(
                "[SIMULATION] Added an invalid task "
                "to the first plan"
            )

        print("Proposed tasks:", tasks)

        return {
            "plan": tasks,
            "plan_attempts": attempt,
            "plan_status": "PROPOSED",
        }

    except Exception as exc:

        print("Planning error:", exc)

        return {
            "plan": [],
            "plan_attempts": attempt,
            "plan_status": "INVALID",
            "feedback": (
                "The planner failed to produce a "
                f"valid structured response: {exc}"
            ),
        }


# --------------------------------------------------
# STEP 7: VALIDATE PLAN AGAINST USER REQUIREMENTS
# --------------------------------------------------

def validate_plan(state: PlannerState) -> dict:

    print("\n[NODE] validate_plan")

    if state["plan_status"] != "PROPOSED":
        return {}

    tasks = state["plan"]

    planned = [
        task["status"]
        for task in tasks
    ]

    required = state["required_statuses"]

    # First validate trusted application inputs.
    if (
        not 1 <= len(required) <= MAX_WORKERS
        or len(set(required)) != len(required)
        or not set(required).issubset(ALLOWED_STATUSES)
        or not (0 <= state["min_amount"] < float("inf"))
    ):
        return {
            "plan_status": "INVALID",
            "feedback": "Invalid trusted request inputs"
        }

    if (
        not 1 <= len(tasks) <= MAX_WORKERS
        or len(planned) != len(set(planned))
        or set(planned) != set(required)
    ):
        feedback = (
            "Your proposed tasks do not match "
            "the required categories. "
            f"Required: {required}. "
            f"Proposed: {planned}. "
            "Return exactly the required statuses, "
            "without duplicates or extra tasks."
        )

        print("VALIDATION FAILED:", feedback)

        return {
            "plan_status": "INVALID",
            "feedback": feedback,
        }

    print("Plan validated:", planned)

    return {
        "plan_status": "VALIDATED",
        "feedback": "",
    }


# --------------------------------------------------
# STEP 8: ROUTE AFTER VALIDATION
# --------------------------------------------------

def route_after_validation(state: PlannerState):

    if state["plan_status"] == "VALIDATED":

        return [
            Send(
                "analyse_claims",
                {
                    "status": task["status"],
                    "min_amount": state["min_amount"],
                    "attempt": state["plan_attempts"],
                    "scenario": state["scenario"],
                }
            )
            for task in state["plan"]
        ]

    if state["plan_attempts"] < MAX_PLAN_ATTEMPTS:
        print("[ROUTER] Invalid plan -> replan")
        return "generate_plan"

    print("[ROUTER] Planning budget exhausted")
    return "finish_failure"


# --------------------------------------------------
# STEP 9: REUSABLE WORKER NODE
# --------------------------------------------------

def analyse_claims(state: WorkerState) -> dict:

    status = state["status"]
    attempt = state["attempt"]

    print(
        f"\n[WORKER] {status}, "
        f"planning attempt {attempt}"
    )

    try:

        # Controlled demonstration:
        # Under Review fails once, but succeeds
        # when the task is executed again.
        if (
            state["scenario"] == "worker_failure"
            and status == "Under Review"
            and attempt == 1
        ):
            raise ConnectionError(
                "Simulated temporary data-source failure"
            )

        matching = [
            claim
            for claim in CLAIMS
            if claim[1] == status
            and claim[2] >= state["min_amount"]
        ]

        finding = {
            "status": status,
            "attempt": attempt,
            "success": True,
            "claim_ids": [
                claim[0] for claim in matching
            ],
            "claim_count": len(matching),
            "total_amount": round(
                sum(claim[2] for claim in matching),
                2
            ),
        }

    except ConnectionError as exc:

        # Convert a known transient failure into
        # a structured result for the graph.
        finding = {
            "status": status,
            "attempt": attempt,
            "success": False,
            "error": str(exc),
        }

    return {
        "findings": [finding]
    }


# --------------------------------------------------
# STEP 10: ASSESS WORKER RESULTS
# --------------------------------------------------

def assess_results(state: PlannerState) -> dict:

    print("\n[NODE] assess_results")

    attempt = state["plan_attempts"]

    # VERY IMPORTANT:
    # Only examine results from the current
    # planning attempt.
    current = [
        finding
        for finding in state["findings"]
        if finding["attempt"] == attempt
    ]

    expected = {
        task["status"]
        for task in state["plan"]
    }

    actual = [
        item["status"]
        for item in current
    ]

    if (
        len(actual) != len(expected)
        or set(actual) != expected
    ):
        return {
            "plan_status": "EXECUTION_FAILED",
            "feedback": (
                "Some planned tasks did not return "
                "exactly one result. Retry the "
                "required analysis."
            ),
        }

    errors = [
        item
        for item in current
        if not item["success"]
    ]

    if errors:

        error_summary = [
            {
                "status": item["status"],
                "error": item["error"],
            }
            for item in errors
        ]

        print("Worker failures:", error_summary)

        return {
            "plan_status": "EXECUTION_FAILED",
            "feedback": (
                "The previous plan was valid, but "
                f"execution encountered: {error_summary}. "
                "Recreate the required tasks so they "
                "can be retried."
            ),
        }

    print("All current worker results succeeded")

    return {
        "plan_status": "COMPLETED",
        "feedback": "",
    }


# --------------------------------------------------
# STEP 11: ROUTE AFTER RESULT ASSESSMENT
# --------------------------------------------------

def route_after_assessment(state: PlannerState) -> str:

    if state["plan_status"] == "COMPLETED":
        return "finish_success"

    if state["plan_attempts"] < MAX_PLAN_ATTEMPTS:
        print("[ROUTER] Worker failure -> replan")
        return "generate_plan"

    print("[ROUTER] Recovery budget exhausted")
    return "finish_failure"


# --------------------------------------------------
# STEP 12: BUILD SUCCESS REPORT
# --------------------------------------------------

def finish_success(state: PlannerState) -> dict:

    attempt = state["plan_attempts"]

    current = sorted(
        [
            item
            for item in state["findings"]
            if item["attempt"] == attempt
        ],
        key=lambda item: item["status"],
    )

    total = round(
        sum(
            item["total_amount"]
            for item in current
        ),
        2
    )

    return {
        "report": {
            "status": "SUCCESS",
            "planning_attempts": attempt,
            "findings": current,
            "combined_total": total,
        }
    }


# --------------------------------------------------
# STEP 13: BUILD FAILURE REPORT
# --------------------------------------------------

def finish_failure(state: PlannerState) -> dict:

    return {
        "report": {
            "status": "FAILED",
            "planning_attempts": state["plan_attempts"],
            "reason": state["feedback"],
        }
    }


# --------------------------------------------------
# STEP 14: CONSTRUCT THE GRAPH
# --------------------------------------------------

builder = StateGraph(PlannerState)

builder.add_node("generate_plan", generate_plan)
builder.add_node("validate_plan", validate_plan)
builder.add_node("analyse_claims", analyse_claims)
builder.add_node("assess_results", assess_results)
builder.add_node("finish_success", finish_success)
builder.add_node("finish_failure", finish_failure)

builder.add_edge(START, "generate_plan")

builder.add_edge(
    "generate_plan",
    "validate_plan"
)

builder.add_conditional_edges(
    "validate_plan",
    route_after_validation,
    [
        "generate_plan",
        "analyse_claims",
        "finish_failure",
    ]
)

# After dispatched workers finish, assess results.
builder.add_edge(
    "analyse_claims",
    "assess_results"
)

builder.add_conditional_edges(
    "assess_results",
    route_after_assessment,
    [
        "generate_plan",
        "finish_success",
        "finish_failure",
    ]
)

builder.add_edge("finish_success", END)
builder.add_edge("finish_failure", END)

graph = builder.compile()


# --------------------------------------------------
# STEP 15: RUN AN EXPERIMENT
# --------------------------------------------------

def main():

    scenario = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "normal"
    )

    if scenario not in {
        "normal",
        "bad_plan",
        "worker_failure",
    }:
        raise SystemExit(
            "Use: normal, bad_plan, or worker_failure"
        )

    initial_state: PlannerState = {
        "goal": (
            "Compare Approved and Under Review "
            "insurance claims worth at least $1000."
        ),
        "required_statuses": [
            "Approved",
            "Under Review",
        ],
        "min_amount": 1000.0,

        "plan": [],
        "plan_attempts": 0,
        "plan_status": "PENDING",

        "feedback": "",
        "findings": [],
        "report": {},

        "scenario": scenario,
    }

    print("\nSCENARIO:", scenario)

    result = graph.invoke(
        initial_state,
        config={
            "max_concurrency": MAX_WORKERS,
            "recursion_limit": 40,
        },
    )

    print("\n========== FINAL REPORT ==========")

    report = result["report"]

    print("Status:", report["status"])
    print(
        "Planning attempts:",
        report["planning_attempts"]
    )

    if report["status"] == "SUCCESS":

        for item in report["findings"]:
            print(
                item["status"],
                "->",
                item["total_amount"]
            )

        print(
            "Combined total:",
            report["combined_total"]
        )

    else:

        print("Reason:", report["reason"])

    print("\n========== ALL WORKER HISTORY ==========")

    for item in result["findings"]:
        print(item)


if __name__ == "__main__":
    main()
