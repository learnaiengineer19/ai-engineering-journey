
"""
LESSON 19 - PART 7: LLM-DRIVEN PLANNING
======================================

FLOW:
User goal
  -> Qwen2 generates a structured plan
  -> Python validates the plan
  -> LangGraph dispatches dynamic workers
  -> Workers calculate claim exposures
  -> Results are combined into a report

RUN:
python -m langgraph_lessons.llm_planning_graph
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError


# --------------------------------------------------
# STEP 1: SAMPLE DATA AND POLICY
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
MIN_AMOUNT = 1000.0


# --------------------------------------------------
# STEP 2: STRUCTURED PLAN SCHEMA
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
# STEP 3: DEFINE MAIN GRAPH STATE
# --------------------------------------------------

class PlannerState(TypedDict):

    goal: str

    # Stored as dictionaries after validation.
    plan: list[dict]

    plan_status: str
    validation_error: str

    # Parallel worker results accumulate here.
    findings: Annotated[list[dict], operator.add]

    report: dict


# --------------------------------------------------
# STEP 4: DEFINE WORKER-SPECIFIC INPUT
# --------------------------------------------------

class WorkerState(TypedDict):

    status: str
    min_amount: float


# --------------------------------------------------
# STEP 5: INITIALIZE QWEN2
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)

# The LLM will return an AnalysisPlan object.
structured_planner = model.with_structured_output(
    AnalysisPlan,
    method="json_schema"
)


# --------------------------------------------------
# STEP 6: LLM PLANNING NODE
# --------------------------------------------------

def generate_plan(state: PlannerState) -> dict:

    print("\n[NODE] generate_plan")
    print("Goal:", state["goal"])

    messages = [
        (
            "system",
            "You are an insurance portfolio planner. "
            "Your job is to propose analysis tasks, "
            "not execute them. "
            "Available claim statuses are: "
            "Approved, Under Review, Rejected. "
            "Select only categories relevant to the goal. "
            "Use at most three tasks. "
            "Do not repeat a category. "
            "Do not invent tools or claim data. "
            "Return a structured AnalysisPlan."
        ),
        (
            "human",
            state["goal"]
        )
    ]

    try:
        plan = structured_planner.invoke(messages)

        if not isinstance(plan, AnalysisPlan):
            raise ValueError(
                "Model did not return an AnalysisPlan"
            )

        print("\nLLM PLAN:")
        print(plan.model_dump_json(indent=2))

        return {
            "plan": [
                task.model_dump()
                for task in plan.tasks
            ],
            "plan_status": "PROPOSED",
            "validation_error": "",
        }

    except (ValidationError, ValueError, Exception) as exc:

        # Fail closed rather than invent a fallback plan.
        return {
            "plan": [],
            "plan_status": "REJECTED",
            "validation_error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


# --------------------------------------------------
# STEP 7: VALIDATE THE MODEL'S PLAN
# --------------------------------------------------

def validate_plan(state: PlannerState) -> dict:

    print("\n[NODE] validate_plan")

    if state["plan_status"] != "PROPOSED":
        return {}

    tasks = state["plan"]

    if not 1 <= len(tasks) <= MAX_WORKERS:
        return {
            "plan_status": "REJECTED",
            "validation_error": "Invalid worker count"
        }

    statuses = [
        task["status"]
        for task in tasks
    ]

    if len(statuses) != len(set(statuses)):
        return {
            "plan_status": "REJECTED",
            "validation_error": "Duplicate status tasks"
        }

    if not set(statuses).issubset(ALLOWED_STATUSES):
        return {
            "plan_status": "REJECTED",
            "validation_error": "Unauthorized status"
        }

    print("Plan validated.")
    print("Worker count:", len(tasks))

    return {
        "plan_status": "VALIDATED"
    }


# --------------------------------------------------
# STEP 8: DYNAMIC DISPATCHER
# --------------------------------------------------

def dispatch_workers(state: PlannerState):

    print("\n[ROUTER] dispatch_workers")

    if state["plan_status"] != "VALIDATED":
        print("Plan rejected. No workers dispatched.")
        return "combine_results"

    sends = []

    for task in state["plan"]:

        sends.append(
            Send(
                "analyse_claims",
                {
                    "status": task["status"],
                    "min_amount": MIN_AMOUNT
                }
            )
        )

    print("Dispatching:", len(sends), "workers")

    return sends


# --------------------------------------------------
# STEP 9: ONE REUSABLE WORKER NODE
# --------------------------------------------------

def analyse_claims(state: WorkerState) -> dict:

    status = state["status"]

    print(f"\n[WORKER] Analysing {status}")

    matching = [
        claim
        for claim in CLAIMS
        if claim[1] == status
        and claim[2] >= state["min_amount"]
    ]

    finding = {
        "status": status,
        "claim_ids": [
            claim[0] for claim in matching
        ],
        "claim_count": len(matching),
        "total_amount": round(
            sum(claim[2] for claim in matching),
            2
        )
    }

    return {
        "findings": [finding]
    }


# --------------------------------------------------
# STEP 10: COMBINE RESULTS
# --------------------------------------------------

def combine_results(state: PlannerState) -> dict:

    print("\n[NODE] combine_results")

    if state["plan_status"] != "VALIDATED":

        return {
            "report": {
                "status": "FAILED",
                "reason": state["validation_error"],
                "findings": [],
            }
        }

    findings = sorted(
        state["findings"],
        key=lambda x: x["status"]
    )

    expected = [
        task["status"]
        for task in state["plan"]
    ]

    actual = [
        item["status"]
        for item in findings
    ]

    # The model's plan was validated earlier.
    # Now verify that every planned task completed.
    if sorted(actual) != sorted(expected):
        return {
            "report": {
                "status": "FAILED",
                "reason": "Incomplete worker results",
                "findings": findings,
            }
        }

    return {
        "report": {
            "status": "SUCCESS",
            "worker_count": len(findings),
            "findings": findings,
            "combined_total": round(
                sum(
                    item["total_amount"]
                    for item in findings
                ),
                2
            ),
        }
    }


# --------------------------------------------------
# STEP 11: BUILD THE GRAPH
# --------------------------------------------------

builder = StateGraph(PlannerState)

builder.add_node("generate_plan", generate_plan)
builder.add_node("validate_plan", validate_plan)
builder.add_node("analyse_claims", analyse_claims)
builder.add_node("combine_results", combine_results)

builder.add_edge(START, "generate_plan")

builder.add_edge(
    "generate_plan",
    "validate_plan"
)

builder.add_conditional_edges(
    "validate_plan",
    dispatch_workers,
    ["analyse_claims", "combine_results"]
)

builder.add_edge(
    "analyse_claims",
    "combine_results"
)

builder.add_edge("combine_results", END)

graph = builder.compile()


# --------------------------------------------------
# STEP 12: RUN THE WORKFLOW
# --------------------------------------------------

if __name__ == "__main__":

    initial_state: PlannerState = {
        "goal": (
            "Compare approved and under-review insurance claims worth at least $1000."
        ),
        "plan": [],
        "plan_status": "PENDING",
        "validation_error": "",
        "findings": [],
        "report": {},
    }

    result = graph.invoke(
        initial_state,
        config={
            "max_concurrency": MAX_WORKERS,
            "recursion_limit": 15
        }
    )

    print("\n========== FINAL REPORT ==========")

    print("Plan:", result["plan"])
    print("Status:", result["report"]["status"])

    for finding in result["report"]["findings"]:
        print(
            finding["status"],
            "->",
            finding["claim_count"],
            "claims -> $",
            finding["total_amount"]
        )

    if result["report"]["status"] == "SUCCESS":
        print(
            "\nCombined total:",
            result["report"]["combined_total"]
        )
    else:
        print(
            "Reason:",
            result["report"]["reason"]
        )
