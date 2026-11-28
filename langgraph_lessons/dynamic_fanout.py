
"""
LESSON 19 - PART 4: DYNAMIC FAN-OUT
==================================

Learn:
1. LangGraph Send()
2. Runtime-generated worker tasks
3. Worker-specific state
4. Parallel results with reducers
5. Safe worker-count limits
6. Handling zero tasks

RUN:
python -m langgraph_lessons.dynamic_fanout
"""

import asyncio
import operator
import time

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# --------------------------------------------------
# STEP 1: SAMPLE DATA
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
    "Rejected"
}

MAX_WORKERS = 3


# --------------------------------------------------
# STEP 2: DEFINE THE MAIN GRAPH STATE
# --------------------------------------------------

class OverallState(TypedDict):

    # Requested analysis categories
    statuses: list[str]

    min_amount: float

    # Append contributions from parallel workers.
    findings: Annotated[list[dict], operator.add]

    report: dict


# --------------------------------------------------
# STEP 3: DEFINE WORKER-SPECIFIC STATE
# --------------------------------------------------

class WorkerState(TypedDict):

    # Each worker receives ONE status.
    status: str

    min_amount: float


# --------------------------------------------------
# STEP 4: PREPARE AND VALIDATE TASKS
# --------------------------------------------------

def prepare_analysis(state: OverallState) -> dict:

    print("\n[NODE] prepare_analysis")

    statuses = state["statuses"]

    # Avoid launching duplicate work.
    unique_statuses = list(dict.fromkeys(statuses))

    if len(unique_statuses) > MAX_WORKERS:
        raise ValueError(
            f"Maximum {MAX_WORKERS} workers allowed"
        )

    unknown = [
        status
        for status in unique_statuses
        if status not in ALLOWED_STATUSES
    ]

    if unknown:
        raise ValueError(
            f"Unsupported statuses: {unknown}"
        )

    if state["min_amount"] < 0:
        raise ValueError(
            "Minimum amount cannot be negative"
        )

    print("Requested workers:", len(unique_statuses))

    return {
        "statuses": unique_statuses
    }


# --------------------------------------------------
# STEP 5: DYNAMIC ROUTER
# --------------------------------------------------

def dispatch_workers(state: OverallState):

    print("\n[ROUTER] dispatch_workers")

    # No work requested: proceed directly to report.
    if not state["statuses"]:
        print("No workers required")
        return "combine_results"

    tasks = []

    for status in state["statuses"]:

        tasks.append(
            Send(
                "analyse_claims",
                {
                    "status": status,
                    "min_amount": state["min_amount"]
                }
            )
        )

    print("Workers dispatched:", len(tasks))

    return tasks


# --------------------------------------------------
# STEP 6: ONE REUSABLE WORKER NODE
# --------------------------------------------------

async def analyse_claims(
    state: WorkerState
) -> dict:

    status = state["status"]
    min_amount = state["min_amount"]

    print(f"\n[WORKER START] {status}")

    # Simulates I/O latency.
    await asyncio.sleep(1)

    matching = [
        claim
        for claim in CLAIMS
        if claim[1] == status
        and claim[2] >= min_amount
    ]

    finding = {
        "status": status,
        "claim_ids": [
            claim[0]
            for claim in matching
        ],
        "claim_count": len(matching),
        "total_amount": round(
            sum(
                claim[2]
                for claim in matching
            ),
            2
        )
    }

    print(f"[WORKER DONE] {status}")

    # Return an update to OverallState.
    # The reducer combines all worker outputs.
    return {
        "findings": [finding]
    }


# --------------------------------------------------
# STEP 7: COMBINE ALL WORKER RESULTS
# --------------------------------------------------

def combine_results(state: OverallState) -> dict:

    print("\n[NODE] combine_results")

    # Don't rely on parallel completion order.
    findings = sorted(
        state["findings"],
        key=lambda result: result["status"]
    )

    # Defensive check: every requested status
    # must have exactly one result.
    expected = set(state["statuses"])

    actual = [
        result["status"]
        for result in findings
    ]

    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "Missing or duplicate worker results"
        )

    combined_total = round(
        sum(
            result["total_amount"]
            for result in findings
        ),
        2
    )

    return {
        "report": {
            "worker_count": len(findings),
            "findings": findings,
            "combined_total": combined_total
        }
    }


# --------------------------------------------------
# STEP 8: CREATE GRAPH
# --------------------------------------------------

builder = StateGraph(OverallState)

builder.add_node(
    "prepare_analysis",
    prepare_analysis
)

builder.add_node(
    "analyse_claims",
    analyse_claims
)

builder.add_node(
    "combine_results",
    combine_results
)


# --------------------------------------------------
# STEP 9: DEFINE GRAPH EDGES
# --------------------------------------------------

builder.add_edge(
    START,
    "prepare_analysis"
)

# Dynamic routing replaces hardcoded parallel edges.
builder.add_conditional_edges(
    "prepare_analysis",
    dispatch_workers,
    ["analyse_claims", "combine_results"]
)

# Once dispatched workers complete, combine results.
builder.add_edge(
    "analyse_claims",
    "combine_results"
)

builder.add_edge(
    "combine_results",
    END
)


# --------------------------------------------------
# STEP 10: COMPILE GRAPH
# --------------------------------------------------

graph = builder.compile()


# --------------------------------------------------
# STEP 11: RUN MULTIPLE SCENARIOS
# --------------------------------------------------

async def run_scenario(statuses: list[str]):

    print("\n" + "=" * 50)
    print("REQUESTED STATUSES:", statuses)
    print("=" * 50)

    initial_state: OverallState = {
        "statuses": statuses,
        "min_amount": 1000.0,
        "findings": [],
        "report": {}
    }

    start = time.perf_counter()

    result = await graph.ainvoke(
        initial_state,
        config={
            "max_concurrency": MAX_WORKERS,
            "recursion_limit": 15
        }
    )

    elapsed = time.perf_counter() - start

    print("\nFINAL REPORT:")

    print(
        "Workers:",
        result["report"]["worker_count"]
    )

    for finding in result["report"]["findings"]:
        print(
            finding["status"],
            "->",
            finding["total_amount"]
        )

    print(
        "Combined:",
        result["report"]["combined_total"]
    )

    print(f"Elapsed: {elapsed:.2f}s")


async def main():

    # One worker
    await run_scenario(["Approved"])

    # Two workers
    await run_scenario([
        "Approved",
        "Under Review"
    ])

    # Three workers
    await run_scenario([
        "Approved",
        "Under Review",
        "Rejected"
    ])

    # Zero workers
    await run_scenario([])


if __name__ == "__main__":
    asyncio.run(main())
