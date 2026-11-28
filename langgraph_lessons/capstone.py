
"""
LESSON 19 - PART 10: LANGGRAPH CAPSTONE

Features:
- Qwen2 structured planning
- Plan validation and budget controls
- Human approval and SQLite persistence
- Dynamic specialist dispatch with Send()
- Retry failed workers only
- Dependency-injected planner for testing

Commands:
python -m langgraph_lessons.capstone start --thread demo-1
python -m langgraph_lessons.capstone status --thread demo-1
python -m langgraph_lessons.capstone approve --thread demo-1
python -m langgraph_lessons.capstone reject --thread demo-1
"""

import argparse
import json
import operator
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from pydantic import BaseModel, Field


# ============================================
# 1. CONFIGURATION
# ============================================

DB_PATH = Path(
    "langgraph_lessons/capstone_checkpoints.sqlite"
)

MAX_PLAN_ATTEMPTS = 2
MAX_WORKER_ROUNDS = 2
MAX_CONCURRENT_WORKERS = 2

ROLE_COSTS = {
    "exposure": 2,
    "quality": 1,
    "exceptions": 1,
}

CLAIMS = [
    {"id": "CLM1001", "status": "Under Review", "amount": 4500},
    {"id": "CLM1002", "status": "Approved", "amount": 850},
    {"id": "CLM1003", "status": "Rejected", "amount": 2200},
    {"id": "CLM1004", "status": "Approved", "amount": 3200},
    {"id": "CLM1005", "status": "Approved", "amount": 1800},
]


# ============================================
# 2. STRUCTURED PLAN SCHEMA
# ============================================

class Task(BaseModel):
    role: Literal["exposure", "quality", "exceptions"]
    reason: str = Field(min_length=1)


class Plan(BaseModel):
    tasks: list[Task] = Field(min_length=1, max_length=3)


# ============================================
# 3. GRAPH STATE
# ============================================

class State(TypedDict):
    goal: str
    required_roles: list[str]
    budget_limit: int

    plan: list[dict]
    plan_attempts: int
    plan_status: str
    feedback: str

    approved: bool | None
    reserved_units: int
    worker_round: int

    fail_once: bool

    results: Annotated[list[dict], operator.add]
    report: dict


class WorkerState(TypedDict):
    role: str
    worker_round: int
    fail_once: bool


# ============================================
# 4. LLM PLANNER
# ============================================

def ollama_planner(state: State) -> list[dict]:
    """
    Called only when a real planning step executes.
    Tests can inject a fake planner instead.
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="qwen2:7b",
        temperature=0,
    ).with_structured_output(
        Plan,
        method="json_schema"
    )

    response = llm.invoke([
        (
            "system",
            "You are an insurance-review coordinator. "
            "Available roles: exposure, quality, exceptions. "
            "Create exactly one task for every required role. "
            "Do not include unrequested roles. "
            "Use the feedback to correct previous mistakes. "
            "Return the requested structured plan."
        ),
        (
            "human",
            f"Goal: {state['goal']}\n"
            f"Required roles: {state['required_roles']}\n"
            f"Feedback: {state['feedback'] or 'None'}"
        ),
    ])

    if not isinstance(response, Plan):
        raise ValueError("Unexpected planner response")

    return [
        task.model_dump()
        for task in response.tasks
    ]


# ============================================
# 5. SPECIALIST IMPLEMENTATIONS
# ============================================

def exposure_specialist():
    approved = sum(
        c["amount"] for c in CLAIMS
        if c["status"] == "Approved"
    )
    pending = sum(
        c["amount"] for c in CLAIMS
        if c["status"] == "Under Review"
    )
    return {
        "approved": approved,
        "pending": pending,
        "combined": approved + pending,
    }


def quality_specialist():
    allowed = {"Approved", "Under Review", "Rejected"}

    invalid = [
        c["id"] for c in CLAIMS
        if c["amount"] < 0
        or not c["id"]
        or c["status"] not in allowed
    ]
    return {
        "records_checked": len(CLAIMS),
        "invalid_ids": invalid,
    }


def exceptions_specialist():
    threshold = 3000
    ids = [
        c["id"] for c in CLAIMS
        if c["amount"] > threshold
    ]
    return {
        "threshold": threshold,
        "claim_ids": ids,
    }


HANDLERS = {
    "exposure": exposure_specialist,
    "quality": quality_specialist,
    "exceptions": exceptions_specialist,
}


# ============================================
# 6. BUILD A TESTABLE GRAPH
# ============================================

def build_graph(checkpointer, planner_fn=ollama_planner):
    """
    Dependency injection:
    - Real application uses Qwen2.
    - Tests pass a predictable fake planner.
    """

    def create_plan(state: State) -> dict:
        attempt = state["plan_attempts"] + 1

        print(f"[PLAN] Attempt {attempt}")

        try:
            tasks = planner_fn(state)

            return {
                "plan": tasks,
                "plan_attempts": attempt,
                "plan_status": "PROPOSED",
            }

        except Exception as exc:
            return {
                "plan": [],
                "plan_attempts": attempt,
                "plan_status": "INVALID",
                "feedback": (
                    f"Planner error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def validate(state: State) -> dict:
        if state["plan_status"] != "PROPOSED":
            return {}

        required = state["required_roles"]

        if (
            not 1 <= len(required) <= 3
            or len(required) != len(set(required))
            or not set(required).issubset(ROLE_COSTS)
            or type(state["budget_limit"]) is not int
            or state["budget_limit"] < 0
        ):
            return {
                "plan_status": "INVALID_INPUT",
                "feedback": "Invalid trusted request constraints",
            }

        tasks = state["plan"]

        if not isinstance(tasks, list) or any(
            not isinstance(task, dict)
            or task.get("role") not in ROLE_COSTS
            for task in tasks
        ):
            return {
                "plan_status": "INVALID",
                "feedback": "Invalid planned specialist roles",
            }

        planned = [task["role"] for task in tasks]

        if (
            len(planned) != len(set(planned))
            or set(planned) != set(required)
        ):
            return {
                "plan_status": "INVALID",
                "feedback": (
                    f"Required roles: {required}; "
                    f"proposed roles: {planned}. "
                    "Return exactly the required roles."
                ),
            }

        total = sum(ROLE_COSTS[r] for r in planned)

        if total > state["budget_limit"]:
            return {
                "plan_status": "BUDGET_REJECTED",
                "feedback": (
                    f"Need {total} budget units; "
                    f"available {state['budget_limit']}"
                ),
            }

        return {
            "plan_status": "VALIDATED",
            "reserved_units": total,
            "feedback": "",
        }

    def after_validation(state: State) -> str:
        if state["plan_status"] == "VALIDATED":
            return "approval"

        # Only LLM-generated mistakes can be replanned.
        if (
            state["plan_status"] == "INVALID"
            and state["plan_attempts"] < MAX_PLAN_ATTEMPTS
        ):
            return "create_plan"

        return "finish"

    def approval(state: State) -> dict:
        decision = interrupt({
            "question": "Approve execution of these specialists?",
            "tasks": state["plan"],
            "reserved_units": state["reserved_units"],
        })

        if type(decision) is not bool:
            raise ValueError(
                "Approval must be a boolean"
            )

        return {"approved": decision}

    def latest_successes(state: State) -> dict:
        """
        Keep the latest successful result per role.

        Earlier successful work can be reused when
        another specialist fails.
        """
        successes = {}

        for item in state["results"]:
            if item["success"]:
                successes[item["role"]] = item

        return successes

    def dispatch(state: State):
        if state["approved"] is not True:
            return "finish"

        successes = latest_successes(state)

        missing = [
            task["role"]
            for task in state["plan"]
            if task["role"] not in successes
        ]

        if not missing:
            return "finish"

        print(
            f"[DISPATCH] Round {state['worker_round']}: "
            f"{missing}"
        )

        return [
            Send(
                "worker",
                {
                    "role": role,
                    "worker_round": state["worker_round"],
                    "fail_once": state["fail_once"],
                },
            )
            for role in missing
        ]

    def worker(state: WorkerState) -> dict:
        role = state["role"]
        round_number = state["worker_round"]

        print(f"[WORKER] {role}, round {round_number}")

        try:
            # Demonstration of one transient failure.
            if (
                state["fail_once"]
                and role == "quality"
                and round_number == 1
            ):
                raise ConnectionError(
                    "Simulated quality source failure"
                )

            # Small delay makes concurrent work observable.
            time.sleep(0.2)

            output = HANDLERS[role]()

            item = {
                "role": role,
                "round": round_number,
                "success": True,
                "output": output,
            }

        except ConnectionError as exc:
            item = {
                "role": role,
                "round": round_number,
                "success": False,
                "error": str(exc),
            }

        return {"results": [item]}

    def assess(state: State) -> dict:
        successful = set(latest_successes(state))
        expected = {t["role"] for t in state["plan"]}

        if successful == expected:
            return {"plan_status": "COMPLETED"}

        return {
            "plan_status": "WORKER_FAILED",
            "feedback": (
                f"Missing successful roles: "
                f"{sorted(expected - successful)}"
            ),
        }

    def after_assessment(state: State) -> str:
        if state["plan_status"] == "COMPLETED":
            return "finish"

        if state["worker_round"] < MAX_WORKER_ROUNDS:
            return "prepare_retry"

        return "finish"

    def prepare_retry(state: State) -> dict:
        return {
            "worker_round": state["worker_round"] + 1
        }

    def finish(state: State) -> dict:
        if state["approved"] is False:
            return {
                "report": {
                    "status": "REJECTED_BY_HUMAN",
                }
            }

        if state["plan_status"] != "COMPLETED":
            return {
                "report": {
                    "status": "FAILED",
                    "reason": state["feedback"],
                }
            }

        successes = latest_successes(state)

        return {
            "report": {
                "status": "SUCCESS",
                "plan_attempts": state["plan_attempts"],
                "worker_rounds": state["worker_round"],
                "reserved_units": state["reserved_units"],
                "specialists": {
                    role: item["output"]
                    for role, item in sorted(successes.items())
                },
            }
        }

    # ========================================
    # 7. GRAPH CONSTRUCTION
    # ========================================

    builder = StateGraph(State)

    builder.add_node("create_plan", create_plan)
    builder.add_node("validate", validate)
    builder.add_node("approval", approval)
    builder.add_node("worker", worker)
    builder.add_node("assess", assess)
    builder.add_node("prepare_retry", prepare_retry)
    builder.add_node("finish", finish)

    builder.add_edge(START, "create_plan")
    builder.add_edge("create_plan", "validate")

    builder.add_conditional_edges(
        "validate",
        after_validation,
        ["create_plan", "approval", "finish"],
    )

    builder.add_conditional_edges(
        "approval",
        dispatch,
        ["worker", "finish"],
    )

    builder.add_edge("worker", "assess")

    builder.add_conditional_edges(
        "assess",
        after_assessment,
        ["finish", "prepare_retry"],
    )

    builder.add_conditional_edges(
        "prepare_retry",
        dispatch,
        ["worker", "finish"],
    )

    builder.add_edge("finish", END)

    return builder.compile(
        checkpointer=checkpointer
    )


# ============================================
# 8. INITIAL WORKFLOW STATE
# ============================================

def initial_state(
    roles=None,
    budget=4,
    fail_once=False,
):
    if roles is None:
        roles = ["exposure", "quality", "exceptions"]

    return {
        "goal": (
            "Review claim exposure, data quality "
            "and unusually large claims."
        ),
        "required_roles": roles,
        "budget_limit": budget,

        "plan": [],
        "plan_attempts": 0,
        "plan_status": "PENDING",
        "feedback": "",

        "approved": None,
        "reserved_units": 0,
        "worker_round": 1,

        "fail_once": fail_once,

        "results": [],
        "report": {},
    }


# ============================================
# 9. COMMAND-LINE INTERFACE
# ============================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["start", "status", "approve", "reject"],
    )
    parser.add_argument(
        "--thread",
        required=True,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--fail-once",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.thread.strip():
        parser.error("--thread cannot be empty")

    config = {
        "configurable": {"thread_id": args.thread},
        "max_concurrency": MAX_CONCURRENT_WORKERS,
        "recursion_limit": 40,
    }

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SqliteSaver.from_conn_string(
        str(DB_PATH)
    ) as saver:

        graph = build_graph(saver)
        snapshot = graph.get_state(config)

        if args.command == "start":
            if snapshot.values:
                raise RuntimeError(
                    "Thread already exists. "
                    "Choose a new --thread value."
                )

            result = graph.invoke(
                initial_state(
                    budget=args.budget,
                    fail_once=args.fail_once,
                ),
                config=config,
                durability="sync",
            )

        elif args.command == "status":
            print("State:", snapshot.values)
            print("Next:", snapshot.next)
            print(
                "Pending approvals:",
                [
                    i.value
                    for task in snapshot.tasks
                    for i in task.interrupts
                ],
            )
            return

        else:
            pending = [
                i
                for task in snapshot.tasks
                for i in task.interrupts
            ]

            if not pending:
                raise RuntimeError(
                    "No pending approval to resume"
                )

            result = graph.invoke(
                Command(
                    resume=(args.command == "approve")
                ),
                config=config,
                durability="sync",
            )

        print(json.dumps(result, indent=2, default=str))
        print("Next:", graph.get_state(config).next)


if __name__ == "__main__":
    main()
