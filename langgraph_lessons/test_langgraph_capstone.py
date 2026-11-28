
import pytest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_lessons.capstone import (
    build_graph,
    initial_state,
)


def good_planner(state):
    return [
        {
            "role": role,
            "reason": "Required specialist",
        }
        for role in state["required_roles"]
    ]


def run_graph(
    *,
    budget=4,
    fail_once=False,
    planner=good_planner,
    approval=True,
):
    graph = build_graph(
        InMemorySaver(),
        planner_fn=planner,
    )

    config = {
        "configurable": {
            "thread_id": "test-workflow"
        },
        "max_concurrency": 2,
        "recursion_limit": 40,
    }

    first = graph.invoke(
        initial_state(
            budget=budget,
            fail_once=fail_once,
        ),
        config,
    )

    pending = first.get("__interrupt__", ())

    if not pending:
        return first, graph, config

    final = graph.invoke(
        Command(resume=approval),
        config,
    )

    return final, graph, config


def test_approval_and_success():
    result, graph, config = run_graph()

    assert result["report"]["status"] == "SUCCESS"
    assert result["report"]["reserved_units"] == 4
    assert result["report"]["worker_rounds"] == 1

    assert (
        result["report"]["specialists"]
        ["exposure"]["combined"]
        == 10350
    )

    assert graph.get_state(config).next == ()


def test_rejected_by_human():
    result, _, _ = run_graph(approval=False)

    assert (
        result["report"]["status"]
        == "REJECTED_BY_HUMAN"
    )

    assert result["results"] == []


def test_budget_blocks_execution():
    result, _, _ = run_graph(budget=3)

    assert result["report"]["status"] == "FAILED"
    assert "Need 4" in result["report"]["reason"]

    assert result["results"] == []


def test_retry_only_failed_worker():
    result, _, _ = run_graph(fail_once=True)

    assert result["report"]["status"] == "SUCCESS"
    assert result["report"]["worker_rounds"] == 2

    quality_results = [
        r for r in result["results"]
        if r["role"] == "quality"
    ]

    exposure_results = [
        r for r in result["results"]
        if r["role"] == "exposure"
    ]

    exception_results = [
        r for r in result["results"]
        if r["role"] == "exceptions"
    ]

    assert len(quality_results) == 2
    assert len(exposure_results) == 1
    assert len(exception_results) == 1

    assert quality_results[0]["success"] is False
    assert quality_results[1]["success"] is True


def test_invalid_plan_gets_replanned():
    attempts = 0

    def planner_with_one_bad_attempt(state):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return [
                {
                    "role": "exposure",
                    "reason": "Incomplete plan",
                }
            ]

        return good_planner(state)

    result, _, _ = run_graph(
        planner=planner_with_one_bad_attempt
    )

    assert result["report"]["status"] == "SUCCESS"
    assert result["report"]["plan_attempts"] == 2
    assert attempts == 2


def test_persistent_interrupt_state():
    saver = InMemorySaver()

    graph = build_graph(
        saver,
        planner_fn=good_planner,
    )

    config = {
        "configurable": {
            "thread_id": "persistent-test"
        }
    }

    first = graph.invoke(
        initial_state(),
        config,
    )

    assert first.get("__interrupt__")

    snapshot = graph.get_state(config)

    assert snapshot.next == ("approval",)
    assert snapshot.values["reserved_units"] == 4

    final = graph.invoke(
        Command(resume=True),
        config,
    )

    assert final["report"]["status"] == "SUCCESS"
