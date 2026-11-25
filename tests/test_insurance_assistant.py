
"""
LESSON 18 - PART 10: AUTOMATED TESTING

Run:
python -m pytest tests/test_insurance_assistant.py -v

No Ollama inference or paid API calls are needed
for these tests.
"""

import json
import sqlite3

import pytest
from pydantic import ValidationError

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

import langchain_lessons.insurance_assistant as app


# --------------------------------------------------
# TEST DATABASE FIXTURE
# --------------------------------------------------

@pytest.fixture
def test_database(tmp_path, monkeypatch):
    """
    Create an isolated SQLite database for each test.

    tmp_path is a built-in pytest fixture.
    monkeypatch temporarily replaces app.DB_FILE.
    """

    db_path = tmp_path / "test_claims.db"

    with sqlite3.connect(db_path) as connection:

        connection.execute(
            """
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                amount REAL NOT NULL,
                claim_type TEXT NOT NULL
            )
            """
        )

        records = [
            ("CLM1001", "Under Review", 4500, "Motor Accident"),
            ("CLM1002", "Approved", 850, "Windshield Damage"),
            ("CLM1003", "Rejected", 2200, "Mechanical Breakdown"),
            ("CLM1004", "Approved", 3200, "Motor Accident"),
            ("CLM1005", "Approved", 1800, "Vehicle Damage"),
        ]

        connection.executemany(
            """
            INSERT INTO claims
                (claim_id, status, amount, claim_type)
            VALUES (?, ?, ?, ?)
            """,
            records
        )

        connection.commit()

    monkeypatch.setattr(
        app,
        "DB_FILE",
        db_path
    )

    return db_path


# --------------------------------------------------
# TEST 1: FIND APPROVED CLAIMS
# --------------------------------------------------

def test_find_approved_claims(test_database):

    result = app.read_claims(
        status="Approved",
        min_amount=1000
    )

    assert len(result) == 2

    claim_ids = {
        claim["claim_id"]
        for claim in result
    }

    assert claim_ids == {
        "CLM1004",
        "CLM1005"
    }


# --------------------------------------------------
# TEST 2: TOOL RETURNS CORRECT TOTAL
# --------------------------------------------------

def test_calculate_total(test_database):

    # LangChain tools are invoked through .invoke()
    result = app.calculate_total.invoke({
        "claim_ids": [
            "CLM1004",
            "CLM1005"
        ]
    })

    # Part 9 explicitly returned JSON strings.
    data = result

    assert data["success"] is True
    assert data["claim_count"] == 2
    assert data["total_amount"] == 5000.0


# --------------------------------------------------
# TEST 3: DUPLICATE IDS DON'T DOUBLE COUNT
# --------------------------------------------------

def test_duplicate_claim_ids(test_database):

    result = app.calculate_total.invoke({
        "claim_ids": [
            "CLM1004",
            "CLM1004",
            "CLM1005"
        ]
    })

    data = result

    assert data["claim_count"] == 2
    assert data["total_amount"] == 5000.0


# --------------------------------------------------
# TEST 4: UNKNOWN CLAIM IDS FAIL
# --------------------------------------------------

def test_unknown_claim_id(test_database):

    result = app.calculate_total.invoke({
        "claim_ids": ["CLM9999"]
    })

    data = result

    assert data["success"] is False
    assert "CLM9999" in data["missing_ids"]


# --------------------------------------------------
# TEST 5: INVALID PARAMETERS ARE REJECTED
# --------------------------------------------------

def test_negative_minimum_amount(test_database):

    with pytest.raises(ValidationError):

        app.find_claims.invoke({
            "status": "Approved",
            "min_amount": -100
        })


# --------------------------------------------------
# HELPER: SIMULATE A COMPLETE AGENT HISTORY
# --------------------------------------------------

def build_fake_agent_result(
    search_claims,
    reported_total
):
    """
    Construct realistic LangChain messages.

    This is a test double. It simulates an agent
    history without asking Qwen2 to generate it.
    """

    search_call_id = "call_search_1"
    total_call_id = "call_total_1"

    return {
        "messages": [
            HumanMessage(
                content=app.QUESTION
            ),

            AIMessage(
                content="",
                tool_calls=[{
                    "name": "find_claims",
                    "args": {
                        "status": "Approved",
                        "min_amount": 1000
                    },
                    "id": search_call_id,
                    "type": "tool_call"
                }]
            ),

            ToolMessage(
                content=json.dumps(search_claims),
                name="find_claims",
                tool_call_id=search_call_id
            ),

            AIMessage(
                content="",
                tool_calls=[{
                    "name": "calculate_total",
                    "args": {
                        "claim_ids": [
                            "CLM1004",
                            "CLM1005"
                        ]
                    },
                    "id": total_call_id,
                    "type": "tool_call"
                }]
            ),

            ToolMessage(
                content=json.dumps({
                    "success": True,
                    "claim_ids": [
                        "CLM1004",
                        "CLM1005"
                    ],
                    "claim_count": 2,
                    "total_amount": reported_total
                }),
                name="calculate_total",
                tool_call_id=total_call_id
            ),

            AIMessage(
                content="Two approved claims were found."
            )
        ]
    }


# --------------------------------------------------
# TEST 6: CORRECT AGENT OUTPUT IS VERIFIED
# --------------------------------------------------

def test_verified_report(test_database):

    claims = app.read_claims(
        "Approved",
        1000
    )

    fake_result = build_fake_agent_result(
        search_claims=claims,
        reported_total=5000
    )

    report = app.verify_and_build_report(
        fake_result
    )

    assert report.verification_status == "verified"
    assert report.claim_count == 2
    assert report.total_amount == 5000
    assert report.tool_requests == 2
    assert report.llm_responses == 3


# --------------------------------------------------
# TEST 7: INCORRECT TOTAL MUST BE REJECTED
# --------------------------------------------------

def test_incorrect_total_fails(test_database):

    claims = app.read_claims(
        "Approved",
        1000
    )

    fake_result = build_fake_agent_result(
        search_claims=claims,
        reported_total=5500
    )

    with pytest.raises(
        ValueError,
        match="Verification failed"
    ):

        app.verify_and_build_report(
            fake_result
        )


# --------------------------------------------------
# TEST 8: MISSING SEARCH RESULT MUST FAIL
# --------------------------------------------------

def test_missing_claim_fails(test_database):

    claims = app.read_claims(
        "Approved",
        1000
    )

    # Deliberately omit CLM1005.
    incomplete_claims = claims[:1]

    fake_result = build_fake_agent_result(
        search_claims=incomplete_claims,
        reported_total=5000
    )

    with pytest.raises(
        ValueError,
        match="Verification failed"
    ):

        app.verify_and_build_report(
            fake_result
        )
