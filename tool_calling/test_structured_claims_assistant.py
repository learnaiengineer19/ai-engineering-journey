
"""
LESSON 16 - PART 9.5: INTEGRATION TESTING
========================================

OBJECTIVE:
Test our structured insurance assistant without
requiring Ollama or a running REST API server.

CONCEPTS:
1. pytest fixtures
2. monkeypatch
3. Fake LLM tool-call responses
4. Temporary SQLite databases
5. Error-path testing
6. Structured response assertions

RUN FROM PROJECT ROOT:
python -m pytest tool_calling/test_structured_claims_assistant.py -v
"""

from types import SimpleNamespace

import pytest
import requests

from tool_calling import structured_claims_assistant as assistant
from tool_calling import setup_claims_db


# --------------------------------------------------
# STEP 1: CREATE FAKE LLM RESPONSES
# --------------------------------------------------

def fake_tool_response(name, arguments):
    """
    Simulate the structure returned by Ollama
    when an LLM requests a Python tool.
    """

    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments
        )
    )

    return SimpleNamespace(
        message=SimpleNamespace(
            tool_calls=[tool_call]
        )
    )


def mock_llm(monkeypatch, name, arguments):
    """
    Replace Ollama's chat() with a deterministic
    response requesting the specified tool.
    """

    def fake_chat(**kwargs):
        return fake_tool_response(name, arguments)

    monkeypatch.setattr(
        assistant,
        "chat",
        fake_chat
    )


# --------------------------------------------------
# STEP 2: CREATE TEMPORARY SQLITE DATABASE
# --------------------------------------------------

@pytest.fixture
def test_database(tmp_path, monkeypatch):
    """
    Create a separate SQLite database for testing.

    This avoids modifying the developer's
    existing claims.db file.
    """

    database_path = tmp_path / "test_claims.db"

    monkeypatch.setattr(
        setup_claims_db,
        "DB_PATH",
        database_path
    )

    monkeypatch.setattr(
        assistant,
        "DB_PATH",
        database_path
    )

    setup_claims_db.setup_database()

    return database_path


# --------------------------------------------------
# TEST 1: SUCCESSFUL DATABASE SEARCH
# --------------------------------------------------

def test_database_search(monkeypatch, test_database):

    mock_llm(
        monkeypatch,
        "find_claims_database",
        {
            "status": "Approved",
            "min_amount": 1000
        }
    )

    result = assistant.run_assistant(
        "Find approved claims above $1000 using SQLite"
    )

    data = result["response"]

    assert data["success"] is True
    assert data["source"] == "database"
    assert data["count"] == 2

    claim_ids = {
        claim["claim_id"]
        for claim in data["claims"]
    }

    assert claim_ids == {"CLM1004", "CLM1005"}

    assert result["metrics"]["llm_calls"] == 1
    assert result["metrics"]["tool_executions"] == 1


# --------------------------------------------------
# TEST 2: SUCCESSFUL REST API SEARCH
# --------------------------------------------------

def test_api_search(monkeypatch):

    mock_llm(
        monkeypatch,
        "find_claims_api",
        {
            "status": "Rejected",
            "min_amount": 2000
        }
    )

    # Mock the API tool, avoiding any HTTP request.
    def fake_api(status, min_amount):
        assert status == "Rejected"
        assert min_amount == 2000

        return {
            "count": 1,
            "claims": [
                {
                    "claim_id": "CLM1003",
                    "status": "Rejected",
                    "amount": 2200.0,
                    "claim_type": "Mechanical Breakdown"
                }
            ]
        }

    monkeypatch.setattr(
        assistant,
        "find_claims_api",
        fake_api
    )

    result = assistant.run_assistant(
        "Find rejected claims above $2000 using REST API"
    )

    assert result["response"]["source"] == "api"
    assert result["response"]["count"] == 1
    assert (
        result["response"]["claims"][0]["claim_id"]
        == "CLM1003"
    )


# --------------------------------------------------
# TEST 3: SUCCESSFUL SEARCH WITH ZERO RESULTS
# --------------------------------------------------

def test_no_matching_claims(monkeypatch, test_database):

    mock_llm(
        monkeypatch,
        "find_claims_database",
        {
            "status": "Approved",
            "min_amount": 10000
        }
    )

    result = assistant.run_assistant(
        "Find approved claims above $10000 using SQLite"
    )

    assert result["response"]["success"] is True
    assert result["response"]["count"] == 0
    assert result["response"]["claims"] == []


# --------------------------------------------------
# TEST 4: INVALID LLM ARGUMENTS
# --------------------------------------------------

def test_invalid_arguments(monkeypatch):

    mock_llm(
        monkeypatch,
        "find_claims_database",
        {
            "status": "InvalidStatus",
            "min_amount": -100
        }
    )

    result = assistant.run_assistant(
        "Find claims using SQLite"
    )

    assert result["success"] is False
    assert result["error"] == (
        "Input or output validation failed"
    )
    assert result["tool_executions"] == 0


# --------------------------------------------------
# TEST 5: UNAUTHORIZED TOOL
# --------------------------------------------------

def test_unauthorized_tool(monkeypatch):

    mock_llm(
        monkeypatch,
        "delete_all_claims",
        {}
    )

    result = assistant.run_assistant(
        "Find approved claims using SQLite"
    )

    assert result["success"] is False
    assert result["error"] == "Tool is not permitted"


# --------------------------------------------------
# TEST 6: REST API TIMEOUT
# --------------------------------------------------

def test_api_timeout(monkeypatch):

    mock_llm(
        monkeypatch,
        "find_claims_api",
        {
            "status": "Approved",
            "min_amount": 1000
        }
    )

    def failing_api(status, min_amount):
        raise requests.Timeout("Simulated timeout")

    monkeypatch.setattr(
        assistant,
        "find_claims_api",
        failing_api
    )

    result = assistant.run_assistant(
        "Find approved claims above $1000 using REST API"
    )

    assert result["success"] is False
    assert result["error"] == "REST API request timed out"


# --------------------------------------------------
# TEST 7: NO TOOL SELECTED
# --------------------------------------------------

def test_no_tool_selected(monkeypatch):

    def fake_chat(**kwargs):

        return SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=[]
            )
        )

    monkeypatch.setattr(
        assistant,
        "chat",
        fake_chat
    )

    result = assistant.run_assistant(
        "Find approved claims using SQLite"
    )

    assert result["success"] is False
    assert result["error"] == (
        "Expected exactly one tool request."
    )
