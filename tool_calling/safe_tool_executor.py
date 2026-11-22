
"""
LESSON 16 - PART 7: SAFE TOOL EXECUTION
=======================================

OBJECTIVE:
Build a reusable executor that validates tool
requests before calling Python functions.

CONCEPTS:
1. Tool allowlists
2. Pydantic argument validation
3. Field constraints and strict types
4. Exception handling
5. Tool execution limits
6. Structured error responses

RUN:
python -m tool_calling.safe_tool_executor
"""

import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)


# --------------------------------------------------
# STEP 1: SAMPLE DATABASE
# --------------------------------------------------

CLAIMS_DATABASE = {
    "CLM1001": {"status": "Under Review", "amount": 4500},
    "CLM1002": {"status": "Approved", "amount": 850},
    "CLM1003": {"status": "Rejected", "amount": 2200},
    "CLM1004": {"status": "Approved", "amount": 3200},
    "CLM1005": {"status": "Approved", "amount": 1800},
}


# --------------------------------------------------
# STEP 2: PYDANTIC ARGUMENT MODELS
# --------------------------------------------------

class ClaimIdArgs(BaseModel):
    """Validate arguments for individual claim lookup."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    claim_id: str = Field(
        pattern=r"^CLM[0-9]{4}$"
    )


class FindClaimsArgs(BaseModel):
    """Validate claim search filters."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    status: Literal[
        "Approved",
        "Rejected",
        "Under Review"
    ]

    min_amount: float = Field(
        ge=0,
        allow_inf_nan=False
    )


# --------------------------------------------------
# STEP 3: APPROVED PYTHON TOOLS
# --------------------------------------------------

def get_claim_details(claim_id: str) -> dict:
    """Return details for an individual claim."""

    claim = CLAIMS_DATABASE.get(claim_id)

    if claim is None:
        return {"error": "Claim not found"}

    return {"claim_id": claim_id, **claim}


def find_claims(
    status: str,
    min_amount: float
) -> list[dict]:
    """Find claims matching status and minimum amount."""

    return [
        {"claim_id": claim_id, **claim}
        for claim_id, claim in CLAIMS_DATABASE.items()
        if (
            claim["status"] == status
            and claim["amount"] >= min_amount
        )
    ]


# --------------------------------------------------
# STEP 4: ALLOWLIST AND ARGUMENT SCHEMAS
# --------------------------------------------------

# Only registered functions may execute.

AVAILABLE_TOOLS = {
    "get_claim_details": get_claim_details,
    "find_claims": find_claims,
}

ARGUMENT_MODELS = {
    "get_claim_details": ClaimIdArgs,
    "find_claims": FindClaimsArgs,
}


# --------------------------------------------------
# STEP 5: SAFE TOOL EXECUTOR
# --------------------------------------------------

class SafeToolExecutor:
    """
    Execute approved tools with validated arguments.

    max_calls limits the total number of tool
    executions attempted through this instance.
    """

    def __init__(self, max_calls: int = 5):

        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be positive")

        self.max_calls = max_calls
        self.call_count = 0

    def execute(self, name: str, arguments: dict) -> dict:
        """
        Validate and execute one tool request.

        Returns:
            A structured success or error result.
        """

        # ------------------------------------------
        # CHECK 1: APPROVED TOOL
        # ------------------------------------------

        if name not in AVAILABLE_TOOLS:

            return {
                "success": False,
                "error": "Unauthorized or unknown tool"
            }

        # ------------------------------------------
        # CHECK 2: EXECUTION LIMIT
        # ------------------------------------------

        if self.call_count >= self.max_calls:

            return {
                "success": False,
                "error": "Tool execution limit reached"
            }

        # Count an approved tool attempt, even if
        # its arguments are subsequently invalid.
        self.call_count += 1

        # ------------------------------------------
        # CHECK 3: PYDANTIC VALIDATION
        # ------------------------------------------

        try:

            schema = ARGUMENT_MODELS[name]

            validated = schema.model_validate(arguments)

        except ValidationError as exc:

            return {
                "success": False,
                "error": "Invalid tool arguments",
                "details": [
                    {
                        "field": ".".join(
                            str(item) for item in error["loc"]
                        ),
                        "message": error["msg"]
                    }
                    for error in exc.errors()
                ]
            }

        # ------------------------------------------
        # CHECK 4: EXECUTE APPROVED FUNCTION
        # ------------------------------------------

        try:

            function = AVAILABLE_TOOLS[name]

            result = function(
                **validated.model_dump()
            )

            return {
                "success": True,
                "data": result
            }

        except Exception:

            # Avoid exposing internal error details.
            return {
                "success": False,
                "error": "Tool execution failed"
            }


# --------------------------------------------------
# STEP 6: TEST TOOL EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    executor = SafeToolExecutor(max_calls=5)

    test_cases = [
        (
            "Valid claim lookup",
            "get_claim_details",
            {"claim_id": "CLM1004"}
        ),
        (
            "Invalid claim ID",
            "get_claim_details",
            {"claim_id": "INVALID"}
        ),
        (
            "Negative amount",
            "find_claims",
            {
                "status": "Approved",
                "min_amount": -500
            }
        ),
        (
            "Unauthorized tool",
            "delete_all_claims",
            {}
        ),
        (
            "Valid claim search",
            "find_claims",
            {
                "status": "Approved",
                "min_amount": 1000
            }
        )
    ]

    for label, name, arguments in test_cases:

        print(f"\n--- {label} ---")

        result = executor.execute(
            name=name,
            arguments=arguments
        )

        print(json.dumps(result, indent=2))

    print("\nApproved tool attempts:", executor.call_count)


    print("\n--- EXECUTION LIMIT TEST ---")

    limited_executor = SafeToolExecutor(max_calls=2)

    for attempt in range(1, 5):

        result = limited_executor.execute(
            name="get_claim_details",
            arguments={"claim_id": "CLM1004"}
        )

        print(
            f"Attempt {attempt}: "
            f"{result}"
        )