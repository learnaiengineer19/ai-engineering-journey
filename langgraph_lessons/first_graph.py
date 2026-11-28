from typing import TypedDict

class ClaimState(TypedDict):
    status: str
    min_amount: float
    claims: list[dict]
    total_amount: float
    verified: bool

SAMPLE_CLAIMS = [
    {
        "claim_id": "CLM1002",
        "status": "Approved",
        "amount": 850.0
    },
    {
        "claim_id": "CLM1004",
        "status": "Approved",
        "amount": 3200.0
    },
    {
        "claim_id": "CLM1005",
        "status": "Approved",
        "amount": 1800.0
    }
]


def find_claims(state: ClaimState) -> dict:
    print("[NODE] find_claims")

    matching = [
        claim
        for claim in SAMPLE_CLAIMS
        if claim["status"] == state["status"]
        and claim["amount"] >= state["min_amount"]
    ]

    print("matching claims ", matching)

    return {"claims": matching}


def calculate_total(state: ClaimState) -> dict:
    print("[NODE] calculate_total")

    total = sum(
        claim["amount"]
        for claim in state["claims"]
    )

    return {"total_amount": round(total, 2)}


def verify_total(state: ClaimState) -> dict:
    print("[NODE] verify_total")

    expected = sum(
        claim["amount"]
        for claim in SAMPLE_CLAIMS
        if claim["status"] == state["status"]
        and claim["amount"] >= state["min_amount"]
    )

    return {
        "verified": abs(
            state["total_amount"] - expected
        ) < 0.01
    }

from langgraph.graph import StateGraph, START, END


# Create a graph whose shared state follows ClaimState.
builder = StateGraph(ClaimState)

# Register our three functions as graph nodes.
builder.add_node("find_claims", find_claims)
builder.add_node("calculate_total", calculate_total)
builder.add_node("verify_total", verify_total)

# Define how execution moves between nodes.
builder.add_edge(START, "find_claims")
builder.add_edge("find_claims", "calculate_total")
builder.add_edge("calculate_total", "verify_total")
builder.add_edge("verify_total", END)

# Compile the graph into an executable object.
graph = builder.compile()


if __name__ == "__main__":

    initial_state = {
        "status": "Approved",
        "min_amount": 1000,
        "claims": [],
        "total_amount": 0.0,
        "verified": False
    }

    print("\n========== INVOKE ==========")

    result = graph.invoke(initial_state)

    print("\nFinal state:")
    print(result)

    print("\n========== STREAM UPDATES ==========")

    for update in graph.stream(
        initial_state,
        stream_mode="updates"
    ):
        print(update)