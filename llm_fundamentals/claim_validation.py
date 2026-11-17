from pydantic import BaseModel,ValidationError


class InsuranceClaim(BaseModel):
    claim_type: str
    incident_date: str
    estimated_cost: int


claim_data = {
    "claim_type": "car",
    "incident_date": "2026-10-15",
    "estimated_cost": 3500
}

invalid_claim = {
    "claim_type": "car",
    "incident_date": "2026-10-15",
    "estimated_cost": "three thousand dollars"
}


try:
    claim = InsuranceClaim(**invalid_claim)
    print(claim.claim_type)
    print(claim.incident_date)
    print(claim.estimated_cost)
except ValidationError as error:
    print("error:", error)

