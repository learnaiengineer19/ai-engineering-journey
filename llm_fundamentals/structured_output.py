import json

llm_response = '''
{
    "claim_type": "car",
    "incident_date": "2026-10-15",
    "estimated_cost": 3500
}
'''

llm_response = json.loads(llm_response)
print(llm_response["claim_type"])
print(llm_response["incident_date"])
print(type(llm_response["estimated_cost"]))


from ollama import chat, ResponseError

claim_description = (
    "My car was damaged in an accident on 15 October 2026. "
    "The estimated repair cost is $3,500."
)

messages = [
    {
        "role": "system",
        "content": (
            "You extract structured information from insurance claims. "
            "Return a JSON object with exactly these fields: "
            "claim_type, incident_date, estimated_cost. "
            "Use YYYY-MM-DD for dates and a number for costs."
        )
    },
    {
        "role": "user",
        "content": claim_description
    }
]


try:
    response = chat(
        model="qwen2:7b",
        messages=messages,
        format="json"
    )

except ResponseError as error:
    print("Ollama error:", error)

except ConnectionError as error:
    print("Connection error:", error)

else:
    # This block runs only if the API call succeeded.

    try:
        result = json.loads(response.message.content)

        print("Extracted claim:", result)

        print(
            "Estimated cost:",
            result["estimated_cost"]
        )

        print(
            "Cost type:",
            type(result["estimated_cost"])
        )

    except json.JSONDecodeError as error:
        print("Invalid JSON:", error)

    except KeyError as error:
        print("Missing JSON field:", error)
