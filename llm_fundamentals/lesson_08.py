
"""
LESSON 08: LLM FUNDAMENTALS
===========================

OBJECTIVE:
Understand tokens, next-token prediction,
probabilities and basic text generation.

CONCEPTS COVERED:

1. LARGE LANGUAGE MODELS
   - Neural networks trained to model patterns in data.
   - Autoregressive LLMs predict subsequent tokens.

2. TOKENS
   - Units of text represented by token IDs.
   - Tokens are not necessarily complete words.

3. NEXT-TOKEN PREDICTION
   - Models produce probabilities over possible tokens.
   - A decoding strategy selects the next token.

4. GREEDY DECODING
   - Selects the highest-probability token.

AI ENGINEERING CONNECTION:
Token probabilities and decoding strategies help explain
generation behaviour, output variability and model settings.

NOTE:
This exercise simulates token selection using manually
defined probabilities. It does not use a real LLM.
"""


# --------------------------------------------------
# 1. SIMULATED NEXT-TOKEN PROBABILITIES
# --------------------------------------------------

token_probabilities = {
    " Paris": 0.88,
    " Lyon": 0.05,
    " located": 0.04,
    " known": 0.02,
    " other": 0.01
}


# --------------------------------------------------
# 2. SELECT THE MOST PROBABLE TOKEN
# --------------------------------------------------

next_token = max(
    token_probabilities,
    key=token_probabilities.get
)

print("Selected token:", repr(next_token))
print("Probability:", token_probabilities[next_token])


def predict_next_token(probabilities: dict[str, float]) -> str:
        
    if len(probabilities) == 0:
        raise ValueError("Probabilities dictionary cannot be empty")
    
    highest_prob = max(
        probabilities,
        key=probabilities.get
        )
    return highest_prob

print("Selected token:", repr(predict_next_token(token_probabilities)))

try:
    result = predict_next_token({})
except ValueError as error:
    print('dictionary is empty')


def sample_next_token(probabilities: dict[str, float]) -> str:

    if len(probabilities) == 0:
        raise ValueError("Probabilities dictionary cannot be empty")
    
    import random

    tokens = list(probabilities.keys())
    weights = list(probabilities.values())
    
    sampled_token = random.choices(
        tokens,
        weights,
        k=1
        )[0]
    return sampled_token

# print("Selected token:", repr(sample_next_token(token_probabilities)))

results = {}

for i in range(100000):
    token = sample_next_token(token_probabilities)

    if token not in results:
        results[token] = 0

    results[token] += 1

print(results)


def apply_temperature(
    probabilities: dict[str, float],
    temperature: float
) -> dict[str, float]:
    if temperature < 0:
        return None

    adjusted = {
    token: probability ** (1 / temperature)
    for token, probability in probabilities.items()
    }
    return adjusted

print(apply_temperature(token_probabilities, 1.0))
print(apply_temperature(token_probabilities, 0.5))
print(apply_temperature(token_probabilities, 2.0))