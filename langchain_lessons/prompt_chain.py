
"""
LESSON 18 - PART 2: PROMPT TEMPLATES AND LCEL
==============================================

OBJECTIVE:
Build a reusable prompt -> model -> parser pipeline.

CONCEPTS:
1. ChatPromptTemplate
2. Prompt variables
3. ChatOllama
4. StrOutputParser
5. LCEL pipe operator
6. chain.invoke()

RUN:
python -m langchain_lessons.prompt_chain
"""

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# STEP 1: CREATE A REUSABLE PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an experienced insurance analyst. "
        "Explain insurance concepts clearly and concisely."
    ),
    (
        "human",
        "Explain the insurance concept '{concept}' "
        "using a simple example from {country}."
    )
])


# --------------------------------------------------
# STEP 2: INITIALIZE OUR MODEL
# --------------------------------------------------

model = ChatOllama(
    model="qwen2:7b",
    temperature=0
)


# --------------------------------------------------
# STEP 3: CREATE THE OUTPUT PARSER
# --------------------------------------------------

parser = StrOutputParser()


# --------------------------------------------------
# STEP 4: BUILD THE LCEL PIPELINE
# --------------------------------------------------

chain = prompt | model | parser


# --------------------------------------------------
# STEP 5: EXECUTE WITH DIFFERENT INPUTS
# --------------------------------------------------

examples = [
    {
        "concept": "deductible",
        "country": "Australia"
    },
    {
        "concept": "premium",
        "country": "Australia"
    }
]

sample_input = {
    "concept": "deductible",
    "country": "Australia"
}

formatted = prompt.invoke(sample_input)

print("\nFORMATTED PROMPT:")
print(formatted.to_messages())

response = model.invoke(formatted)

print("\nMODEL RESPONSE TYPE:")
print(type(response).__name__)

parsed = parser.invoke(response)

print("\nPARSED RESPONSE TYPE:")
print(type(parsed).__name__)

for example in examples:

    print("\n==============================")
    print("INPUT:", example)

    result = chain.invoke(example)

    print("\nAI RESPONSE:")
    print(result)

    print("\nRESULT TYPE:", type(result).__name__)
