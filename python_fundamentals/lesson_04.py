
"""
LESSON 04: ERROR HANDLING AND INPUT VALIDATION
==============================================

OBJECTIVE:
Learn how Python handles errors, validates inputs,
and prevents unexpected failures.

CONCEPTS COVERED:

1. EXCEPTIONS
   - Errors that interrupt normal program execution.
   - Common exceptions:
       TypeError         : Incorrect data type
       ValueError        : Invalid value
       KeyError          : Dictionary key does not exist
       IndexError        : List index does not exist
       ZeroDivisionError : Division by zero
       FileNotFoundError : File does not exist

2. TRY / EXCEPT
   - try    : Executes code that might fail.
   - except : Handles a matching exception.

3. RAISE
   - Manually raises an exception.
   - Useful for enforcing input requirements.

4. ISINSTANCE()
   - Checks whether a value belongs to a specific type.
   - Example: isinstance("Hello", str) -> True

5. STRING VALIDATION
   - strip() removes leading and trailing whitespace.
   - An empty string is considered False in Python.
   - not prompt.strip() detects empty or whitespace-only text.

6. ELSE
   - Runs when the try block completes without an exception.

7. FINALLY
   - Runs whether an exception occurs or not.
   - Commonly used for resource cleanup.

8. INPUT VALIDATION
   - Check inputs before processing.
   - Distinguish incorrect types from invalid values.

AI ENGINEERING CONNECTION:
   - Validate user prompts before calling an LLM.
   - Handle invalid API requests and responses.
   - Prevent malformed inputs from reaching retrieval systems.
   - Handle missing documents and unexpected data.
   - Use exception handling to build reliable AI applications.

"""


# --------------------------------------------------
# 1. BASIC EXCEPTION HANDLING
# --------------------------------------------------

try:
    result = 10 / 0
    print(result)

except ZeroDivisionError:
    print("Cannot divide by zero")

print("Program continues...")


# --------------------------------------------------
# 2. INPUT VALIDATION USING RAISE
# --------------------------------------------------

def process_documents(documents):

    # Check whether documents is a list
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list")

    # Check whether the list contains any documents
    if len(documents) == 0:
        raise ValueError("Documents list cannot be empty")

    return len(documents)


print("\nDocument count:", process_documents(["doc1", "doc2"]))


# --------------------------------------------------
# 3. VALIDATE USER PROMPTS
# --------------------------------------------------

def validate_prompt(prompt):

    # Check that the prompt is a string
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")

    # Remove leading and trailing whitespace
    cleaned_prompt = prompt.strip()

    # Reject empty or whitespace-only strings
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty")

    # Return the validated prompt
    return cleaned_prompt


# --------------------------------------------------
# 4. TEST MULTIPLE PROMPTS WITH TRY / EXCEPT
# --------------------------------------------------

test_prompts = [
    "  Explain RAG  ",
    "",
    "   ",
    123,
    None,
    "What is an LLM?"
]

print("\nTesting prompt validation:")

for prompt in test_prompts:

    try:
        result = validate_prompt(prompt)
        print("Valid:", repr(result))

    except (TypeError, ValueError) as error:
        print("Invalid:", str(error))


# --------------------------------------------------
# 5. TRY / EXCEPT / ELSE / FINALLY
# --------------------------------------------------

print("\nTesting exception handling blocks:")

try:
    number = int("abc")

except ValueError:
    print("Invalid number")

else:
    print("Conversion successful:", number)

finally:
    print("Execution completed")


# --------------------------------------------------
# 6. DIFFERENCE BETWEEN EMPTY AND MISSING INPUT
# --------------------------------------------------

# An empty list is still a valid argument.
# But our process_documents() function deliberately
# rejects empty lists using ValueError.

print("\nTesting empty documents:")

try:
    process_documents([])

except ValueError as error:
    print("Validation failed:", error)


# Calling process_documents() without an argument
# would raise TypeError because the parameter is required.

# process_documents()


# --------------------------------------------------
# KEY TAKEAWAYS
# --------------------------------------------------

# 1. try/except handles expected exceptions.
# 2. raise allows developers to enforce validation rules.
# 3. TypeError and ValueError represent different problems.
# 4. strip() helps reject whitespace-only prompts.
# 5. else runs only if try completes without an exception.
# 6. finally executes whether an exception occurs or not.
# 7. Catch specific exceptions instead of hiding all errors.
# 8. Handle invalid inputs before calling AI services.
