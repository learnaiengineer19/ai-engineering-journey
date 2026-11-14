
"""
LESSON 05: MODULES, PACKAGES AND FILE HANDLING
==============================================

OBJECTIVE:
Learn how to organize reusable Python code,
import modules, manage dependencies and read/write files.

CONCEPTS COVERED:

1. MODULES
   - A Python file containing reusable code.
   - Example: prompt_utils.py

2. IMPORTS
   - Allow one Python file to use functions from another.
   - Example: from prompt_utils import validate_prompt

3. PACKAGES
   - Organize related modules into directories.
   - Packages help structure larger Python applications.

4. VIRTUAL ENVIRONMENTS
   - Isolate project-specific dependencies.
   - Our environment is stored in .venv.

5. FILE HANDLING
   - Read and write files using open().
   - Use context managers with 'with'.

6. JSON FILES
   - Store structured data in JSON format.
   - json.load() reads JSON from a file.
   - json.dump() writes JSON to a file.

AI ENGINEERING CONNECTION:
   - Separate retrieval, generation and validation modules.
   - Store and read documents for RAG applications.
   - Manage third-party AI libraries.
   - Read structured configuration and evaluation data.

"""


print("Lesson 05 is running!")

# --------------------------------------------------
# 1. IMPORT A FUNCTION FROM ANOTHER MODULE
# --------------------------------------------------

from prompt_utils import validate_prompt


# --------------------------------------------------
# 2. CALL THE IMPORTED FUNCTION
# --------------------------------------------------

prompt = "   Explain Retrieval-Augmented Generation   "

cleaned_prompt = validate_prompt(prompt)

print("Original prompt:", repr(prompt))
print("Validated prompt:", repr(cleaned_prompt))

# --------------------------------------------------
# 3. READING TEXT FILES
# --------------------------------------------------

from pathlib import Path

# Get the directory containing lesson_05.py
file_path = Path(__file__).parent / "sample_document.txt"

# Open and read the document
with open(file_path, "r", encoding="utf-8") as file:
    document_content = file.read()

print("\nDocument content:")
print(document_content)

print("Document type:", type(document_content))
print("Document length:", len(document_content))

# --------------------------------------------------
# 4. HANDLING FILE ERRORS
# --------------------------------------------------

missing_file = Path(__file__).parent / "missing_document.txt"

try:
    with open(missing_file, "r", encoding="utf-8") as file:
        content = file.read()

except FileNotFoundError as error:
    print("Document could not be found:", error)

else:
    print("Document loaded successfully")
    print(content)

finally:
    print("File operation completed")

from document_loader import load_document

content = load_document(missing_file)
print("Loaded content:", content)
print(load_document.__doc__)


# --------------------------------------------------
# 5. WRITING JSON TO A FILE
# --------------------------------------------------

import json
from pathlib import Path

conversation = {
    "model": "example-model",
    "messages": [
        {
            "role": "user",
            "content": "What is RAG?"
        },
        {
            "role": "assistant",
            "content": "RAG combines retrieval and generation."
        }
    ]
}

# Create the file path
json_path = Path(__file__).parent / "conversation.json"

# Write Python dictionary to JSON file
with open(json_path, "w", encoding="utf-8") as file:
    json.dump(conversation, file, indent=4)

print("Conversation saved successfully!")


"""
LESSON 05: JSON CONFIGURATION EXERCISE
=====================================

OBJECTIVE:
Create a Python dictionary, write it to a JSON file,
read it back, and access the loaded configuration.

CONCEPTS:
- Python dictionaries and data types
- json.dump() and json.load()
- pathlib.Path
- File handling using with
- JSON serialization and deserialization

AI ENGINEERING CONNECTION:
AI applications frequently store model configurations
in JSON files, including model names, numeric parameters,
and boolean feature settings.
"""

import json
from pathlib import Path


# --------------------------------------------------
# 1. CREATE MODEL CONFIGURATION
# --------------------------------------------------

model_config = {
    "model": "example-model",
    "temperature": 0.7,
    "max_tokens": 1000,
    "streaming": False,
}


# --------------------------------------------------
# 2. WRITE CONFIGURATION TO JSON FILE
# --------------------------------------------------

model_config_path = Path(__file__).parent / "model_config.json"

with open(model_config_path, "w", encoding="utf-8") as file:
    json.dump(model_config, file, indent=4)


# --------------------------------------------------
# 3. READ CONFIGURATION FROM JSON FILE
# --------------------------------------------------

with open(model_config_path, "r", encoding="utf-8") as file:
    loaded_config = json.load(file)


# --------------------------------------------------
# 4. PRINT CONFIGURATION VALUES
# --------------------------------------------------

print("Model:", loaded_config["model"])
print("Temperature:", loaded_config["temperature"])
print("Max tokens:", loaded_config["max_tokens"])
print("Streaming:", loaded_config["streaming"])


# --------------------------------------------------
# 5. VERIFY DATA TYPES
# --------------------------------------------------

print("\nData types:")

for key, value in loaded_config.items():
    print(key, ":", type(value))
