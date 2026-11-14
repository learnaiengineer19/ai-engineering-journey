
"""
LESSON 06: CLASSES, OBJECTS, TYPE HINTS AND TESTING
==================================================

OBJECTIVE:
Understand object-oriented programming and learn
how to organize Python code into reusable classes.

CONCEPTS COVERED:

1. CLASS
   - A blueprint used to create objects.
   - Defined using the 'class' keyword.

2. OBJECT
   - An instance of a class.
   - Each object can store its own data.

3. ATTRIBUTES
   - Variables that belong to an object.
   - Example: self.filename

4. METHODS
   - Functions defined inside a class.
   - Example: document.word_count()

5. __init__()
   - Initializes an object's attributes.

6. SELF
   - Refers to the current object instance.

AI ENGINEERING CONNECTION:
Classes are used to organize document processors,
retrievers, LLM clients, AI agents and evaluation systems.

"""




from document import Document

# --------------------------------------------------
# 2. CREATE OBJECTS
# --------------------------------------------------

document1 = Document(
    "rag.txt",
    "RAG combines retrieval and generation"
)

document2 = Document(
    "agents.txt",
    "AI agents can use tools"
)

document1.content = "Updated content"

# --------------------------------------------------
# 3. ACCESS OBJECT ATTRIBUTES AND METHODS
# --------------------------------------------------

print("Document 1:", document1.filename)
print("Content:", document1.content)
print("Word count:", document1.word_count())
print(document1.character_count())
print(document1.get_metadata())

print("\nDocument 2:", document2.filename)
print("Content:", document2.content)
print("Word count:", document2.word_count())

