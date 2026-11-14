
"""
PROMPT UTILITIES
================

Reusable functions for validating LLM prompts.

This module can be imported into other Python files.
"""


def validate_prompt(prompt):
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")

    cleaned_prompt = prompt.strip()

    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty")

    return cleaned_prompt
