
"""
LESSON 14 - PART 8: APPLICATION ERRORS
=====================================

OBJECTIVE:
Define application-specific exceptions so that
our Streamlit frontend can display meaningful
errors without exposing technical internals.

CONCEPTS:
1. Custom exception classes
2. Exception inheritance
3. Separating technical failures from UI messages
"""


class RAGServiceError(Exception):
    """Base exception for known RAG service failures."""


class OllamaUnavailableError(RAGServiceError):
    """Ollama service could not be reached."""


class PolicyIndexUnavailableError(RAGServiceError):
    """The insurance policy search index is unavailable."""


class RAGProcessingError(RAGServiceError):
    """The RAG pipeline failed during processing."""
