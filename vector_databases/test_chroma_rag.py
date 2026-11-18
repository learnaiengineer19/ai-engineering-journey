
"""
LESSON 11: UNIT TESTING CHROMADB RAG
===================================

OBJECTIVE:
Test RAG behaviour without calling real models
or connecting to ChromaDB.

CONCEPTS:
1. unittest
2. unittest.mock.patch
3. Mocking external dependencies
4. Input validation tests
5. Testing retrieved context
6. Testing generated answers

RUN:
python -m unittest discover -s vector_databases \
    -p "test_chroma_rag.py" -v
"""

import unittest
from unittest.mock import patch

from chroma_rag import ask_rag


class TestChromaRAG(unittest.TestCase):

    # ----------------------------------------------
    # TEST 1: VALID QUESTION
    # ----------------------------------------------

    @patch("chroma_rag.generate_answer")
    @patch("chroma_rag.search_documents")
    def test_valid_question(
        self,
        mock_search,
        mock_generate
    ):

        mock_search.return_value = {
            "ids": [["motor_001"]],
            "documents": [[
                "Vehicle accident claims require a police report."
            ]],
            "metadatas": [[{"category": "motor"}]],
            "distances": [[0.6664]]
        }

        mock_generate.return_value = (
            "A police report is required [Document 1]."
        )

        answer = ask_rag(
            "What documents do I need after a car crash?"
        )

        self.assertIn(
            "police report",
            answer
        )

        mock_search.assert_called_once()

        mock_generate.assert_called_once()


    # ----------------------------------------------
    # TEST 2: EMPTY QUERY
    # ----------------------------------------------

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            ask_rag("")


    # ----------------------------------------------
    # TEST 3: INVALID QUERY TYPE
    # ----------------------------------------------

    def test_invalid_query_type(self):
        with self.assertRaises(TypeError):
            ask_rag(123)


    # ----------------------------------------------
    # TEST 4: NO RETRIEVED DOCUMENTS
    # ----------------------------------------------

    @patch("chroma_rag.generate_answer")
    @patch("chroma_rag.search_documents")
    def test_no_documents(self, mock_search, mock_generate):

        mock_search.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        answer = ask_rag("What is my policy coverage?")

        self.assertIn(
            "do not contain enough information",
            answer
        )

        # The LLM must not be called when no documents exist.
        mock_generate.assert_not_called()


    # ----------------------------------------------
    # TEST 5: OLLAMA API FAILURE
    # ----------------------------------------------

    @patch("chroma_rag.search_documents")
    def test_ollama_failure(self, mock_search):

        mock_search.side_effect = ConnectionError(
            "Ollama server unavailable"
        )

        answer = ask_rag("What documents are required?")

        self.assertIsNone(answer)


if __name__ == "__main__":
    unittest.main()
