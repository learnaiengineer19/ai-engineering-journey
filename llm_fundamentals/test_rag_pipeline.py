
"""
LESSON 10: RAG PIPELINE UNIT TESTS
=================================

OBJECTIVE:
Test our reusable RAG function.

CONCEPTS:
1. unittest
2. Input validation
3. Mocking external dependencies
4. Deterministic tests

AI ENGINEERING CONNECTION:
Reliable AI applications need tests that can
run without depending on live model responses.
"""

import unittest
from unittest.mock import patch, MagicMock

from rag_pipeline import ask_rag


class TestRAGPipeline(unittest.TestCase):

    # TEST 1: Empty query
    def test_empty_query(self):

        with self.assertRaises(ValueError):
            ask_rag("")

    # TEST 2: Whitespace-only query
    def test_whitespace_query(self):

        with self.assertRaises(ValueError):
            ask_rag("   ")

    # TEST 3: Invalid data type
    def test_invalid_query_type(self):

        with self.assertRaises(TypeError):
            ask_rag(123)

    
    # TEST 4: Generate answer using mocked dependencies
    @patch("rag_pipeline.chat")
    @patch("rag_pipeline.search_saved_documents")
    def test_rag_with_mocked_llm(
        self,
        mock_search,
        mock_chat
    ):

        # Step 1: Simulate retrieved documents
        mock_search.return_value = [
            (
                "Vehicle accident claims require a police report.",
                0.88
            ),
            (
                "Vehicle collision claims require photographs.",
                0.82
            )
        ]

        # Step 2: Simulate the LLM response
        mock_response = MagicMock()

        mock_response.message.content = (
            "According to Document 1, "
            "a police report is required."
        )

        mock_chat.return_value = mock_response

        # Step 3: Call our actual RAG function
        answer = ask_rag(
            "What documents do I need after a car crash?"
        )

        # Step 4: Check the generated answer
        self.assertEqual(
            answer,
            "According to Document 1, "
            "a police report is required."
        )

        # Step 5: Verify the LLM was called
        mock_chat.assert_called_once()

        # Step 6: Inspect the context sent to the LLM
        messages = mock_chat.call_args.kwargs["messages"]

        self.assertIn(
            "[Document 1]",
            messages[1]["content"]
        )

        self.assertIn(
            "Vehicle accident claims require a police report.",
            messages[1]["content"]
        )



if __name__ == "__main__":
    unittest.main()
