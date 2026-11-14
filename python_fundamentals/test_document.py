
"""
LESSON 06: UNIT TESTING
=======================

OBJECTIVE:
Automatically verify that the Document class
behaves as expected.

CONCEPTS:
- unittest
- TestCase
- assertEqual()
- Test discovery
"""

import unittest

from document import Document


class TestDocument(unittest.TestCase):

    def setUp(self):
        self.document = Document(
            "test.txt",
            "Hello AI Engineer"
        )

    def test_word_count(self):
        self.assertEqual(self.document.word_count(), 3)

    def test_character_count(self):
        self.assertEqual(self.document.character_count(), 17)

    def test_get_metadata(self):
        self.assertEqual(self.document.get_metadata(), { "filename": "test.txt", "word_count": 3, "character_count": 17 })


if __name__ == "__main__":
    unittest.main()
