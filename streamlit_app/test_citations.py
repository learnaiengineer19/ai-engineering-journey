
"""
LESSON 14 - PART 9: CITATION UNIT TESTS

OBJECTIVE:
Test citation-number validation independently
of Streamlit, Ollama and ChromaDB.

RUN:
python -m pytest streamlit_app/test_citations.py -v
"""

from streamlit_app.citations import analyse_citations


def test_valid_citation():

    sources = [{}, {}, {}]

    result = analyse_citations(
        answer="Engine damage is excluded [Source 1].",
        sources=sources
    )

    assert result["valid_numbers"] == [1]
    assert result["invalid_numbers"] == []


def test_invalid_citation():

    sources = [{}, {}, {}]

    result = analyse_citations(
        answer="See [Source 9].",
        sources=sources
    )

    assert result["invalid_numbers"] == [9]


def test_missing_citation():

    sources = [{}, {}]

    result = analyse_citations(
        answer="No relevant information was found.",
        sources=sources
    )

    assert result["has_citations"] is False


def test_duplicate_citations():

    sources = [{}, {}]

    result = analyse_citations(
        answer="See [Source 1] and [Source 1].",
        sources=sources
    )

    assert result["valid_numbers"] == [1]
