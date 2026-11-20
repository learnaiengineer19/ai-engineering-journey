
"""
LESSON 13 - PART 10.6: AUTOMATED REGRESSION TESTS
=================================================

OBJECTIVE:
Use pytest to verify our retrieval evaluation
metrics and citation validation logic.

TEST COVERAGE:
1. Hit Rate@K
2. Reciprocal Rank@K
3. Mean Reciprocal Rank
4. Valid citations
5. Invalid source references
6. Malformed citations
7. Missing citations

RUN FROM PROJECT ROOT:
python -m pytest advanced_rag/evaluation/test_metrics.py -v
"""

import pytest

from advanced_rag.evaluation.evaluate_retrieval import (
    hit_rate_at_k,
    reciprocal_rank_at_k,
    mean_reciprocal_rank,
)

from advanced_rag.evaluation.citation_validator import (
    validate_citations,
)


# --------------------------------------------------
# TEST 1: HIT RATE - RELEVANT CHUNK FOUND
# --------------------------------------------------

def test_hit_rate_relevant_chunk_found():

    retrieved = ["chunk_a", "chunk_b", "chunk_c"]
    relevant = ["chunk_b"]

    assert hit_rate_at_k(retrieved, relevant, k=3) == 1


# --------------------------------------------------
# TEST 2: HIT RATE - RELEVANT CHUNK NOT FOUND
# --------------------------------------------------

def test_hit_rate_relevant_chunk_missing():

    retrieved = ["chunk_a", "chunk_b"]
    relevant = ["chunk_c"]

    assert hit_rate_at_k(retrieved, relevant, k=2) == 0


# --------------------------------------------------
# TEST 3: HIT RATE - OUTSIDE TOP K
# --------------------------------------------------

def test_hit_rate_respects_top_k():

    retrieved = ["chunk_a", "chunk_b", "chunk_c"]
    relevant = ["chunk_c"]

    assert hit_rate_at_k(retrieved, relevant, k=2) == 0


# --------------------------------------------------
# TEST 4: RECIPROCAL RANK
# --------------------------------------------------

@pytest.mark.parametrize(
    "retrieved,relevant,expected",
    [
        (["a", "b", "c"], ["a"], 1.0),
        (["a", "b", "c"], ["b"], 0.5),
        (["a", "b", "c"], ["c"], 1 / 3),
        (["a", "b", "c"], ["d"], 0.0),
    ]
)
def test_reciprocal_rank(retrieved, relevant, expected):

    actual = reciprocal_rank_at_k(
        retrieved,
        relevant,
        k=3
    )

    assert actual == pytest.approx(expected)


# --------------------------------------------------
# TEST 5: MEAN RECIPROCAL RANK
# --------------------------------------------------

def test_mean_reciprocal_rank():

    scores = [1.0, 1 / 3, 0.0]

    expected = 4 / 9

    assert mean_reciprocal_rank(scores) == pytest.approx(
        expected
    )


# --------------------------------------------------
# TEST 6: VALID CITATION
# --------------------------------------------------

def test_valid_citation():

    result = validate_citations(
        "Engine failure is excluded [Source 1].",
        total_sources=3
    )

    assert result["is_valid"] is True


# --------------------------------------------------
# TEST 7: INVALID SOURCE NUMBER
# --------------------------------------------------

def test_invalid_source_number():

    result = validate_citations(
        "Engine failure is excluded [Source 5].",
        total_sources=3
    )

    assert result["is_valid"] is False
    assert result["invalid_source_ids"] == [5]


# --------------------------------------------------
# TEST 8: MALFORMED CITATION
# --------------------------------------------------

def test_malformed_citation():

    result = validate_citations(
        "Engine failure is excluded [Source 1, Page 3].",
        total_sources=3
    )

    assert result["is_valid"] is False
    assert result["malformed_citations"] == [
        "[Source 1, Page 3]"
    ]


# --------------------------------------------------
# TEST 9: MISSING CITATION
# --------------------------------------------------

def test_missing_citation():

    result = validate_citations(
        "Engine failure is excluded.",
        total_sources=3
    )

    assert result["is_valid"] is False
    assert result["has_citations"] is False
