import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import paper_ranking, score_question, check_citations


def test_paper_ranking_order():
    chunks = [
        {"arxiv_id": "2310.001", "score": 0.9},
        {"arxiv_id": "2310.001", "score": 0.85},
        {"arxiv_id": "2310.002", "score": 0.8},
        {"arxiv_id": "2310.003", "score": 0.75},
    ]
    ranked = paper_ranking(chunks)
    assert ranked == ["2310.001", "2310.002", "2310.003"]


def test_score_question_hit():
    chunks = [
        {"arxiv_id": "2310.001", "score": 0.9, "rerank_score": 2.5},
        {"arxiv_id": "2310.002", "score": 0.8, "rerank_score": 1.2},
    ]
    metrics = score_question(chunks, relevant_ids=["2310.001"], top_k_chunks=5)
    assert metrics["hit@1"] is True
    assert metrics["hit@3"] is True
    assert metrics["mrr"] == 1.0
    assert metrics["llm_sees_it"] is True


def test_check_citations():
    sources = [
        {"arxiv_id": "2310.03123", "title": "Paper A"},
        {"arxiv_id": "2401.12345", "title": "Paper B"},
    ]
    answer = "Based on ReAct (2310.03123) and Reflexion (9999.99999), agents can plan."
    cit = check_citations(answer, sources)
    assert "2310.03123" in cit["valid_citations"]
    assert "9999.99999" in cit["hallucinated_citations"]
    assert cit["citation_fidelity"] == 0.5
