import sys
import os
from unittest.mock import MagicMock

for mod in ["faiss", "torch", "sentence_transformers", "transformers", "rank_bm25"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retriever import Retriever


def test_bm25_tokenization_strips_punctuation():
    tokens = Retriever._tokenize_for_bm25("How do tools? use APIs, ReAct.")
    assert "tools" in tokens
    assert "tools?" not in tokens
    assert "apis" in tokens
    assert "react" in tokens


def test_bm25_tokenization_stopwords_and_casing():
    tokens = Retriever._tokenize_for_bm25("WHAT is SWE-bench?!")
    assert "what" in tokens
    assert "is" in tokens
    assert "swe" in tokens
    assert "bench" in tokens
