import sys
import os
from unittest.mock import MagicMock

# Mock heavy/GPU modules if not installed in lightweight testing environment
for mod in ["faiss", "torch", "transformers", "sentence_transformers", "pymupdf4llm", "rank_bm25", "arxiv"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_pipeline import RagPipeline


def test_is_no_info_detection():
    # Exact default marker
    assert RagPipeline._is_no_info("I don't have enough information in the retrieved papers to answer this.")
    # Paraphrased refusals
    assert RagPipeline._is_no_info("I do not have enough information to answer your question.")
    assert RagPipeline._is_no_info("The provided context does not contain any details on this.")
    assert RagPipeline._is_no_info("The retrieved papers do not mention cooking.")
    assert RagPipeline._is_no_info("This is unrelated to LLM agents research.")
    # Valid answers should NOT be flagged as refusal
    assert not RagPipeline._is_no_info("ReAct combines reasoning and acting by generating reasoning traces before tool calls.")


def test_contextualize_query_with_history():
    history = [
        {"role": "user", "content": "Tell me about the Reflexion architecture."},
        {"role": "assistant", "content": "Reflexion uses verbal reinforcement learning..."},
    ]
    # Referential question with pronoun 'it'
    q = "What are its main limitations?"
    ctx_q = RagPipeline._contextualize_query(q, history)
    assert "Reflexion" in ctx_q
    assert "limitations" in ctx_q

    # Standalone question should not be mangled
    standalone_q = "How does tree-of-thought search operate in LLM planning?"
    ctx_standalone = RagPipeline._contextualize_query(standalone_q, history)
    assert ctx_standalone == standalone_q
