"""
End-to-end RAG pipeline: retrieval (with guarded live arXiv fallback) + generation.

Flow per question:
  1. Validate the question (not empty, not too long).
  2. Retrieve top-N candidates from FAISS + BM25 (hybrid, via RRF).
  3. If the best cosine score is below SIMILARITY_THRESHOLD and the fallback is
     allowed (enabled, under the size cap, past the cooldown): try the live
     fallback and re-retrieve.
  4. Rerank the candidates with the cross-encoder down to top-k.
  5. Drop chunks below MIN_RERANK_SCORE. If nothing is left, REFUSE in code.
  6. Otherwise generate a grounded, cited answer (optionally streamed).
"""
import time

from retriever import Retriever
from reranker import Reranker
from generator import Generator
from live_ingest import fallback_fetch_and_ingest
import config
import re


class RagPipeline:
    def __init__(self):
        print("Loading retriever (embedding model + FAISS index + BM25)...")
        self.retriever = Retriever()
        print("Loading reranker (cross-encoder)...")
        self.reranker = Reranker()
        print("Loading generator (Qwen3-8B, 4-bit)...")
        self.generator = Generator()
        self._last_fallback = float("-inf")   # monotonic time of the last fallback attempt
        print("Ready.")

    @staticmethod
    def _result(answer, sources=None, fallback=False, refused=False, best=0.0):
        return {
            "answer": answer,
            "sources": sources or [],
            "used_live_fallback": fallback,
            "refused": refused,
            "best_score": best,
        }

    @staticmethod
    def _is_no_info(answer: str) -> bool:
        """True if the model itself answered with a refusal or no-info disclaimer."""
        low = answer.lower().replace("\u2019", "'")
        no_info_markers = [
            config.NO_INFO_MARKER,
            "i do not have enough information",
            "not enough information in the retrieved",
            "not enough information in the provided",
            "retrieved papers do not provide",
            "retrieved papers do not contain",
            "retrieved papers do not mention",
            "provided papers do not contain",
            "provided context does not contain",
            "context does not mention",
            "context does not provide",
            "do not mention",
            "does not mention",
            "no mention of",
            "none of the retrieved",
            "not mentioned in the retrieved",
            "does not contain information about",
            "unrelated to llm agent",
            "unrelated to llm-based",
            "cannot answer this question based on",
        ]
        return any(m in low for m in no_info_markers)

    @staticmethod
    def _contextualize_query(question: str, history: list = None) -> str:
        """
        If question contains referential words (it, its, this, etc.) or is very short,
        augment it with key terms from the previous conversation turn for retrieval.
        """
        if not history:
            return question

        import re
        words = set(re.findall(r"\b\w+\b", question.lower()))
        pronouns = {"it", "its", "they", "them", "their", "this", "that", "these", "those", "framework", "approach", "method"}
        is_referential = bool(words & pronouns) or len(words) <= 5

        if is_referential:
            last_user = ""
            for turn in reversed(history):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    last_user = turn.get("content", "")
                    break
                elif isinstance(turn, (list, tuple)) and len(turn) == 2 and turn[0]:
                    last_user = str(turn[0])
                    break
            if last_user:
                return f"{last_user} {question}"

        return question

    def _fallback_allowed(self):
        if not config.ALLOW_LIVE_FALLBACK:
            return False, "live fallback is disabled"
        if len(self.retriever.known_arxiv_ids) >= config.MAX_INDEXED_PAPERS:
            return False, "index is at its size cap"
        wait = config.LIVE_FALLBACK_COOLDOWN_SECONDS - (time.monotonic() - self._last_fallback)
        if wait > 0:
            return False, f"cooldown ({wait:.0f}s left)"
        return True, ""

    # ---- shared retrieval + rerank logic (used by both ask and ask_stream) ---

    def _retrieve_and_rerank(self, question, top_k):
        """
        Run the full first-stage retrieval (FAISS + BM25 hybrid) with optional
        live fallback, then rerank with the cross-encoder.

        Returns (usable_chunks, used_live_fallback, best_score).
        """
        retrieved = self.retriever.retrieve(question, top_k=config.RERANK_TOP_N)
        # Best COSINE score (from FAISS), not RRF — used for fallback decision.
        best_score = max((c["score"] for c in retrieved), default=0.0)
        used_live_fallback = False

        if best_score < config.SIMILARITY_THRESHOLD:
            allowed, why = self._fallback_allowed()
            if allowed:
                self._last_fallback = time.monotonic()   # counts even if it finds nothing
                try:
                    n_new = fallback_fetch_and_ingest(question, self.retriever)
                except Exception as e:
                    print(f"  ! live fallback failed: {e}")
                    n_new = 0
                if n_new > 0:
                    used_live_fallback = True
                    retrieved = self.retriever.retrieve(question, top_k=config.RERANK_TOP_N)
                    best_score = max((c["score"] for c in retrieved), default=0.0)
            else:
                print(f"  (live fallback skipped: {why})")

        # Cross-encoder reranking: top-N candidates → top-k
        reranked = self.reranker.rerank(question, retrieved, top_k=top_k)
        usable = [c for c in reranked if c["rerank_score"] >= config.MIN_RERANK_SCORE]
        return usable, used_live_fallback, best_score

    # ---- non-streaming ask ------------------------------------------------

    @staticmethod
    def _extract_cited_arxiv_ids(answer: str) -> set:
        """Extract arXiv-style IDs cited in the generated answer."""
        return set(re.findall(r"\b\d{4}\.\d{4,5}\b", answer or ""))

    @staticmethod
    def _has_invalid_citations(answer: str, sources: list) -> bool:
        """True if the answer cites an arXiv ID that was not retrieved."""
        cited = RagPipeline._extract_cited_arxiv_ids(answer)
        if not cited:
            return False

        allowed = {str(s["arxiv_id"]) for s in sources}
        return not cited.issubset(allowed)


    
    def ask(self, question: str, history: list = None, top_k: int = None):
        question = (question or "").strip()
        if not question:
            return self._result(config.EMPTY_QUESTION_MESSAGE, refused=True)
        if len(question) > config.MAX_QUESTION_CHARS:
            return self._result(
                f"Please keep questions under {config.MAX_QUESTION_CHARS} characters.", refused=True
            )

        top_k = top_k or config.TOP_K
        search_query = self._contextualize_query(question, history)
        usable, used_live_fallback, best_score = self._retrieve_and_rerank(search_query, top_k)

        if not usable:
            return self._result(
                config.NO_INFO_MESSAGE, fallback=used_live_fallback, refused=True, best=best_score
            )

        sources = [
            {"title": c["title"], "arxiv_id": c["arxiv_id"], "section": c["section"],
            "score": c["score"], "rerank_score": c["rerank_score"]}
            for c in usable
        ]

        answer = self.generator.generate(question, usable, history=history)

        if self._is_no_info(answer):
            return self._result(answer, fallback=used_live_fallback, refused=True, best=best_score)

        if self._has_invalid_citations(answer, sources):
            return self._result(
                config.NO_INFO_MESSAGE,     
                fallback=used_live_fallback,
                refused=True,
                best=best_score,
            )

        return self._result(answer, sources, fallback=used_live_fallback, best=best_score)

    # ---- streaming ask ----------------------------------------------------

    def ask_stream(self, question: str, history: list = None, top_k: int = None):
        """
        Streaming version of ask(). Yields result dicts with a growing 'answer'
        field as tokens are generated.

        Retrieval, fallback, and reranking happen synchronously before streaming
        begins — only the LLM generation is streamed.
        """
        question = (question or "").strip()
        if not question:
            yield self._result(config.EMPTY_QUESTION_MESSAGE, refused=True)
            return
        if len(question) > config.MAX_QUESTION_CHARS:
            yield self._result(
                f"Please keep questions under {config.MAX_QUESTION_CHARS} characters.", refused=True
            )
            return

        top_k = top_k or config.TOP_K
        search_query = self._contextualize_query(question, history)
        usable, used_live_fallback, best_score = self._retrieve_and_rerank(search_query, top_k)

        if not usable:
            yield self._result(
                config.NO_INFO_MESSAGE, fallback=used_live_fallback, refused=True, best=best_score
            )
            return

        sources = [
            {"title": c["title"], "arxiv_id": c["arxiv_id"], "section": c["section"],
             "score": c["score"], "rerank_score": c["rerank_score"]}
            for c in usable
        ]

        # Stream the LLM generation token by token
        partial = ""
        for token in self.generator.generate_stream(question, usable, history=history):
            partial += token
            is_refusal = self._is_no_info(partial)
            yield self._result(
                partial,
                sources=[] if is_refusal else sources,
                fallback=used_live_fallback,
                refused=is_refusal,
                best=best_score,
            )

        if self._has_invalid_citations(partial, sources):
            yield self._result(
                config.NO_INFO_MESSAGE,
                fallback=used_live_fallback,
                refused=True,
                best=best_score,
            )


if __name__ == "__main__":
    pipeline = RagPipeline()
    print("\nRAG pipeline ready. Type a question (or 'quit').\n")
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        result = pipeline.ask(q)
        if result["used_live_fallback"]:
            print("\n[fetched fresh papers from arXiv for this question]")
        print("\nAnswer:\n" + result["answer"])
        if result["sources"]:
            print("\nRetrieved passages (the answer may use only some of these):")
            for s in result["sources"]:
                print(f"  - {s['title']} (arXiv:{s['arxiv_id']}, {s['section']}, "
                      f"cosine={s['score']:.3f}, rerank={s['rerank_score']:.3f})")
        else:
            print(f"\n(best retrieval score was {result['best_score']:.3f})")
        print()