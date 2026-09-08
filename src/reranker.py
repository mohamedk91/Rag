"""
Cross-encoder reranker: re-scores candidate chunks from first-stage retrieval
(FAISS + BM25) using a model that reads (query, chunk) as a single input.

Why a reranker helps:
  Bi-encoder retrieval (FAISS) embeds query and document independently, so it
  can miss nuanced relevance signals. A cross-encoder sees both texts together
  and is significantly more accurate — but too slow to run on every chunk in the
  index, so we only rerank the top-N candidates from the fast first stage.

Model: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa backbone, ~568M params).
  Fits comfortably on cuda:1 alongside the 0.6B embedding model.
  Outputs logits: positive = relevant, negative = not relevant.
"""
import threading

import _disable_torchvision  # noqa: F401  must run before sentence_transformers import
import torch
from sentence_transformers import CrossEncoder

import config


class Reranker:
    def __init__(self):
        # default_activation_function=Identity() returns RAW LOGITS.
        # Without this, CrossEncoder applies Sigmoid for single-output models,
        # making every score > 0 and the MIN_RERANK_SCORE=0.0 threshold a no-op.
        self.model = CrossEncoder(
            config.RERANKER_MODEL_NAME,
            device=config.RERANKER_DEVICE,
            default_activation_function=torch.nn.Identity(),
        )
        self._lock = threading.Lock()   # one predict() at a time on the GPU

    def rerank(self, query: str, chunks: list, top_k: int = None) -> list:
        """
        Re-score chunks with the cross-encoder and return the top-k.

        Each returned chunk gets a ``rerank_score`` field added.
        The original ``score`` (cosine from FAISS) is preserved for diagnostics.
        """
        top_k = top_k or config.TOP_K
        if not chunks:
            return []

        pairs = [(query, c["chunk_text"]) for c in chunks]
        with self._lock:
            scores = self.model.predict(pairs, activation_fct=torch.nn.Identity())

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]
