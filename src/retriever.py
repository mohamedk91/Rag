"""
Retrieval: embed a user query and fetch the top-k most similar chunks from the
FAISS index, combined with BM25 sparse retrieval via Reciprocal Rank Fusion.
Supports adding new chunks at runtime (live fallback) in a thread-safe,
crash-safe way.

Invariant: FAISS row i  <->  self.chunks[i]  <->  line i of chunks_store.jsonl
"""
import json
import re
import threading

import _disable_torchvision  # noqa: F401  must run before sentence_transformers import
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
from embed_text import doc_text
from fileio import atomic_write_faiss, atomic_write_jsonl

_BM25_TOKEN_RE = re.compile(r"\w+")   # strips punctuation: "tools?" -> "tools"


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        self.model.max_seq_length = config.EMBED_MAX_SEQ_LENGTH
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)

        self.chunks = []
        with open(config.CHUNKS_STORE_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))

        self._reconcile()
        self.known_arxiv_ids = {c["arxiv_id"] for c in self.chunks}

        self._model_lock = threading.Lock()   # one encode() at a time on the GPU
        self._index_lock = threading.Lock()   # guards index + chunks + known_arxiv_ids + bm25

        self._build_bm25()

    def _reconcile(self):
        """Repair or reject a chunk file that does not line up with the index."""
        n_index, n_chunks = self.index.ntotal, len(self.chunks)
        if n_chunks < n_index:
            raise RuntimeError(
                f"Index has {n_index} vectors but the chunk file has only {n_chunks} lines. "
                "They are out of sync; rebuild with 3_chunk.py and 4_embed_index.py."
            )
        if n_chunks > n_index:
            # Left over from a crash between the two writes. Extra lines are orphans: ignore them.
            print(f"  ! chunk file has {n_chunks - n_index} orphan line(s) from an interrupted save; ignoring them.")
            self.chunks = self.chunks[:n_index]

    # ---- BM25 index -------------------------------------------------------

    @staticmethod
    def _tokenize_for_bm25(text: str) -> list:
        """Regex tokenizer: lowercases and strips punctuation so 'tools?' matches 'tools'."""
        return _BM25_TOKEN_RE.findall(text.lower())

    def _build_bm25(self):
        """Build the BM25 index from current chunks. Called during init and after add_chunks."""
        if self.chunks:
            corpus = [self._tokenize_for_bm25(c["chunk_text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None

    # ---- embedding -------------------------------------------------------

    def _embed_queries(self, texts):
        """Queries get the task instruction prefix."""
        with self._model_lock:
            vecs = self.model.encode(
                texts, prompt=config.QUERY_INSTRUCTION,
                convert_to_numpy=True, normalize_embeddings=True,
            )
        return vecs.astype(np.float32)

    def _embed_documents(self, texts):
        """Documents (chunks, abstracts) get NO prefix."""
        with self._model_lock:
            vecs = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True,
            )
        return vecs.astype(np.float32)

    def similarity_to_query(self, query: str, texts: list) -> list:
        """Cosine similarity between a query and each text (used by the live fallback)."""
        q = self._embed_queries([query])
        d = self._embed_documents(texts)
        return (d @ q.T).ravel().tolist()

    # ---- search ----------------------------------------------------------

    def retrieve(self, query: str, top_k: int = None):
        """
        Hybrid retrieval: FAISS (dense) + BM25 (sparse), fused with RRF.

        Each result carries:
          - score:     cosine similarity from FAISS (0.0 if the chunk came only from BM25)
          - rrf_score: combined Reciprocal Rank Fusion score used for ordering
        """
        top_k = top_k or config.TOP_K
        n_candidates = max(top_k, config.RERANK_TOP_N)
        query_vec = self._embed_queries([query])       # slow part: outside the index lock

        with self._index_lock:
            if not self.chunks or self.index.ntotal == 0:
                return []

            # --- Dense retrieval (FAISS) ---
            n_search = min(n_candidates, self.index.ntotal)
            scores, ids = self.index.search(query_vec, n_search)
            dense = {}
            for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
                if idx == -1:
                    continue
                dense[int(idx)] = {"rank": rank, "score": float(score)}

            # --- Sparse retrieval (BM25) ---
            sparse = {}
            if self.bm25 is not None:
                tokens = self._tokenize_for_bm25(query)
                bm25_scores = self.bm25.get_scores(tokens)
                top_indices = bm25_scores.argsort()[-n_candidates:][::-1]
                for rank, idx in enumerate(top_indices, 1):
                    if bm25_scores[idx] > 0:
                        sparse[int(idx)] = {"rank": rank}

            # --- Reciprocal Rank Fusion ---
            all_ids = set(dense.keys()) | set(sparse.keys())
            fused = []
            for idx in all_ids:
                rrf_score = 0.0
                if idx in dense:
                    rrf_score += 1.0 / (config.RRF_K + dense[idx]["rank"])
                if idx in sparse:
                    rrf_score += 1.0 / (config.RRF_K + sparse[idx]["rank"])

                chunk = self.chunks[idx]
                fused.append({
                    "score": dense[idx]["score"] if idx in dense else 0.0,
                    "rrf_score": rrf_score,
                    "arxiv_id": chunk["arxiv_id"],
                    "title": chunk["title"],
                    "section": chunk["section"],
                    "chunk_text": chunk["chunk_text"],
                })

            fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:top_k]

    # ---- adding chunks at runtime ------------------------------------------

    def add_chunks(self, new_chunk_records: list, persist: bool = True) -> int:
        """
        Embed and add chunks from new papers. Returns how many chunks were added.
        - Papers that are already indexed are skipped (safe to call twice).
        - If saving to disk fails, the in-memory index is rolled back and the error is raised.
        """
        if not new_chunk_records:
            return 0

        with self._index_lock:
            fresh = [c for c in new_chunk_records if c["arxiv_id"] not in self.known_arxiv_ids]
        if not fresh:
            return 0

        vecs = self._embed_documents([doc_text(c) for c in fresh])   # slow: outside the lock

        with self._index_lock:
            # Re-check: another ingest may have added the same paper while we were embedding.
            keep = [i for i, c in enumerate(fresh) if c["arxiv_id"] not in self.known_arxiv_ids]
            if not keep:
                return 0
            fresh = [fresh[i] for i in keep]
            vecs = vecs[keep]

            n_old = self.index.ntotal
            self.index.add(vecs)
            self.chunks.extend(fresh)

            if persist:
                try:
                    atomic_write_jsonl(config.CHUNKS_STORE_PATH, self.chunks)   # chunk file FIRST
                    atomic_write_faiss(self.index, config.FAISS_INDEX_PATH)     # index SECOND
                except Exception:
                    self.index.remove_ids(faiss.IDSelectorRange(n_old, n_old + len(fresh)))
                    del self.chunks[n_old:]
                    raise

            self.known_arxiv_ids.update(c["arxiv_id"] for c in fresh)
            self._build_bm25()   # rebuild BM25 to include the new chunks
            return len(fresh)