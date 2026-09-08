"""
Step 4: Embed chunks and build a FAISS index.

Model choice (locked in during planning): Qwen3-Embedding-0.6B
  - current-gen, strong retrieval quality for its size
  - English academic text, no need for the multilingual giants
  - small enough to run on cuda:1 alongside Qwen3-8B on cuda:0

Output:
  - data/faiss.index          FAISS index (flat, cosine via inner product on
                               normalized vectors)
  - data/chunks_store.jsonl   chunk records in the SAME ORDER as the FAISS ids,
                               so faiss result index i -> line i of this file
"""
import json

import _disable_torchvision  # noqa: F401  must run before sentence_transformers import
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from embed_text import doc_text
from fileio import atomic_write_faiss, atomic_write_jsonl


def build_index():
    chunks = []
    with open(config.CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise SystemExit(f"No chunks found in {config.CHUNKS_JSONL}. Run 3_chunk.py first.")

    print(f"Loaded {len(chunks)} chunks. Embedding with {config.EMBEDDING_MODEL_NAME} ...")

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
    model.max_seq_length = config.EMBED_MAX_SEQ_LENGTH

    # Embed "Title / Section / text", not the bare text. Stored chunk_text is unchanged.
    texts = [doc_text(c) for c in chunks]

    # Safety check: how many chunks would still be truncated by the embedder?
    lengths = [len(model.tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    too_long = sum(1 for n in lengths if n > config.EMBED_MAX_SEQ_LENGTH)
    print(f"Token lengths: max={max(lengths)}, mean={sum(lengths) / len(lengths):.0f}, "
          f"over limit ({config.EMBED_MAX_SEQ_LENGTH}): {too_long}")
    if too_long:
        print("WARNING: some chunks (text + title/section header) will be truncated. "
              "Lower MAX_CHUNK_TOKENS or raise EMBED_MAX_SEQ_LENGTH.")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(embeddings)

    # Chunk file FIRST, index SECOND. A crash between the two leaves extra
    # lines in the chunk file (harmless, ignored on load), never an index that
    # points at missing text. Row i of the index == line i of the chunk file.
    atomic_write_jsonl(config.CHUNKS_STORE_PATH, chunks)
    atomic_write_faiss(index, config.FAISS_INDEX_PATH)

    print(f"Indexed {index.ntotal} vectors (dim={dim}).")
    print(f"FAISS index saved to:  {config.FAISS_INDEX_PATH}")
    print(f"Chunk store saved to:  {config.CHUNKS_STORE_PATH}")


if __name__ == "__main__":
    build_index()