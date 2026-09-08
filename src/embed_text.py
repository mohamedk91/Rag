"""
The ONE place that decides what text gets embedded for a chunk.

Both 4_embed_index.py (initial index) and retriever.add_chunks (live-ingested
papers) must call doc_text(), otherwise chunks from the two sources would be
embedded in different formats and compared unfairly against the same query.

The stored chunk_text stays clean. The generator reads the plain text; only the
embedder sees the title/section header.
"""

MAX_TITLE_CHARS = 200   # keeps a pathological title from eating the token budget


def doc_text(chunk: dict) -> str:
    """Return the text to embed for one chunk record."""
    title = chunk["title"].strip()[:MAX_TITLE_CHARS]
    section = chunk["section"].strip()

    header = f"Title: {title}"
    if section and section != "Preamble":      # "Preamble" says nothing useful about the content
        header += f"\nSection: {section}"
    return f"{header}\n\n{chunk['chunk_text']}"
