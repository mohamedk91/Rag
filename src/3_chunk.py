"""
Step 3: Chunk the markdown papers (batch, over the initial fetched corpus).

Splitting/sub-splitting logic lives in chunking.py so it can be reused by
live_ingest.py for papers fetched on-the-fly at query time.

Output: data/chunks/chunks.jsonl
  one JSON object per line: {arxiv_id, title, section, chunk_text}

After writing, prints a quality report (what was dropped and why, and how many
section labels still look suspicious) so you can judge the chunker without
opening the file.
"""
import json
import os
from collections import Counter

import config
from chunking import chunk_markdown_text, looks_like_author_line


def chunk_all_papers():
    os.makedirs(config.CHUNKS_DIR, exist_ok=True)

    with open(config.METADATA_JSON) as f:
        papers = {p["arxiv_id"]: p for p in json.load(f)}

    stats = Counter()
    sections = Counter()
    n_chunks = 0
    with open(config.CHUNKS_JSONL, "w", encoding="utf-8") as out_f:
        for arxiv_id, paper in papers.items():
            md_path = os.path.join(config.MD_DIR, f"{arxiv_id}.md")
            if not os.path.exists(md_path):
                continue

            with open(md_path, encoding="utf-8") as f:
                md_text = f.read()

            records = chunk_markdown_text(md_text, arxiv_id, paper["title"], stats=stats)
            for record in records:
                out_f.write(json.dumps(record) + "\n")
                sections[record["section"]] += 1
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {len(papers)} papers to {config.CHUNKS_JSONL}")

    print("\n--- chunk quality report ---")
    print(f"  kept                      : {stats['kept']}")
    print(f"  dropped (reference-like)  : {stats['dropped_reference_like']}")
    print(f"  dropped (too short)       : {stats['dropped_too_short']}")
    print(f"  dropped (table)           : {stats['dropped_table']}")
    print(f"  dropped (symbol soup)     : {stats['dropped_symbols']}")
    print(f"  sections with oversized table rows removed: {stats['sections_with_oversized_table_rows']}")
    print(f"  labelled 'Preamble'       : {stats['kept_preamble']}")
    long_labels = sum(n for s, n in sections.items() if len(s) > 60)
    author_labels = sum(n for s, n in sections.items() if looks_like_author_line(s))
    print(f"  labels longer than 60 chars   : {long_labels}   (should be near 0)")
    print(f"  labels that look like authors : {author_labels}   (should be near 0)")
    print("  most common section labels:")
    for label, n in sections.most_common(8):
        print(f"    {n:5d}  {label[:70]}")


if __name__ == "__main__":
    chunk_all_papers()