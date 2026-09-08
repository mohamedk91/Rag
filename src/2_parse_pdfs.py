"""
Step 2: PDF -> Markdown conversion.

Why: raw PDF text extraction loses reading order on 2-column academic papers
and mixes headers/footers/references into the body. pymupdf4llm infers heading
structure from font size/position, giving us clean Markdown with '#'-style
headings that map to paper sections (Abstract, Introduction, Method, ...).
This is what step 3 (chunking) relies on.

Output: data/papers_md/<arxiv_id>.md  (one markdown file per paper)
"""
import json
import os

import pymupdf4llm

import config


def parse_all_pdfs():
    os.makedirs(config.MD_DIR, exist_ok=True)

    with open(config.METADATA_JSON) as f:
        papers = json.load(f)

    ok, failed = 0, []
    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        pdf_path = paper["pdf_path"]
        md_path = os.path.join(config.MD_DIR, f"{arxiv_id}.md")

        print(f"[{i+1}/{len(papers)}] {arxiv_id}")

        if os.path.exists(md_path):
            ok += 1
            continue  # already converted, resumable across reruns

        if not os.path.exists(pdf_path):
            print(f"  ! missing PDF, skipping")
            failed.append(arxiv_id)
            continue

        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            ok += 1
        except Exception as e:
            print(f"  ! failed to convert {arxiv_id}: {e}")
            failed.append(arxiv_id)

    print(f"\nConverted {ok}/{len(papers)} papers to markdown.")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    parse_all_pdfs()
