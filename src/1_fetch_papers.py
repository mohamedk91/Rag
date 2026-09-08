"""
Step 1: Fetch paper metadata + PDFs from arXiv.

Scope (locked in during planning):
  - category: cs.CL
  - topic filter: "LLM agents"
  - cap: config.MAX_PAPERS papers

Uses the `arxiv` python package, which wraps the official arXiv API.
Output:
  - data/papers_pdf/<arxiv_id>.pdf   (one PDF per paper)
  - data/papers_metadata.json        (title, authors, arxiv_id, abstract, url per paper)
"""
import json
import os
import time

import arxiv

import config


def fetch_metadata_and_pdfs():
    os.makedirs(config.PDF_DIR, exist_ok=True)

    # Combine the category filter with the topic search, e.g.:
    #   cat:cs.CL AND all:"LLM agents"
    query = f'cat:{config.ARXIV_CATEGORY} AND all:"{config.ARXIV_SEARCH_QUERY}"'

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=config.MAX_PAPERS,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers_metadata = []
    for i, result in enumerate(client.results(search)):
        arxiv_id = result.get_short_id()  # e.g. '2310.03123v2'
        base_id = arxiv_id.split("v")[0]  # strip version for a stable filename

        pdf_path = os.path.join(config.PDF_DIR, f"{base_id}.pdf")

        print(f"[{i+1}/{config.MAX_PAPERS}] {base_id} - {result.title.strip()[:80]}")

        # Download PDF if we don't already have it (resumable across reruns)
        if not os.path.exists(pdf_path):
            try:
                result.download_pdf(dirpath=config.PDF_DIR, filename=f"{base_id}.pdf")
                time.sleep(1)  # be polite to arXiv's servers
            except Exception as e:
                print(f"  ! failed to download {base_id}: {e}")
                continue

        papers_metadata.append({
            "arxiv_id": base_id,
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip(),
            "published": str(result.published.date()),
            "url": result.entry_id,
            "pdf_path": pdf_path,
        })

    with open(config.METADATA_JSON, "w") as f:
        json.dump(papers_metadata, f, indent=2)

    print(f"\nDone. Fetched {len(papers_metadata)} papers.")
    print(f"Metadata saved to: {config.METADATA_JSON}")
    print(f"PDFs saved to:     {config.PDF_DIR}")


if __name__ == "__main__":
    fetch_metadata_and_pdfs()
