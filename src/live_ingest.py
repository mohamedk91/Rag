"""
Live fallback ingestion.

Used when the local FAISS index has nothing relevant enough for a question
(see config.SIMILARITY_THRESHOLD). Flow:

  1. Turn the question into a few keywords (not an exact-phrase search).
  2. Search arXiv (cs.CL) with those keywords; relax the query if it finds nothing.
  3. Score each candidate's title+abstract against the question and keep only
     the relevant ones, BEFORE downloading any PDF.
  4. Download + parse the survivors, chunk them, add them to the retriever
     (saved to disk).
"""
import json
import os
import re
import threading
import time

import arxiv
import pymupdf4llm

import config
from chunking import chunk_markdown_text
from fileio import atomic_write_json

_AGENT_SCOPE_RE = re.compile(config.AGENT_SCOPE_PATTERN, re.IGNORECASE)

# One client for the whole process, so arXiv's 3s spacing between requests
# is respected across calls.
_client = arxiv.Client(
    page_size=config.LIVE_FETCH_SEARCH_RESULTS, delay_seconds=3, num_retries=3
)

# Only one live fallback may run at a time; extra requests skip it instead of
# piling up on arXiv.
_fallback_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Step 1: question -> keywords
# ---------------------------------------------------------------------------

_STOPWORDS = set("""
a an the and or but if then than so of to in on at by for with from into over about as
is are was were be been being do does did done have has had can could should would may
might will shall what which who whom whose when where why how this that these those it
its they them their there here i we you he she me my our your not no yes also very more
most much many some any each other such only just like etc e.g i.e
""".split())

# Words that appear in nearly every paper of this corpus, or are generic filler.
# Extend this list whenever you see a weak keyword in the printed arXiv query.
_GENERIC = set("""
llm llms llm-based llm-powered llm-driven llm-augmented large language
paper papers approach approaches method methods technique techniques
common commonly different main key major typical typically various several
recent current new existing use used uses using work works working
handle handles handling explain describe compare comparison difference differences
research study studies result results system systems based
combine combines combining
evaluate evaluates evaluating evaluation evaluations
""".split())

# Words with hyphens/dots inside ("long-term", "gpt-4.1") count as one token.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*")


def extract_keywords(question: str, max_keywords: int = None):
    """
    Pick the most search-worthy words from a natural-language question.
    Ranking (higher = more specific):
      +2  contains a digit or hyphen          (gpt-4, swe-bench, long-term)
      +2  has a capital after the first char  (ReAct, WebArena, APIs)
      +1  long word (>= 7 letters)
    Ties keep their original order in the question.
    """
    max_keywords = max_keywords or config.LIVE_FETCH_MAX_KEYWORDS
    scored, seen = [], set()

    for pos, tok in enumerate(_TOKEN_RE.findall(question)):
        low = tok.lower()
        if len(low) < 3 or low in _STOPWORDS or low in _GENERIC or low in seen:
            continue
        seen.add(low)

        score = 0
        if any(ch.isdigit() for ch in tok) or "-" in tok:
            score += 2
        if any(ch.isupper() for ch in tok[1:]):
            score += 2
        if len(low) >= 7:
            score += 1
        scored.append((-score, pos, low))

    scored.sort()
    return [low for _, _, low in scored[:max_keywords]]


def build_arxiv_query(keywords):
    """cat:cs.CL AND all:"kw1" AND all:"kw2" ..."""
    terms = " AND ".join(f'all:"{k}"' for k in keywords)
    return f"cat:{config.ARXIV_CATEGORY} AND {terms}"


# ---------------------------------------------------------------------------
# Step 2: search arXiv (metadata only, no downloads)
# ---------------------------------------------------------------------------

def _base_id(short_id: str) -> str:
    """'2310.03123v2' -> '2310.03123' (removes ONLY a trailing version suffix)."""
    return re.sub(r"v\d+$", "", short_id)


def search_candidates(question: str, known_ids: set):
    """
    Search arXiv with the question's keywords. If the AND query returns nothing
    new, drop the lowest-ranked keyword and retry (down to 2 terms).
    Returns a list of arxiv.Result objects not already in the index.
    """
    keywords = extract_keywords(question)
    if not keywords:
        print("  Could not extract any keywords from the question.")
        return []

    n = len(keywords)
    min_terms = min(2, n)
    attempts = 0

    while n >= min_terms and attempts < config.LIVE_FETCH_MAX_ATTEMPTS:
        query = build_arxiv_query(keywords[:n])
        print(f"  arXiv query: {query}")

        search = arxiv.Search(
            query=query,
            max_results=config.LIVE_FETCH_SEARCH_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            results = list(_client.results(search))
        except Exception as e:
            print(f"  ! arXiv search failed: {e}")
            return []

        fresh = [r for r in results if _base_id(r.get_short_id()) not in known_ids]
        if fresh:
            return fresh[: config.LIVE_FETCH_CANDIDATES]

        n -= 1
        attempts += 1

    return []


# ---------------------------------------------------------------------------
# Step 3: relevance check on title + abstract (before any PDF download)
# ---------------------------------------------------------------------------

def rank_candidates(question: str, candidates: list, retriever):
    """
    Keep only candidates that (1) mention agents in their title/abstract and
    (2) have a title+abstract similar enough to the question.

    The scope check (1) stops off-topic NLP papers (e.g. a BERT question) from being
    downloaded into an index that is only meant to cover LLM agents.
    """
    if config.REQUIRE_AGENT_SCOPE:
        in_scope = [r for r in candidates if _AGENT_SCOPE_RE.search(f"{r.title} {r.summary}")]
        dropped = len(candidates) - len(in_scope)
        if dropped:
            print(f"  Dropped {dropped} candidate(s) that do not mention agents (out of scope).")
        candidates = in_scope
        if not candidates:
            return []

    docs = [f"{r.title.strip()}\n{r.summary.strip()}" for r in candidates]
    sims = retriever.similarity_to_query(question, docs)

    scored = sorted(zip(sims, candidates), key=lambda x: x[0], reverse=True)

    # Printed on purpose: these numbers help you calibrate the threshold.
    print("  Candidate similarity (question vs title+abstract):")
    for s, r in scored:
        mark = "keep" if s >= config.LIVE_FETCH_MIN_SIMILARITY else "skip"
        print(f"    [{mark}] {s:.3f}  {_base_id(r.get_short_id())}  {r.title.strip()[:70]}")

    kept = [r for s, r in scored if s >= config.LIVE_FETCH_MIN_SIMILARITY]
    return kept[: config.LIVE_FETCH_MAX_PAPERS]


# ---------------------------------------------------------------------------
# Step 4: download + parse the survivors
# ---------------------------------------------------------------------------

def download_and_parse(result):
    """
    Download the PDF and convert it to Markdown. Files are written under a temp
    name and renamed only when complete, so an interrupted run can never leave
    a half-written file that later runs would mistake for a finished one.
    Returns a metadata dict, or None on failure.
    """
    arxiv_id = _base_id(result.get_short_id())
    pdf_path = os.path.join(config.PDF_DIR, f"{arxiv_id}.pdf")
    md_path = os.path.join(config.MD_DIR, f"{arxiv_id}.md")

    try:
        if not os.path.exists(pdf_path):
            tmp_name = f"{arxiv_id}.pdf.part"
            result.download_pdf(dirpath=config.PDF_DIR, filename=tmp_name)
            os.replace(os.path.join(config.PDF_DIR, tmp_name), pdf_path)
            time.sleep(1)  # be polite to arXiv

        if not os.path.exists(md_path):
            md_text = pymupdf4llm.to_markdown(pdf_path)
            tmp_md = md_path + ".tmp"
            with open(tmp_md, "w", encoding="utf-8") as f:
                f.write(md_text)
            os.replace(tmp_md, md_path)

        return {
            "arxiv_id": arxiv_id,
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip(),
            "published": str(result.published.date()),
            "url": result.entry_id,
            "pdf_path": pdf_path,
        }
    except Exception as e:
        print(f"  ! live fetch failed for {arxiv_id}: {e}")
        return None


def ingest_new_papers(retriever, new_papers: list):
    """
    Chunk newly fetched papers and add them to the retriever (in memory + saved).
    Returns the number of chunks actually added.
    """
    all_new_chunks = []
    for paper in new_papers:
        md_path = os.path.join(config.MD_DIR, f"{paper['arxiv_id']}.md")
        if not os.path.exists(md_path):
            continue
        with open(md_path, encoding="utf-8") as f:
            md_text = f.read()
        all_new_chunks.extend(chunk_markdown_text(md_text, paper["arxiv_id"], paper["title"]))

    n_added = retriever.add_chunks(all_new_chunks, persist=True)

    # Keep the master metadata file in sync (written atomically).
    if os.path.exists(config.METADATA_JSON):
        with open(config.METADATA_JSON, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_ids = {p["arxiv_id"] for p in existing}
    existing.extend(p for p in new_papers if p["arxiv_id"] not in existing_ids)
    atomic_write_json(config.METADATA_JSON, existing)

    return n_added


# ---------------------------------------------------------------------------
# Entry point used by rag_pipeline.py
# ---------------------------------------------------------------------------

def fallback_fetch_and_ingest(query: str, retriever):
    """
    Full fallback flow. Returns how many new chunks were added
    (0 if nothing new and relevant was found, or a fallback is already running).
    """
    if not _fallback_lock.acquire(blocking=False):
        print("  Another live fallback is already running; skipping this one.")
        return 0
    try:
        print(f"  Local index not confident enough - searching arXiv live for: '{query}'")
        os.makedirs(config.PDF_DIR, exist_ok=True)
        os.makedirs(config.MD_DIR, exist_ok=True)

        candidates = search_candidates(query, retriever.known_arxiv_ids)
        if not candidates:
            print("  No new papers found on arXiv for this question.")
            return 0

        selected = rank_candidates(query, candidates, retriever)
        if not selected:
            print("  Found papers, but none were relevant enough to ingest.")
            return 0

        new_papers = [p for p in (download_and_parse(r) for r in selected) if p]
        if not new_papers:
            return 0

        print(f"  Ingesting {len(new_papers)} paper(s): {[p['arxiv_id'] for p in new_papers]}")
        n_chunks = ingest_new_papers(retriever, new_papers)
        print(f"  Ingested {n_chunks} new chunks into the index.")
        return n_chunks
    finally:
        _fallback_lock.release()