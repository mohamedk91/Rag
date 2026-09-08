"""
Central configuration for the RAG pipeline.
Every script imports from here so paths/model names/params stay in one place.
"""
import os

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_DIR = os.path.join(DATA_DIR, "papers_pdf")
MD_DIR = os.path.join(DATA_DIR, "papers_md")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
CHUNKS_JSONL = os.path.join(CHUNKS_DIR, "chunks.jsonl")
METADATA_JSON = os.path.join(DATA_DIR, "papers_metadata.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_STORE_PATH = os.path.join(DATA_DIR, "chunks_store.jsonl")  # chunk text aligned to faiss ids

# ---- arXiv fetch ----
ARXIV_CATEGORY = "cs.CL"
ARXIV_SEARCH_QUERY = "LLM agents"          # topic narrowing within cs.CL
MAX_PAPERS = 150                            # tractable size for a focused demo

# ---- Chunking (all sizes are REAL TOKENS of the embedding model's tokenizer) ----
MAX_CHUNK_TOKENS = 400       # body text per chunk
CHUNK_OVERLAP_TOKENS = 50    # overlap between sub-chunks of a long section

# ---- Embedding ----
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DEVICE = "cuda:1"   # second T4 in the 2xT4 split
# Hard cap the embedder applies. Must be > MAX_CHUNK_TOKENS + title/section
# prefix (~60 tokens) so nothing is ever truncated.
EMBED_MAX_SEQ_LENGTH = 512

# Instruction prepended to QUERIES only (never to documents).
QUERY_INSTRUCTION = (
    "Instruct: Given a question about LLM agent research, "
    "retrieve passages from arXiv papers that answer it\nQuery: "
)

# ---- Generation ----
LLM_MODEL_NAME = "Qwen/Qwen3-8B"
LLM_DEVICE = "cuda:0"         # first T4 in the 2xT4 split
LLM_ENABLE_THINKING = False   # off for fast, direct RAG answers
LLM_MAX_NEW_TOKENS = 768      # 512 can truncate answers that end with citations

# ---- Retrieval (calibrate with calibrate.py; RE-RUN it after any retrieval change) ----
TOP_K = 5
SIMILARITY_THRESHOLD = 0.65  # top-1 below this -> try live fallback
MIN_RERANK_SCORE = 0.80       # cross-encoder logit below this -> chunk dropped (calibrated)
MIN_CHUNK_SCORE = 0.30        # LEGACY: superseded by MIN_RERANK_SCORE; kept for calibrate.py
NO_INFO_MESSAGE = "I don't have enough information in the retrieved papers to answer this."
NO_INFO_MARKER = "i don't have enough information"   # lowercase; detects model self-refusal
EMPTY_QUESTION_MESSAGE = "Please type a question."
MAX_QUESTION_CHARS = 500

# ---- Cross-encoder reranker (re-scores first-stage candidates) ----
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cuda:1"    # shares GPU with the embedding model
RERANK_TOP_N = 20             # first-stage candidates before reranking to TOP_K      

# ---- BM25 hybrid search (fused with dense FAISS via Reciprocal Rank Fusion) ----
RRF_K = 60                    # RRF constant (standard default)

# ---- Live fallback fetch (papers not yet in the local index) ----
LIVE_FETCH_MAX_PAPERS = 3          # max papers downloaded + ingested per fallback
LIVE_FETCH_SEARCH_RESULTS = 20     # results requested from arXiv per search (one API page)
LIVE_FETCH_CANDIDATES = 10         # new (not-yet-indexed) papers examined per search
LIVE_FETCH_MAX_KEYWORDS = 4        # keywords used to build the arXiv query
LIVE_FETCH_MAX_ATTEMPTS = 3        # query relaxations before giving up
# Min cosine similarity between question and a candidate's title+abstract
# for it to be downloaded. PLACEHOLDER: calibrate.py suggests a value.
LIVE_FETCH_MIN_SIMILARITY = 0.40

# ---- Scope guard (live fallback only downloads papers about agents) ----
REQUIRE_AGENT_SCOPE = True
AGENT_SCOPE_PATTERN = r"\bagents?\b"

# ---- Deployment / safety ----
def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes")

# RAG_SHARE=1 publishes a public Gradio link (needed on Kaggle to reach the demo).
SHARE_PUBLIC = _env_flag("RAG_SHARE", False)

# The fallback writes to your index: ON for private use, OFF for a shared app.
# Override with RAG_ALLOW_FALLBACK=1 or 0.
ALLOW_LIVE_FALLBACK = _env_flag("RAG_ALLOW_FALLBACK", not SHARE_PUBLIC)

LIVE_FALLBACK_COOLDOWN_SECONDS = 120   # minimum gap between fallback attempts
MAX_INDEXED_PAPERS = 400               # fallback stops adding papers beyond this
# ---- Junk-chunk filter (see chunking.is_junk_chunk) ----
MIN_CHUNK_TOKENS = 40            # chunks shorter than this are dropped
MAX_TABLE_LINE_FRACTION = 0.33   # drop if more than this share of lines are markdown table rows
MIN_ALPHA_RATIO = 0.5            # drop if letters are under this share of non-space characters
# ---- Oversized-table handling ----
MAX_TABLE_ROW_CHARS = 3000       # a table row longer than this is a text dump, not data
MAX_BR_TAGS_PER_CHUNK = 5        # a chunk with this many "<br>" tags is a flattened table