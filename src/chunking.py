"""
Shared chunking logic (heading-based split + sub-splitting long sections).
Used by both 3_chunk.py (batch) and live_ingest.py (query-time ingestion).

- Sizes are measured in REAL TOKENS of the embedding model's tokenizer.
- Reference lists / appendices are removed (heading-based + content-based).
- Author/affiliation lines are not treated as section headings.
- Headings are found at ANY '#' depth when they are real top-level sections
  (numbered like '3 Method' or named like 'Introduction'), because PDF parsers
  often emit them as '####' or '#####' for small-font papers.
"""
import re

import config

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
            assert _tokenizer.is_fast, "Need a fast tokenizer for offset mapping"
        except Exception:
            _tokenizer = None
    return _tokenizer


# ---------------------------------------------------------------------------
# Heading classification
# ---------------------------------------------------------------------------

# Leading section numbers: "6 ", "6.1 ", "VII. ", "IV) ".
# Roman numerals are uppercase-only so ordinary words aren't touched.
_NUMBERING_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXL]+)[\.\)]?\s+")

# STOP: this section AND everything after it is dropped.
_NOT_A_REF_LIST = r"(?!\s+(?:to|and|for|in|on|of|between|with)\b)"   # "References to Prior Work" stays

STOP_PATTERNS = [
    r"^reference$",
    r"^references\b" + _NOT_A_REF_LIST,   # also catches "References Shridhar, M., Yuan, X., ..."
    r"^bibliography\b" + _NOT_A_REF_LIST,
    r"^works cited$",
    r"^literature cited$",
    r"^([a-z][\.\)]?\s+)?appendix\b",      # "appendix", "appendix a", "a. appendix"
    r"^appendices\b",
    r"^supplementary (material|information)s?\b",
]

# SKIP: only this section is dropped; later sections are still kept.
SKIP_PATTERNS = [
    r"^acknowledg(e)?ments?$",
    r"^author contributions?$",
]


def normalize_heading(text: str) -> str:
    text = re.sub(r"[*_`#]", "", text).strip()   # markdown decoration
    text = _NUMBERING_RE.sub("", text)           # "6 References" -> "References"
    return text.strip().lower().rstrip(":.")


def _matches_any(text: str, patterns) -> bool:
    h = normalize_heading(text)
    return any(re.match(p, h) for p in patterns)


def is_stop_heading(text: str) -> bool:
    return _matches_any(text, STOP_PATTERNS)


def is_skip_heading(text: str) -> bool:
    return _matches_any(text, SKIP_PATTERNS)


def is_boundary_heading(text: str) -> bool:
    return is_stop_heading(text) or is_skip_heading(text)


# ---------------------------------------------------------------------------
# Content-based safety net
# ---------------------------------------------------------------------------

_CITATION_MARKERS = re.compile(
    r"arXiv preprint|Proceedings of|Association for Computational Linguistics"
    r"|\bCoRR\b|abs/\d{4}\.\d{4,5}|arXiv:\d{4}\.\d{4,5}|\bpp\.\s*\d+"
    r"|International Conference on|Advances in Neural Information|Transactions of the",
    re.IGNORECASE,
)

# Bibliography-style author lists: "Shridhar, M.,"  "Cote, M.-A.,"  "Bisk, Y."
_AUTHOR_INITIALS = re.compile(r"\b[A-Z][^\W\d_]+,\s+[A-Z]\.(?:[-\s]?[A-Z]\.)*")


def looks_like_references(text: str, min_markers: int = 4) -> bool:
    """True if the text is dense with bibliography-style markers or author lists."""
    if len(_CITATION_MARKERS.findall(text)) >= min_markers:
        return True
    return len(_AUTHOR_INITIALS.findall(text)) >= min_markers


# Author/affiliation lines that PDF parsers mistake for headings:
#   "Gavin K. Reynolds[*], Anna Jawor-Baczynska[+]"   "Xiaoyong Wei[10*]"   "a@b.edu"
_AUTHOR_LINE = re.compile(r"[A-Za-z.]\[[^\]\s]{1,8}\]|@")


def looks_like_author_line(text: str) -> bool:
    return bool(_AUTHOR_LINE.search(text))


# Plain author lists with no brackets: "Yuchen Xia, Michael Weyrich, Nasser Jazdi"
_NAME_PART = re.compile(r"^[A-Z][^\W\d_]*\.?(?:[\s-]+[A-Z][^\W\d_]*\.?){1,3}$")   # 2-4 capitalised words


def looks_like_name_list(text: str) -> bool:
    parts = [p.strip() for p in re.split(r",|\band\b|&", re.sub(r"\[[^\]]*\]", "", text)) if p.strip()]
    return len(parts) >= 3 and all(_NAME_PART.match(p) for p in parts)


def _norm_words(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def is_title_heading(heading: str, title: str) -> bool:
    """True if the heading is just the paper's own title (possibly cut short)."""
    h, t = _norm_words(heading), _norm_words(title)
    return len(h) >= 15 and t.startswith(h[:25])


# ---------------------------------------------------------------------------
# Junk-chunk filter (table dumps, symbol soup, tiny fragments)
# ---------------------------------------------------------------------------

_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


def is_table_row(line: str) -> bool:
    """
    True only for a real markdown table row: the line starts AND ends with "|"
    and has at least 3 pipes (i.e. two or more cells), or it is the "|---|---|"
    separator row. Prose that merely contains pipes, such as "|x| <= 1" (absolute
    value) or "a | b" (alternatives), does not match.
    """
    line = line.strip()
    if _TABLE_SEPARATOR_RE.match(line) and "|" in line:
        return True
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 3


def drop_oversized_table_rows(body: str) -> str:
    """
    Remove markdown table rows longer than config.MAX_TABLE_ROW_CHARS from a section body.

    Some papers contain a table whose cells hold whole LLM outputs joined with "<br>"
    (one row can be 250,000 characters). That is a log dump, not a table of results.
    It must be removed BEFORE the body is cut into token windows: after cutting, each
    window is a fragment that no longer starts and ends with "|", so is_junk_chunk
    could not recognise it as table content. Normal paragraphs are never table rows,
    so they are not affected.
    """
    kept = [ln for ln in body.split("\n")
            if not (len(ln) > config.MAX_TABLE_ROW_CHARS and is_table_row(ln))]
    return "\n".join(kept)


def is_junk_chunk(text: str, n_tokens: int):
    """
    Return a short reason string if this chunk is not worth indexing, else None.
    Three independent tests, each tunable from config.py:
      - too short: a fragment carries too little meaning to match a question
      - table dump: most non-empty lines are markdown table rows ("| a | b |")
      - symbol soup: under half of the non-space characters are letters
        (equations, numeric tables, parser debris)
    """
    if n_tokens < config.MIN_CHUNK_TOKENS:
        return "too_short"

    # "<br>" only appears when the PDF parser flattened table cells, so many of them
    # means this chunk is a fragment of a table even if it no longer looks like rows.
    if text.count("<br>") >= config.MAX_BR_TAGS_PER_CHUNK:
        return "table"

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        table_lines = sum(1 for ln in lines if is_table_row(ln))
        if table_lines / len(lines) > config.MAX_TABLE_LINE_FRACTION:
            return "table"

    non_space = [c for c in text if not c.isspace()]
    if non_space and sum(c.isalpha() for c in non_space) / len(non_space) < config.MIN_ALPHA_RATIO:
        return "symbols"

    return None


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

_HASH_RE = re.compile(r"^(#{1,6})\s+(.*)")
_BOLD_RE = re.compile(r"^\*\*(.{2,60}?)\*\*$")


# A real TOP-LEVEL section heading, whatever '#' depth the parser gave it:
#   "1 Introduction"  "2. Related Work"  "VII. Results"  or a well-known section name.
# Subsections like "3.2 Tool Use" deliberately do NOT match (they stay inside their section).
_TOP_SECTION_RE = re.compile(
    r"^(?:\d{1,2}[.)]?|[IVX]{1,5}[.)])\s+\S"
    r"|^(?:abstract|introduction|related work|background|preliminaries|methods?|methodology|"
    r"approach|experiments?|results|evaluation|discussion|conclusions?|limitations)\b",
    re.IGNORECASE,
)


def _clean_heading(text: str) -> str:
    """Strip markdown decoration but KEEP the section number."""
    return re.sub(r"[*_`#]", "", text).strip()


def is_top_section_heading(text: str) -> bool:
    return bool(_TOP_SECTION_RE.match(_clean_heading(text)))


# A short line that is ONLY "References"/"Bibliography" (no '#' or '**') still ends the paper.
_PLAIN_STOP_RE = re.compile(r"^(references?|bibliography|works cited)$")


def _parse_heading(line: str):
    """
    Return the heading text if this line starts a new section, else None.
    - '#', '##', '###' headings split (unless the text is really an author line).
    - '####' and deeper split ONLY if they are real top-level sections
      ("1 Introduction", "Abstract", ...) or stop/skip headings.
    - Bold-only lines follow the same rule as deep headings.
    - A short plain line that is just "References" also splits.
    """
    line = line.strip()

    m = _HASH_RE.match(line)
    if m:
        level, text = len(m.group(1)), m.group(2).strip()
        if is_boundary_heading(text):
            return text
        if looks_like_author_line(text):
            return None
        if level <= 3 or is_top_section_heading(text):
            return text
        return None

    m = _BOLD_RE.match(line)
    if m:
        text = m.group(1).strip()
        if is_boundary_heading(text) or (is_top_section_heading(text) and not looks_like_author_line(text)):
            return text
        return None

    if 0 < len(line) <= 30 and _PLAIN_STOP_RE.match(normalize_heading(line)):
        return line

    return None


def split_by_headings(md_text: str):
    lines = md_text.split("\n")
    sections = []
    current_heading = "Preamble"
    current_body = []

    for line in lines:
        heading = _parse_heading(line)
        if heading is not None:
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = heading
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return sections


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return max(1, len(text.split()))


def _split_into_sentences(text: str) -> list:
    """Split text along paragraph and sentence boundaries."""
    # Split by double newline first to preserve paragraph breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    # Sentence boundary: period/question/exclamation followed by space or newline
    sentence_end_re = re.compile(r'(?<=[.!?])\s+')
    for p in paragraphs:
        p_sents = sentence_end_re.split(p)
        for s in p_sents:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)
    return sentences if sentences else [text]


def token_windows(text: str, max_tokens: int, overlap_tokens: int):
    """
    Split text into windows of at most max_tokens, respecting sentence
    boundaries so chunks do not cut mid-sentence.
    """
    tok = get_tokenizer()
    if tok is not None:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["offset_mapping"]) <= max_tokens:
            return [text]
    else:
        if count_tokens(text) <= max_tokens:
            return [text]

    sentences = _split_into_sentences(text)
    sent_tokens = [count_tokens(s) for s in sentences]

    windows = []
    cur_chunk = []
    cur_tokens = 0

    for s, t_count in zip(sentences, sent_tokens):
        # Edge case: single sentence longer than max_tokens -> fall back to slicing.
        if t_count > max_tokens:
            if cur_chunk:
                windows.append(" ".join(cur_chunk))
                cur_chunk = []
                cur_tokens = 0

            if tok is not None:
                enc_s = tok(s, add_special_tokens=False, return_offsets_mapping=True)
                offsets = enc_s["offset_mapping"]
                start = 0
                step = max(1, max_tokens - overlap_tokens)

                while start < len(offsets):
                    end = min(start + max_tokens, len(offsets))
                    windows.append(s[offsets[start][0]:offsets[end - 1][1]])
                    if end >= len(offsets):
                        break
                    start += step
            else:
                words = s.split()
                start = 0
                step = max(1, max_tokens - overlap_tokens)

                while start < len(words):
                    end = min(start + max_tokens, len(words))
                    windows.append(" ".join(words[start:end]))
                    if end >= len(words):
                        break
                    start += step

            continue
        if cur_tokens + t_count <= max_tokens:
            cur_chunk.append(s)
            cur_tokens += t_count
        else:
            if cur_chunk:
                windows.append(" ".join(cur_chunk))
            # Start new chunk with overlap from the tail of the current chunk
            overlap_chunk = []
            overlap_count = 0
            for prev_s in reversed(cur_chunk):
                p_cnt = count_tokens(prev_s)
                if overlap_count + p_cnt <= overlap_tokens:
                    overlap_chunk.insert(0, prev_s)
                    overlap_count += p_cnt
                else:
                    break
            cur_chunk = overlap_chunk + [s]
            cur_tokens = overlap_count + t_count

    if cur_chunk:
        windows.append(" ".join(cur_chunk))

    return windows if windows else [text]


def chunk_markdown_text(md_text: str, arxiv_id: str, title: str, stats=None):
    """
    Turn one paper's markdown into chunk records:
    {arxiv_id, title, section, chunk_text}

    `stats` is an optional dict-like counter (e.g. collections.Counter). When given,
    it is filled with how many chunks were kept or dropped and why, so 3_chunk.py
    can print a report. live_ingest.py just omits it.
    """
    def bump(key):
        if stats is not None:
            stats[key] += 1

    records = []

    for idx, (heading, body) in enumerate(split_by_headings(md_text)):
        if is_stop_heading(heading):
            break                      # references/appendix: drop it and ALL that follows
        if is_skip_heading(heading):
            continue                   # e.g. acknowledgments: drop just this section
        cleaned = drop_oversized_table_rows(body)
        if cleaned != body:
            bump("sections_with_oversized_table_rows")
            body = cleaned
        if len(body.split()) < 10:
            continue

        # The paper's own title is never a real section name, wherever it appears.
        # A plain list of names is only treated as authors near the top of the paper,
        # because later on it could be a genuine heading like "Prompting, Tuning, and Merging".
        label = heading
        if is_title_heading(heading, title) or (idx < 3 and looks_like_name_list(heading)):
            label = "Preamble"

        for sub_chunk in token_windows(body, config.MAX_CHUNK_TOKENS, config.CHUNK_OVERLAP_TOKENS):
            if looks_like_references(sub_chunk):
                bump("dropped_reference_like")   # safety net: heading missed, content gives it away
                continue
            reason = is_junk_chunk(sub_chunk, count_tokens(sub_chunk))
            if reason:
                bump(f"dropped_{reason}")
                continue
            bump("kept")
            if label == "Preamble":
                bump("kept_preamble")
            records.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "section": label,
                "chunk_text": sub_chunk,
            })

    return records