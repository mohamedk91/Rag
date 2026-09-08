import sys
import os
from unittest.mock import MagicMock

for mod in ["transformers", "torch"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import (
    is_stop_heading,
    is_skip_heading,
    is_table_row,
    is_junk_chunk,
    drop_oversized_table_rows,
    _split_into_sentences,
)


def test_stop_headings():
    assert is_stop_heading("References")
    assert is_stop_heading("6 References")
    assert is_stop_heading("Bibliography")
    assert is_stop_heading("Works Cited")
    assert is_stop_heading("Appendix A")
    assert is_stop_heading("Supplementary Material")
    # Should NOT trigger on reference in normal prose
    assert not is_stop_heading("References to Prior Work")


def test_skip_headings():
    assert is_skip_heading("Acknowledgments")
    assert is_skip_heading("Author Contributions")
    assert not is_skip_heading("Methodology")


def test_table_row_detection():
    assert is_table_row("| Col1 | Col2 | Col3 |")
    assert is_table_row("|---|---|---|")
    # Prose with pipes should NOT match
    assert not is_table_row("The absolute value |x| < 1 holds here.")
    assert not is_table_row("choice A | choice B")


def test_drop_oversized_table_rows():
    long_row = "|" + "x" * 4000 + "|" + "y" * 4000 + "|"
    normal_text = "Here is some research text.\n" + long_row + "\nNext paragraph."
    cleaned = drop_oversized_table_rows(normal_text)
    assert long_row not in cleaned
    assert "Here is some research text." in cleaned
    assert "Next paragraph." in cleaned


def test_junk_chunk_filters():
    assert is_junk_chunk("tiny", n_tokens=10) == "too_short"
    # Symbol soup
    symbol_soup = "!@#$%^&*()_+{}[]:;<>?,./" * 5
    assert is_junk_chunk(symbol_soup, n_tokens=50) == "symbols"


def test_sentence_split():
    text = "LLM agents are versatile. They can plan and act. Can they reflect? Yes!"
    sentences = _split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "LLM agents are versatile."
    assert sentences[1] == "They can plan and act."
    assert sentences[2] == "Can they reflect?"
    assert sentences[3] == "Yes!"
