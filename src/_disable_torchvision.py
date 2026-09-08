"""
Torchvision is not needed anywhere in this project (text-only pipeline:
no images), but transformers/sentence_transformers try to import it
opportunistically for optional image-related features. On some Kaggle
environments torchvision's compiled C++ extension fails to load cleanly
(circular import / mismatched build), which then crashes the *entire*
transformers import even though nothing image-related is being used.

Setting sys.modules["torchvision"] = None is a Python-documented way to
tell the import system "this module is blocked" - any `import torchvision`
elsewhere immediately raises ImportError, which transformers' own
is_torchvision_available() checks already handle gracefully (it just
returns False and skips the optional image code path).

Must be imported BEFORE transformers or sentence_transformers anywhere.
"""
import sys

sys.modules["torchvision"] = None
