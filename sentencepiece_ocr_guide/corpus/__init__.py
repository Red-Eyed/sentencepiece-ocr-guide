"""Corpus-side tooling: measure how sources disagree, then canonicalize them.

This runs *before* training. The model checks in `checks/` run after, and several of them
(`nfc_vocabulary` above all) only ever report defects that originate here.
"""

from sentencepiece_ocr_guide.corpus.axes import DEFAULT_AXES, Action, Axis, axes_with
from sentencepiece_ocr_guide.corpus.canonicalize import canonicalizer, is_canonical
from sentencepiece_ocr_guide.corpus.scan import scan_corpus

__all__ = [
    "DEFAULT_AXES",
    "Action",
    "Axis",
    "axes_with",
    "canonicalizer",
    "is_canonical",
    "scan_corpus",
]
