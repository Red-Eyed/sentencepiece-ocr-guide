"""Corpus-side tooling: measure how sources disagree, then canonicalize them.

This runs *before* training. The model checks in `checks/` run after, and several of them
(`nfc_vocabulary` above all) only ever report defects that originate here.
"""

from sentencepiece_ocr_guide.corpus.axes import DEFAULT_AXES, Action, Axis, axes_with
from sentencepiece_ocr_guide.corpus.canonicalize import canonicalizer, is_canonical
from sentencepiece_ocr_guide.corpus.rewrite import (
    RewriteRun,
    RewriteTally,
    UndecodableLineError,
    rewrite_lines,
)
from sentencepiece_ocr_guide.corpus.scan import scan_corpus
from sentencepiece_ocr_guide.corpus.undecodable import has_undecodable_bytes

__all__ = [
    "DEFAULT_AXES",
    "Action",
    "Axis",
    "RewriteRun",
    "RewriteTally",
    "UndecodableLineError",
    "axes_with",
    "canonicalizer",
    "has_undecodable_bytes",
    "is_canonical",
    "rewrite_lines",
    "scan_corpus",
]
