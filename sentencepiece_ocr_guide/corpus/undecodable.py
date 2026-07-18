"""Detecting bytes that were never valid UTF-8.

Reading with `errors="surrogateescape"` maps each undecodable byte to a lone surrogate in
U+DC80–U+DCFF, so their presence in a decoded line is exact evidence that the source was not
valid UTF-8. Shared by the scanner (which counts them) and the rewriter (which refuses to
canonicalize them).
"""

SURROGATE_ESCAPES = frozenset(map(chr, range(0xDC80, 0xDD00)))


def has_undecodable_bytes(line: str) -> bool:
    return not SURROGATE_ESCAPES.isdisjoint(line)
