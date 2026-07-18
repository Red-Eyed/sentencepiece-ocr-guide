"""Reading the surface text of a vocabulary piece.

SentencePiece encodes a word-initial space as U+2581 (`▁`) rather than a literal space. Checks
that inspect what a piece *says* need that marker removed; the one check that cares whether the
marker is there reads it directly.
"""

SPACE_MARKER = "▁"


def surface(piece: str) -> str:
    """The piece's text with the space marker translated back to a real space."""
    return piece.replace(SPACE_MARKER, " ")


def has_space_marker(piece: str) -> bool:
    return piece.startswith(SPACE_MARKER)
