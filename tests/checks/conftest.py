"""A hand-built tokenizer for driving checks without training a model.

These tests verify the *check logic* — thresholds, skip conditions, evidence. That they can run
against a fake at all is the point of the protocol split: nothing in `checks/` knows what a
SentencePiece model is. Proof that the checks fire on real defects lives in tests/pitfalls/.
"""

from collections.abc import Callable, Sequence

import pytest

from sentencepiece_ocr_guide.checks.piece_text import surface

UNKNOWN_PIECE = "<unk>"
CONTROL_PIECES = ("<s>", "</s>")


class FakeTokenizer:
    """Greedy longest-match tokenizer over an explicit vocabulary.

    `normalizer` models a tokenizer whose normalization differs from ground truth — the defect
    behind failure modes #2 and #3 — by rewriting text on the way in but not on the way out.
    """

    def __init__(
        self,
        pieces: Sequence[str],
        normalizer: Callable[[str], str] | None = None,
        byte_pieces: Sequence[str] = (),
    ) -> None:
        self._pieces = (UNKNOWN_PIECE, *CONTROL_PIECES, *pieces)
        self._normalizer = normalizer or (lambda text: text)
        self._byte_pieces = frozenset(byte_pieces)
        self._by_length = sorted(pieces, key=len, reverse=True)

    def encode(self, text: str) -> list[int]:
        normalized = self._normalizer(text)
        token_ids: list[int] = []
        position = 0
        while position < len(normalized):
            piece = self._longest_match(normalized, position)
            if piece is None:
                token_ids.append(0)  # <unk>
                position += 1
                continue
            token_ids.append(self._pieces.index(piece))
            position += len(surface(piece))
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(
            surface(self._pieces[token_id])
            for token_id in token_ids
            if not self.is_unknown(token_id) and not self.is_control(token_id)
        )

    def __len__(self) -> int:
        return len(self._pieces)

    def piece(self, token_id: int) -> str:
        return self._pieces[token_id]

    def is_byte(self, token_id: int) -> bool:
        return self._pieces[token_id] in self._byte_pieces

    def is_unknown(self, token_id: int) -> bool:
        return token_id == 0

    def is_control(self, token_id: int) -> bool:
        return self._pieces[token_id] in CONTROL_PIECES

    def _longest_match(self, text: str, position: int) -> str | None:
        for piece in self._by_length:
            if text.startswith(surface(piece), position):
                return piece
        return None


def vocabulary_for(*texts: str) -> tuple[str, ...]:
    """Every distinct character in `texts` — a character-level vocabulary."""
    return tuple(sorted({char for text in texts for char in text}))


@pytest.fixture
def latin_tokenizer() -> FakeTokenizer:
    return FakeTokenizer(pieces=vocabulary_for("hello world abc123"))
