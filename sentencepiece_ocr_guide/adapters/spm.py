"""SentencePiece adapter — the imperative shell.

This is the only module in the package that imports `sentencepiece`. Everything in `checks/`
depends on the protocols instead, which is what lets the whole suite run against a fake.
"""

from collections.abc import Sequence
from pathlib import Path

import sentencepiece as spm


class SentencePieceTokenizer:
    """Adapts `SentencePieceProcessor` to the `Tokenizer` protocol."""

    def __init__(self, processor: spm.SentencePieceProcessor) -> None:
        self._processor = processor

    @classmethod
    def from_file(cls, model_path: Path) -> "SentencePieceTokenizer":
        return cls(spm.SentencePieceProcessor(model_file=str(model_path)))

    @classmethod
    def from_proto(cls, model_proto: bytes) -> "SentencePieceTokenizer":
        """Load from an in-memory model — used by tests that train without touching disk."""
        return cls(spm.SentencePieceProcessor(model_proto=model_proto))

    def encode(self, text: str) -> list[int]:
        return self._processor.encode(text, out_type=int)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._processor.decode(list(token_ids))

    def __len__(self) -> int:
        return self._processor.get_piece_size()

    def piece(self, token_id: int) -> str:
        return self._processor.id_to_piece(token_id)

    def is_byte(self, token_id: int) -> bool:
        return self._processor.is_byte(token_id)

    def is_unknown(self, token_id: int) -> bool:
        return self._processor.is_unknown(token_id)

    def is_control(self, token_id: int) -> bool:
        return self._processor.is_control(token_id)
