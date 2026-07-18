"""The narrow interfaces checks depend on.

Checks never import `sentencepiece`. They depend on these protocols, so any tokenizer that
can encode, decode and describe its vocabulary can be validated — and tests can drive every
check with a hand-built fake instead of training a model.

The split into two protocols is deliberate: a check that only inspects the vocabulary
(`digit_pieces`, `cross_script`) has no business receiving an encoder, and vice versa.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Encoder(Protocol):
    """Text to token ids and back."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@runtime_checkable
class Vocabulary(Protocol):
    """The set of pieces and their properties."""

    def __len__(self) -> int: ...

    def piece(self, token_id: int) -> str:
        """The surface string of a token, e.g. `'\\frac'` or `'<0x41>'`."""
        ...

    def is_byte(self, token_id: int) -> bool:
        """True for byte-fallback pieces (`<0x00>`–`<0xFF>`)."""
        ...

    def is_unknown(self, token_id: int) -> bool:
        """True for the `<unk>` piece."""
        ...

    def is_control(self, token_id: int) -> bool:
        """True for control pieces such as `<s>` and `</s>`."""
        ...


class Tokenizer(Encoder, Vocabulary, Protocol):
    """Both halves — what a fully-featured tokenizer provides and the runner passes around."""
