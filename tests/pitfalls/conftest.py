"""Training real tokenizers, deliberately misconfigured.

Every test here follows one shape: train the guide's configuration and a configuration with a
single setting changed to the pitfall value, then assert the relevant check passes on the first
and fails on the second. A check that only ever sees good input proves nothing.

Models are trained into memory, never to disk.
"""

import io
from collections.abc import Sequence
from typing import Any

import sentencepiece as spm

from sentencepiece_ocr_guide.adapters.spm import SentencePieceTokenizer

# docs/03-configuration.md, scaled down to train in under a second. Only vocab_size and the
# corpus size differ from the guide's recommendation.
GUIDE_CONFIG: dict[str, Any] = {
    "model_type": "bpe",
    "vocab_size": 320,
    "character_coverage": 1.0,
    "byte_fallback": True,
    "normalization_rule_name": "identity",
    "add_dummy_prefix": False,
    "remove_extra_whitespaces": False,
    "split_by_unicode_script": True,
    "split_by_whitespace": True,
    "max_sentencepiece_length": 8,
    "hard_vocab_limit": False,
    "minloglevel": 2,
}


def train(corpus: Sequence[str], **overrides: Any) -> SentencePieceTokenizer:
    """Train a tokenizer on `corpus` with the guide's config, plus any overrides."""
    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus),
        model_writer=model_writer,
        **(GUIDE_CONFIG | overrides),
    )
    return SentencePieceTokenizer.from_proto(model_writer.getvalue())


def repeated(lines: Sequence[str], times: int = 200) -> list[str]:
    """A corpus large enough for BPE to learn merges from."""
    return [line for _ in range(times) for line in lines]
