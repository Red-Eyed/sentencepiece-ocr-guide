"""Adapters from concrete tokenizer implementations to the `checks` protocols."""

from sentencepiece_ocr_guide.adapters.spm import SentencePieceTokenizer

__all__ = ["SentencePieceTokenizer"]
