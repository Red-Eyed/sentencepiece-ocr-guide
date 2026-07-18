"""Smoke test — keeps the `just check` gate green until real tests land."""

import sentencepiece_ocr_guide


def test_package_imports() -> None:
    assert sentencepiece_ocr_guide is not None
