from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: python -m spm_ocr_train_bridge <trainer_request.json>",
            file=sys.stderr,
        )
        return 2

    request_path = Path(sys.argv[1])
    request = _read_request(request_path)
    sentencepiece_args = _sentencepiece_args(request)
    output_path = Path(request["output"]["trainer_output"])
    started = time.monotonic()

    try:
        spm.SentencePieceTrainer.train(**sentencepiece_args)
    except Exception as error:
        _write_output(
            output_path,
            {
                "status": "failed",
                "model": request["output"]["model"],
                "vocab": request["output"]["vocab"],
                "elapsed_ms": _elapsed_ms(started),
                "error": str(error),
            },
        )
        raise

    _write_output(
        output_path,
        {
            "status": "succeeded",
            "model": request["output"]["model"],
            "vocab": request["output"]["vocab"],
            "elapsed_ms": _elapsed_ms(started),
        },
    )
    return 0


def _read_request(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sentencepiece_args(request: dict[str, Any]) -> dict[str, Any]:
    args = dict(request["sentencepiece"])
    inputs = args.get("input")
    if isinstance(inputs, list):
        args["input"] = ",".join(str(path) for path in inputs)
    return args


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


if __name__ == "__main__":
    raise SystemExit(main())
