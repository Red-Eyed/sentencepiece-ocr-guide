# spm-ocr

Train a SentencePiece tokenizer for messy multilingual OCR corpora.

This project is for OCR datasets that arrive as real-world dumps: directories, single files,
archives inside archives, compressed text, mixed scripts, weak path hints, extractor artifacts,
and uneven source sizes. The goal is to produce a tokenizer training corpus that is repaired,
logged, lightly balanced, and easy to inspect before training.

## Why Use It

- Pass one corpus path instead of pre-sorting files by extension or language.
- Detect text, archives, and compressed files by content rather than filename.
- Fix OCR-relevant Unicode issues while preserving characters that matter for labels.
- Split oversized OCR lines into safe chunks, skip only unrepairable lines, and log every action.
- Repair line chunks in parallel while writing deterministic corpus outputs.
- Build several meaningful training corpus files instead of one anonymous bag.
- Train SentencePiece with OCR-safe defaults from `cfg.json.ocr`.
- Keep progress visible through long discovery, repair, balancing, and training stages.

## Pipeline

1. Read settings from `cfg.json` and apply the selected OCR preset.
2. Find text from one file or a recursive directory scan, unpacking archives and compressed inputs.
3. Fix the corpus, preserving the original files and logging every repair, chunk, or skipped line.
4. Build several lightly balanced training corpus files under `work_dir/train_corpus/`.
5. Train SentencePiece and write the model, vocabulary, and reports under `work_dir/`.

## Usage

Minimal `cfg.json`:

```json
{
  "preset": "ocr_multilingual",
  "num_threads": 16,
  "corpus": {
    "path": "data/raw-corpus"
  },
  "output": {
    "work_dir": "runs/ocr-spm-v1",
    "model_prefix": "ocr_tokenizer"
  }
}
```

`num_threads` applies to both corpus repair and SentencePiece training.

To use a specialized preset, change only the preset name:

```json
{
  "preset": "ocr_cjk_heavy",
  "num_threads": 16,
  "corpus": {
    "path": "data/raw-corpus"
  },
  "output": {
    "work_dir": "runs/ocr-spm-v1",
    "model_prefix": "ocr_tokenizer"
  }
}
```

Run:

```sh
just train
```

Or directly:

```sh
RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" CFLAGS="${CFLAGS:-} -march=native" cargo run --release -- train --config cfg.json
```

Press Ctrl+C to stop a run. During SentencePiece training, `spm-ocr` also stops the Python
trainer subprocess before exiting.

## Outputs

- `work_dir/train_corpus/*.txt`: meaningful corpus parts used for training.
- `work_dir/fixed_corpus.txt`: repaired audit stream.
- `work_dir/ocr_tokenizer.model`: trained SentencePiece model.
- `work_dir/ocr_tokenizer.vocab`: trained vocabulary.
- `work_dir/reports/corpus_issues.jsonl`: fixes, skips, and source issues.
- `work_dir/reports/balance.json`: balancing summary.
- `work_dir/trainer_request.json`: replayable trainer request.
- `work_dir/trainer_output.json`: trainer status and artifact paths.

## Development

```sh
just check
just fmt
```

Design details live in [docs/10-rust-corpus-fixer-trainer-design.md](docs/10-rust-corpus-fixer-trainer-design.md).
