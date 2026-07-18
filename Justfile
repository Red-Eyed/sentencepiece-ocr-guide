set dotenv-load := false

# List available recipes, grouped
default:
    @just --list --unsorted

# ---------------------------------------------------------------------------
# Workflow — validating a corpus and the tokenizer trained from it
# ---------------------------------------------------------------------------

# Print the end-to-end workflow, with runnable examples
[group('workflow')]
workflow:
    @echo "Order matters: most model defects originate in the corpus, so scanning"
    @echo "first saves you a training run."
    @echo ""
    @echo "  1. just scan corpus/                       which encoding axes vary, per source"
    @echo "  2. just canon corpus/ canonical/           rewrite to canonical form, then verify"
    @echo "  3.    <train your tokenizer on canonical/>"
    @echo "  4. just check-model ocr.model              the artifact checks"
    @echo "     just check-all ocr.model canonical/     both checklists, corpus findings first"
    @echo ""
    @echo "Findings are ranked worst-first and each carries a remedy. A 'fix_corpus'"
    @echo "finding must be acted on before any retrain: retraining alone reproduces it."
    @echo "A SKIP is never a PASS, and a PRESERVE axis with a non-zero count is not a"
    @echo "defect — it is confirmation that script is present in your data."
    @echo ""
    @echo "Every recipe forwards extra flags:"
    @echo "  just scan corpus/ --jobs 2 --json"
    @echo "  just canon corpus/ out/ --decide soft_hyphen_line_final"
    @echo "  just check-model ocr.model --samples gt.txt --symbols syms.txt"
    @echo ""
    @echo "Full option list: just options"

# Step 1 — measure which encoding axes vary, and in which source
[group('workflow')]
[no-exit-message]
scan +PATHS:
    uv run spm-ocr corpus {{ PATHS }}

# Step 2 — rewrite a corpus into canonical form, then re-scan to verify it
[group('workflow')]
[no-exit-message]
canon SOURCE OUT *FLAGS:
    uv run spm-ocr canonicalize {{ SOURCE }} --out {{ OUT }} {{ FLAGS }}

# Step 2, overwriting the originals instead of writing copies
[group('workflow')]
[no-exit-message]
canon-in-place +PATHS:
    uv run spm-ocr canonicalize {{ PATHS }} --in-place

# Step 4 — run the artifact checks against a trained .model
[group('workflow')]
[no-exit-message]
check-model MODEL *FLAGS:
    uv run spm-ocr model {{ MODEL }} {{ FLAGS }}

# Both checklists in one run, corpus findings first
[group('workflow')]
[no-exit-message]
check-all MODEL CORPUS *FLAGS:
    uv run spm-ocr all {{ MODEL }} --corpus {{ CORPUS }} {{ FLAGS }}

# Show every command-line option for every subcommand
[group('workflow')]
options:
    @uv run spm-ocr --help
    @for cmd in corpus canonicalize model all; do \
        echo ""; echo "=== spm-ocr $cmd ==="; uv run spm-ocr $cmd --help; \
    done

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

# Install all dependencies (including dev)
[group('dev')]
sync:
    uv sync --group dev

# Run fmt-check + lint + types + test (CI gate)
[group('dev')]
check: fmt-check lint types test

# Format, lint-fix, then run all checks
[group('dev')]
fix: fmt lint-fix
    just check

# Run tests
[group('dev')]
test *ARGS:
    uv run pytest tests/ -q {{ ARGS }}

# Type-check with pyrefly
[group('dev')]
types:
    uv run pyrefly check sentencepiece_ocr_guide/ tests/

# Lint source and tests
[group('dev')]
lint:
    uv run ruff check sentencepiece_ocr_guide/ tests/

# Lint and auto-fix what's safe
[group('dev')]
lint-fix:
    uv run ruff check --fix sentencepiece_ocr_guide/ tests/

# Format source and tests
[group('dev')]
fmt:
    uv run ruff format sentencepiece_ocr_guide/ tests/

# Check formatting without modifying files
[group('dev')]
fmt-check:
    uv run ruff format --check sentencepiece_ocr_guide/ tests/

# Install git hooks
[group('dev')]
hooks:
    uvx prek install

# Run all git hooks against every file
[group('dev')]
hooks-run:
    uvx prek run --all-files

# Report the interpreter, whether the GIL is off, and the default worker count
[group('dev')]
runtime:
    @uv run python -c "import sys, os; \
        gil = getattr(sys, '_is_gil_enabled', lambda: True)(); \
        print(f'python {sys.version.split()[0]}   GIL enabled: {gil}   cores: {os.cpu_count()}')"
    @uv run python -c "from sentencepiece_ocr_guide.concurrency import default_workers; \
        print(f'default --jobs: {default_workers()}')"

# Remove build artifacts and caches
[group('dev')]
clean:
    rm -rf dist/ .venv/ .ruff_cache/ .pytest_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
