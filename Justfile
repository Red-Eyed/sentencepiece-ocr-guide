set dotenv-load := false

# The tool, built on demand. `cargo run` rebuilds only when something changed, so no recipe
# needs a "build first" note — and cargo's own output goes to stderr, leaving stdout clean
# for `--json`.
spm := "cargo run --release --quiet --"

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
    @echo "  3. spm-ocr train canonical/ \\"
    @echo "       --model-prefix ocr_tokenizer \\"
    @echo "       --training-temp-dir scratch/ --keep-training-file"
    @echo "  4. just check-model ocr_tokenizer.model    the artifact checks"
    @echo "     just check-all ocr_tokenizer.model canonical/"
    @echo "                                             both checklists, corpus findings first"
    @echo ""
    @echo "Findings are ranked worst-first and each carries a remedy. A 'fix_corpus'"
    @echo "finding must be acted on before any retrain: retraining alone reproduces it."
    @echo "A SKIP is never a PASS, and a PRESERVE axis with a non-zero count is not a"
    @echo "defect — it is confirmation that script is present in your data."
    @echo "Math-like lines are reported by default; use --balance-math for math-heavy runs."
    @echo ""
    @echo "The model checks read the .model file itself — its pieces and the trainer"
    @echo "settings recorded in it — so they need no samples and no tokenizer runtime."
    @echo "Two checks do need one: fertility and the exact byte-fallback rate measure"
    @echo "the tokenizer against real text, and are reported as SKIP with the reason."
    @echo ""
    @echo "Every recipe forwards extra flags:"
    @echo "  just scan corpus/ --jobs 2 --json"
    @echo "  just canon corpus/ out/ --decide soft_hyphen_line_final"
    @echo "  just check-model ocr.model --allow-digit-letter-pieces"
    @echo "  spm-ocr train corpus/ --lines 20000000 --json"
    @echo ""
    @echo "Full option list: just options"

# Step 1 — measure which encoding axes vary, and in which source
[group('workflow')]
[no-exit-message]
scan +PATHS:
    {{ spm }} corpus {{ PATHS }}

# Step 2 — rewrite a corpus into canonical form, then re-scan to verify it
[group('workflow')]
[no-exit-message]
canon SOURCE OUT *FLAGS:
    {{ spm }} canonicalize {{ SOURCE }} --out {{ OUT }} {{ FLAGS }}

# Step 2, overwriting the originals instead of writing copies
[group('workflow')]
[no-exit-message]
canon-in-place +PATHS:
    {{ spm }} canonicalize {{ PATHS }} --in-place

# Step 4 — run the artifact checks against a trained .model
[group('workflow')]
[no-exit-message]
check-model MODEL *FLAGS:
    {{ spm }} model {{ MODEL }} {{ FLAGS }}

# Both checklists in one run, corpus findings first
[group('workflow')]
[no-exit-message]
check-all MODEL CORPUS *FLAGS:
    {{ spm }} all {{ MODEL }} --corpus {{ CORPUS }} {{ FLAGS }}

# Show every command-line option for every subcommand
[group('workflow')]
options:
    @{{ spm }} --help
    @for cmd in corpus canonicalize model all train; do \
        echo ""; echo "=== spm-ocr $cmd ==="; {{ spm }} $cmd --help; \
    done

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

# Fetch dependencies
[group('dev')]
sync:
    cargo fetch

# Build the release binary
[group('dev')]
build:
    cargo build --release

# fmt-check + lint + test (CI gate)
[group('dev')]
check: fmt-check lint test

# Format, then run all checks
[group('dev')]
fix: fmt
    just check

# Run tests, unit and integration
[group('dev')]
test *ARGS:
    cargo test {{ ARGS }}

# Clippy with warnings as errors, so the deny-list in Cargo.toml is enforced
[group('dev')]
lint:
    cargo clippy --all-targets -- -D warnings

# Format
[group('dev')]
fmt:
    cargo fmt

# Check formatting without modifying files
[group('dev')]
fmt-check:
    cargo fmt --check

# Install the pre-commit hook, which runs `just check`
[group('dev')]
hooks:
    @mkdir -p .git/hooks
    @printf '#!/bin/sh\nexec just check\n' > .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo "installed .git/hooks/pre-commit -> just check"

# Report the toolchain and the parallelism the scan will default to
[group('dev')]
runtime:
    @cargo --version
    @rustc --version
    @echo "cores: $(getconf _NPROCESSORS_ONLN) (rayon's default --jobs)"

# Remove build artifacts
[group('dev')]
clean:
    cargo clean
