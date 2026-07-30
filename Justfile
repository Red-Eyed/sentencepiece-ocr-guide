set dotenv-load := false

# The tool, built on demand. `cargo run` rebuilds only when something changed, so no recipe
# needs a "build first" note.
spm := "cargo run --release --"

# List available recipes, grouped
default:
    @just --list --unsorted

# ---------------------------------------------------------------------------
# Workflow — train the tokenizer with preprocessing, balancing, and validation
# ---------------------------------------------------------------------------

# Print the tokenizer training path
[group('workflow')]
workflow:
    @echo "Copy cfg.json.example to cfg.json, edit paths/model_prefix if needed,"
    @echo "then run:"
    @echo ""
    @echo "  just train cfg.json"
    @echo ""
    @echo "Training includes corpus discovery, preflight checks, safe in-stream"
    @echo "canonicalization, alpha-balanced sampling, SentencePiece training, and"
    @echo "post-train model checks."

# Train the tokenizer from strict JSON config
[group('workflow')]
[no-exit-message]
train CONFIG:
    {{ spm }} train --config {{ CONFIG }}

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
