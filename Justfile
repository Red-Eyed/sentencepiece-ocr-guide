train config="cfg.json":
    cargo run -- train --config {{config}}

check:
    cargo fmt --all -- --check
    cargo test
    cargo clippy --all-targets --all-features -- -D warnings
    uv run ruff check .
    uv run --with pyrefly pyrefly check

fmt:
    cargo fmt --all
    uv run ruff format .
