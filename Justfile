train config="cfg.json":
    RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" CFLAGS="${CFLAGS:-} -march=native" cargo run --release -- train --config {{config}}

check:
    cargo fmt --all -- --check
    cargo test
    cargo clippy --all-targets --all-features -- -D warnings
    RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" CFLAGS="${CFLAGS:-} -march=native" cargo build --release
    uv run ruff check .
    uv run --with pyrefly pyrefly check

fmt:
    cargo fmt --all
    uv run ruff format .
