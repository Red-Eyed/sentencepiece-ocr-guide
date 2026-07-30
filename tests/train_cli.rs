#![allow(clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::json;

const SOFT_HYPHEN: char = '\u{00AD}';

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("output")
        .join("train_cli")
}

fn reset_output_dir(path: &Path) {
    if path.exists() {
        std::fs::remove_dir_all(path).expect("clear previous train_cli output");
    }
    std::fs::create_dir_all(path).expect("create train_cli output dir");
}

fn path_string(path: &Path) -> String {
    path.to_str()
        .expect("test paths should be UTF-8")
        .to_string()
}

fn write_generated_corpus(corpus_dir: &Path) {
    std::fs::create_dir_all(corpus_dir).expect("create corpus dir");

    let mut lines = Vec::new();
    for index in 0..80 {
        lines.push(format!(
            "invoice page {index} total 12345 serial A{index} multilingual OCR text"
        ));
        lines.push(format!(
            "страница {index} сумма 67890 номер кириллический текст"
        ));
        lines.push(format!("صفحة {index} مجموع 24680 نص عربي واضح"));
        lines.push(format!("पृष्ठ {index} कुल 13579 देवनागरी पाठ"));
        lines.push(format!("第{index}頁 合計 11223 漢字かな OCR"));
        lines.push(format!(
            "hyphenated wei{SOFT_HYPHEN}ter and wrapped line{SOFT_HYPHEN}"
        ));
        if index % 5 == 0 {
            lines.push(format!(
                "formula {index} \\frac {{ a _ {index} }} {{ b }} + \\sum"
            ));
        }
    }

    std::fs::write(corpus_dir.join("generated.txt"), lines.join("\n")).expect("write corpus");
}

fn write_symbols(path: &Path) {
    std::fs::write(path, "\\frac\n\\sum\n_\n{\n}\n").expect("write symbols");
}

fn write_config(
    path: &Path,
    corpus_dir: &Path,
    model_prefix: &Path,
    scratch_dir: &Path,
    symbols: &Path,
) {
    let config = json!({
        "paths": [path_string(corpus_dir)],
        "model_prefix": path_string(model_prefix),
        "lines": 160,
        "alpha": 0.5,
        "seed": 7,
        "vocab_size": 420,
        "character_coverage": 0.9998,
        "max_sentence_length": 512,
        "max_sentencepiece_length": 8,
        "shuffle_buffer_lines": 16,
        "memory_budget_gb": 1,
        "training_temp_dir": path_string(scratch_dir),
        "keep_training_file": true,
        "balance_math": true,
        "math_max_share": 0.05,
        "spm_threads": 1,
        "trainer_backend": "uv-python",
        "spm_train": "spm_train",
        "decide": ["soft_hyphen_line_final", "soft_hyphen_mid_line"],
        "drop_invalid": false,
        "drop_long_lines": false,
        "user_defined_symbols": [],
        "user_defined_symbols_file": path_string(symbols),
    });

    std::fs::write(
        path,
        serde_json::to_string_pretty(&config).expect("serialize config"),
    )
    .expect("write config");
}

fn assert_success(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "train command failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn train_command_trains_a_tokenizer_from_generated_text() {
    let output_dir = output_dir();
    reset_output_dir(&output_dir);

    let corpus_dir = output_dir.join("corpus");
    let scratch_dir = output_dir.join("scratch");
    let symbols = output_dir.join("symbols.txt");
    let config = output_dir.join("cfg.json");
    let model_prefix = output_dir.join("ocr_tokenizer");

    std::fs::create_dir_all(&scratch_dir).expect("create scratch dir");
    write_generated_corpus(&corpus_dir);
    write_symbols(&symbols);
    write_config(&config, &corpus_dir, &model_prefix, &scratch_dir, &symbols);

    let binary = PathBuf::from(env!("CARGO_BIN_EXE_spm-ocr"));
    let output = Command::new(binary)
        .arg("train")
        .arg("--config")
        .arg(&config)
        .arg("--fail-on")
        .arg("blocker")
        .env("UV_CACHE_DIR", output_dir.join("uv-cache"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run train command");

    assert_success(&output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("tokenizer_artifacts"),
        "report should name written artifacts\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains(&path_string(&model_prefix.with_extension("model"))),
        "report should include model path\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains("prepared corpus:"),
        "report should include kept prepared corpus path\nstdout:\n{stdout}"
    );
    assert!(model_prefix.with_extension("model").is_file());
    assert!(model_prefix.with_extension("vocab").is_file());
    assert!(
        scratch_dir
            .read_dir()
            .expect("read scratch dir")
            .next()
            .is_some(),
        "kept training input should remain in scratch dir"
    );
}
