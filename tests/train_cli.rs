#![allow(clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::json;

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
        "decide": [],
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

fn assert_success(output: std::process::Output) {
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
    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus_dir = temp.path().join("corpus");
    let scratch_dir = temp.path().join("scratch");
    let symbols = temp.path().join("symbols.txt");
    let config = temp.path().join("cfg.json");
    let model_prefix = temp.path().join("ocr_tokenizer");

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
        .env("UV_CACHE_DIR", temp.path().join("uv-cache"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run train command");

    assert_success(output);
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
