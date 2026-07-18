//! The model half against real trained artifacts.
//!
//! Every unit test in `src/model/` builds its `Vocabulary` and `Trainer` by hand, which is what
//! makes those tests fast and total — and leaves exactly one thing unproven: that parsing a real
//! `.model` protobuf produces the values the checks are then reasoning about. A wrong field, a
//! misread enum or a proto default silently standing in for a recorded setting would pass every
//! unit test in the tree.
//!
//! The two fixtures are checked in because reproducing them needs a SentencePiece trainer, which
//! this project deliberately does not depend on. They were trained on the same small multilingual
//! corpus and differ only in configuration:
//!
//! - `guide_config.model` — trained the way docs/03-configuration.md argues for.
//! - `stock_defaults.model` — trained with SentencePiece's defaults, which is every silent
//!   failure the guide warns about, at once.

// Everything in this file is test code, but clippy's `allow-*-in-tests` only recognises `#[test]`
// functions and `#[cfg(test)]` modules — so the helpers below, which exist purely to report a
// failed fixture, read to it as production code. A panic in them is the failure report, which is
// the same reason clippy.toml exempts the unit tests.
#![allow(clippy::panic, clippy::expect_used)]

use std::path::PathBuf;

use spm_ocr::model::artifact::{self, Kind};
use spm_ocr::model::suite::{self, Options};
use spm_ocr::report::{Finding, Report, Severity, Status};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn report_for(name: &str) -> Report {
    let parsed = artifact::load(&fixture(name)).expect("fixture should parse");
    suite::check(&parsed, &Options::default())
}

fn finding<'a>(report: &'a Report, check: &str) -> &'a Finding {
    report
        .findings
        .iter()
        .find(|f| f.check == check)
        .unwrap_or_else(|| panic!("no finding named {check}"))
}

#[test]
fn a_real_model_parses_into_the_values_the_checks_read() {
    let parsed = artifact::load(&fixture("guide_config.model")).expect("should parse");

    let trainer = parsed
        .trainer
        .expect("a trained model records its trainer spec");
    assert_eq!(trainer.model_type, sentencepiece_model::ModelType::Bpe);
    assert!(trainer.byte_fallback);
    assert!(trainer.split_digits);
    assert!(trainer.split_by_unicode_script);
    assert_eq!(trainer.max_piece_length, 8);
    assert_eq!(trainer.user_defined_symbols, vec!["\\frac", "\\sum"]);

    let normalizer = parsed.normalizer.expect("and its normalizer spec");
    assert_eq!(normalizer.rule_name, "identity");
    assert!(!normalizer.has_charsmap, "identity folds nothing");
    assert!(!normalizer.add_dummy_prefix);
    assert!(!normalizer.remove_extra_whitespaces);
}

#[test]
fn byte_fallback_produces_one_piece_per_byte() {
    // The fact `no_unknown` rests on: the flag alone is not the guarantee.
    let parsed = artifact::load(&fixture("guide_config.model")).expect("should parse");
    assert_eq!(parsed.vocabulary.count_of(Kind::Byte), 256);
}

#[test]
fn declared_symbols_are_parsed_as_user_defined_pieces() {
    let parsed = artifact::load(&fixture("guide_config.model")).expect("should parse");
    assert!(parsed.vocabulary.has_user_defined("\\frac"));
    assert!(parsed.vocabulary.has_user_defined("\\sum"));
}

#[test]
fn a_model_following_the_guide_passes() {
    let report = report_for("guide_config.model");
    assert!(report.ok(), "unexpected failures: {:?}", report.findings);
    assert_eq!(report.exit_code(Severity::High), 0);
}

#[test]
fn stock_defaults_fail_every_check_the_guide_warns_about() {
    let report = report_for("stock_defaults.model");

    // Failure mode #1: no byte fallback, so some input encodes to an unrecoverable <unk>.
    let unknown = finding(&report, "no_unknown");
    assert_eq!(unknown.status, Status::Failed);
    assert_eq!(unknown.severity, Severity::Blocker);

    for check in [
        "no_phantom_prefix",    // add_dummy_prefix defaults on
        "whitespace_preserved", // remove_extra_whitespaces defaults on
        "split_digits",         // defaults off
        "algorithm",            // defaults to Unigram
    ] {
        assert_eq!(
            finding(&report, check).status,
            Status::Failed,
            "{check} should fail on stock defaults"
        );
    }

    assert_eq!(report.exit_code(Severity::High), 1);
}

#[test]
fn ordinary_corpus_frequency_merges_multi_digit_pieces() {
    // The claim behind `split_digits=True`: nothing exotic is needed to get `100` as one token,
    // just numbers appearing in the corpus at ordinary rates.
    let report = report_for("stock_defaults.model");

    let digits = finding(&report, "digit_pieces");
    assert_eq!(digits.status, Status::Failed);
    assert!(
        !digits.evidence.is_empty(),
        "the merged pieces should be named"
    );
}

#[test]
fn a_folding_normalizer_is_reported_rather_than_failed() {
    // stock defaults use nmt_nfkc. Whether that is wrong depends on the ground truth, which the
    // tool cannot see — so it must not fail the run.
    let report = report_for("stock_defaults.model");

    let rule = finding(&report, "normalization_rule");
    assert_eq!(rule.status, Status::Passed);
    assert!(rule.summary.contains("folds characters"));
}

#[test]
fn the_encode_dependent_checks_skip_on_a_real_model_too() {
    let report = report_for("guide_config.model");

    for check in ["fertility", "byte_fallback_rate"] {
        let skipped = finding(&report, check);
        assert_eq!(skipped.status, Status::Skipped);
        assert_ne!(
            skipped.severity,
            Severity::Info,
            "a skip keeps the severity of the check it stands in for"
        );
    }
}

#[test]
fn a_file_that_is_not_a_model_is_an_error_not_a_panic() {
    let error = artifact::load(&fixture("does_not_exist.model"));
    assert!(error.is_err());
}
