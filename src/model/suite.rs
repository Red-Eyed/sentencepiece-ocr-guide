//! The model checklist: every failure mode the guide can detect from the artifact alone.
//!
//! Assembling the suite is deliberately separate from the checks themselves. Adding a check
//! means adding a function and one line here, never editing an existing check.
//!
//! Two checks from the guide are absent by necessity rather than by oversight, and say so in the
//! report. `fertility` and the byte-fallback *rate* measure the tokenizer against real text, so
//! both need to encode it — and encoding needs a SentencePiece runtime, which reading the model
//! file does not give us. They are skipped with the reason attached and keep their severity,
//! because a check that could not run must never read as one that passed.

use crate::model::artifact::{Artifact, Normalizer, Trainer, Vocabulary};
use crate::model::{budget, config, pieces};
use crate::report::{Finding, Remedy, Report, Severity};

/// The tunables a caller may reasonably disagree with.
#[derive(Debug, Clone)]
pub struct Options {
    pub max_digit_piece_length: usize,
    /// Whether a digit fusing with a letter counts as a cross-script merge.
    pub digits_are_a_script: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_digit_piece_length: pieces::DEFAULT_MAX_DIGIT_PIECE_LENGTH,
            digits_are_a_script: true,
        }
    }
}

/// Run the standard suite over a parsed model.
pub fn check(artifact: &Artifact, options: &Options) -> Report {
    let mut findings = vocabulary_findings(&artifact.vocabulary, options);

    findings.extend(match &artifact.trainer {
        Some(trainer) => trainer_findings(&artifact.vocabulary, trainer),
        None => unavailable(TRAINER_CHECKS, "the model records no trainer spec"),
    });

    findings.extend(match &artifact.normalizer {
        Some(normalizer) => normalizer_findings(normalizer),
        None => unavailable(NORMALIZER_CHECKS, "the model records no normalizer spec"),
    });

    findings.extend(needs_a_tokenizer_runtime());

    Report::new(findings)
}

fn vocabulary_findings(vocabulary: &Vocabulary, options: &Options) -> Vec<Finding> {
    if vocabulary.is_empty() {
        return unavailable(VOCABULARY_CHECKS, "the model carries no pieces");
    }

    vec![
        pieces::digit_pieces(vocabulary, options.max_digit_piece_length).about(12),
        pieces::cross_script_pieces(vocabulary, options.digits_are_a_script).about(11),
        pieces::nfc_vocabulary(vocabulary).about(4),
        budget::vocabulary_budget(vocabulary).about(13),
    ]
}

fn trainer_findings(vocabulary: &Vocabulary, trainer: &Trainer) -> Vec<Finding> {
    vec![
        config::no_unknown(vocabulary, trainer).about(1),
        // `protected_symbols` is uncited: command atomicity is argued in docs/06-math-latex.md,
        // and the nearest numbered mode (#14, text/math imbalance) is a different failure.
        pieces::protected_symbols(vocabulary, trainer),
        config::algorithm(trainer).about(6),
        config::split_digits(trainer).about(12),
        config::script_splitting(trainer).about(11),
        // Spans #8, #9, #10 and #20 at once, so it cites none of them and reports the numbers.
        config::trainer_settings(trainer),
    ]
}

fn normalizer_findings(normalizer: &Normalizer) -> Vec<Finding> {
    vec![
        config::no_phantom_prefix(normalizer).about(21),
        config::whitespace_preserved(normalizer).about(2),
        config::normalization_rule(normalizer).about(2),
    ]
}

/// Each check, the severity it carries, and the guide entry it covers.
type Covered = (&'static str, Severity, Option<u8>);

const VOCABULARY_CHECKS: &[Covered] = &[
    ("digit_pieces", Severity::High, Some(12)),
    ("cross_script_pieces", Severity::High, Some(11)),
    ("nfc_vocabulary", Severity::Blocker, Some(4)),
];

const TRAINER_CHECKS: &[Covered] = &[
    ("no_unknown", Severity::Blocker, Some(1)),
    ("protected_symbols", Severity::High, None),
    ("algorithm", Severity::Medium, Some(6)),
    ("split_digits", Severity::High, Some(12)),
    ("script_splitting", Severity::Medium, Some(11)),
];

const NORMALIZER_CHECKS: &[Covered] = &[
    ("no_phantom_prefix", Severity::High, Some(21)),
    ("whitespace_preserved", Severity::High, Some(2)),
];

/// Checks that need to encode text, which reading the model file does not enable.
const NEEDS_ENCODING: &[Covered] = &[
    ("fertility", Severity::Medium, Some(13)),
    ("byte_fallback_rate", Severity::High, Some(17)),
];

fn needs_a_tokenizer_runtime() -> Vec<Finding> {
    unavailable(
        NEEDS_ENCODING,
        "measuring this requires encoding text, which reading the model file does not provide",
    )
}

/// Skips that keep the severity of the check they stand in for.
fn unavailable(checks: &[Covered], reason: &str) -> Vec<Finding> {
    checks
        .iter()
        .map(|(name, severity, mode)| {
            let finding = Finding::skipped(*name, reason).graded(*severity, Remedy::RetrainConfig);
            match mode {
                Some(mode) => finding.about(*mode),
                None => finding,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::artifact::{Kind, Piece};
    use crate::report::Status;
    use sentencepiece_model::ModelType;

    fn recommended_trainer() -> Trainer {
        Trainer {
            model_type: ModelType::Bpe,
            vocab_size: 40000,
            byte_fallback: true,
            split_digits: true,
            split_by_unicode_script: true,
            max_piece_length: 8,
            max_line_bytes: 4192,
            character_coverage: 0.9998,
            user_defined_symbols: Vec::new(),
        }
    }

    fn recommended_normalizer() -> Normalizer {
        Normalizer {
            rule_name: "identity".to_string(),
            has_charsmap: false,
            add_dummy_prefix: false,
            remove_extra_whitespaces: false,
        }
    }

    fn healthy_vocabulary() -> Vocabulary {
        let mut pieces: Vec<Piece> = (0..256)
            .map(|byte| Piece::new(format!("<0x{byte:02X}>"), Kind::Byte))
            .collect();
        pieces.push(Piece::new("hello", Kind::Normal));
        pieces.push(Piece::new("漢字", Kind::Normal));
        Vocabulary::new(pieces)
    }

    fn healthy() -> Artifact {
        Artifact {
            vocabulary: healthy_vocabulary(),
            trainer: Some(recommended_trainer()),
            normalizer: Some(recommended_normalizer()),
        }
    }

    fn finding<'a>(report: &'a Report, name: &str) -> &'a Finding {
        report
            .findings
            .iter()
            .find(|f| f.check == name)
            .unwrap_or_else(|| panic!("no finding named {name}"))
    }

    #[test]
    fn a_model_following_the_guide_has_no_failures() {
        let report = check(&healthy(), &Options::default());
        assert!(report.ok(), "unexpected failures: {:?}", report.findings);
    }

    #[test]
    fn the_two_encode_dependent_checks_are_skipped_with_a_reason() {
        let report = check(&healthy(), &Options::default());

        for name in ["fertility", "byte_fallback_rate"] {
            let skipped = finding(&report, name);
            assert_eq!(skipped.status, Status::Skipped);
            assert!(skipped.summary.contains("encoding text"));
        }
    }

    #[test]
    fn a_skipped_check_keeps_the_severity_it_stands_in_for() {
        // The point of the whole Status/Severity split: a skip must not read as clean.
        let report = check(&healthy(), &Options::default());
        assert_eq!(
            finding(&report, "byte_fallback_rate").severity,
            Severity::High
        );
    }

    #[test]
    fn a_model_missing_its_trainer_spec_skips_rather_than_assumes() {
        // Assuming the proto defaults would be a verdict on a setting nobody recorded.
        let artifact = Artifact {
            trainer: None,
            ..healthy()
        };
        let report = check(&artifact, &Options::default());

        let skipped = finding(&report, "no_unknown");
        assert_eq!(skipped.status, Status::Skipped);
        assert_eq!(skipped.severity, Severity::Blocker);
        assert!(report.ok(), "a skip is not a failure");
    }

    #[test]
    fn defects_surface_as_failures() {
        let mut trainer = recommended_trainer();
        trainer.byte_fallback = false;
        trainer.split_digits = false;

        let mut normalizer = recommended_normalizer();
        normalizer.add_dummy_prefix = true;

        let artifact = Artifact {
            vocabulary: healthy_vocabulary(),
            trainer: Some(trainer),
            normalizer: Some(normalizer),
        };
        let report = check(&artifact, &Options::default());

        assert!(!report.ok());
        assert_eq!(report.worst_severity(), Severity::Blocker);
        assert_eq!(finding(&report, "no_unknown").status, Status::Failed);
        assert_eq!(finding(&report, "split_digits").status, Status::Failed);
        assert_eq!(finding(&report, "no_phantom_prefix").status, Status::Failed);
    }

    #[test]
    fn an_empty_vocabulary_skips_the_piece_checks() {
        let artifact = Artifact {
            vocabulary: Vocabulary::default(),
            ..healthy()
        };
        let report = check(&artifact, &Options::default());
        assert_eq!(finding(&report, "nfc_vocabulary").status, Status::Skipped);
    }
}
