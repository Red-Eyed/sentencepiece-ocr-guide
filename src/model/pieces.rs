//! Checks that read the vocabulary itself.
//!
//! What the trained artifact says about the corpus behind it. Every one of these is a statement
//! over the whole inventory rather than over a sample, so a pass here means the property holds
//! for every input the tokenizer can ever be given.

use crate::model::artifact::{Trainer, Vocabulary, surface};
use crate::report::{Finding, Remedy, Severity};
use crate::writing::{is_all_digits, writing_systems_in};
use unicode_normalization::{UnicodeNormalization, is_nfc};

/// A digit-only piece longer than this teaches the decoder to reproduce familiar numbers.
pub const DEFAULT_MAX_DIGIT_PIECE_LENGTH: usize = 1;

/// No digit-only piece longer than `max_length` characters.
///
/// Failure mode #12: a merged multi-digit token trains the model to reproduce *frequent* numbers
/// rather than read the digits on the page. On invoices, tables and math — where the numbers are
/// the whole point and are never the ones in the corpus — that is a direct accuracy loss.
pub fn digit_pieces(vocabulary: &Vocabulary, max_length: usize) -> Finding {
    let offenders: Vec<String> = vocabulary
        .inspectable()
        .filter(|piece| {
            let text = surface(&piece.text);
            let text = text.trim();
            is_all_digits(text) && text.chars().count() > max_length
        })
        // The raw text, because `250` and `▁250` are distinct vocabulary entries.
        .map(|piece| format!("{:?}", piece.text))
        .collect();

    if offenders.is_empty() {
        return Finding::passed(
            "digit_pieces",
            format!("no digit-only piece longer than {max_length}"),
        );
    }

    Finding::failed(
        "digit_pieces",
        format!(
            "{} digit-only pieces exceed {max_length} characters",
            offenders.len()
        ),
    )
    .with_evidence(offenders)
    .graded(Severity::High, Remedy::RetrainConfig)
}

/// No piece may straddle two writing systems.
///
/// Failure mode #11: a piece spanning Latin and Han spends a vocabulary slot on a sequence that
/// occurs only at incidental boundaries, and adds a confusable class. This is what
/// `split_by_unicode_script=True` is meant to prevent — this check confirms that it did.
pub fn cross_script_pieces(vocabulary: &Vocabulary, digits_are_a_script: bool) -> Finding {
    let offenders: Vec<String> = vocabulary
        .inspectable()
        .filter_map(|piece| {
            let text = surface(&piece.text);
            let mut present = writing_systems_in(&text);

            if !digits_are_a_script {
                present.retain(|writing| writing.name() != "Digit");
            }

            if present.len() <= 1 {
                return None;
            }

            let names: Vec<&str> = present.iter().map(|w| w.name()).collect();
            Some(format!("{text:?} spans {}", names.join("+")))
        })
        .collect();

    if offenders.is_empty() {
        return Finding::passed(
            "cross_script_pieces",
            "no cross-script pieces in the vocabulary",
        );
    }

    Finding::failed(
        "cross_script_pieces",
        format!(
            "{} vocabulary pieces span more than one script",
            offenders.len()
        ),
    )
    .with_evidence(offenders)
    .graded(Severity::High, Remedy::RetrainConfig)
}

/// Every piece must already be in NFC.
///
/// Failure mode #4. A corpus mixing NFC and NFD trains a tokenizer where `café` composed and
/// `café` decomposed are unrelated token sequences for text that renders identically — every
/// affected grapheme trains at a fraction of its true frequency.
///
/// This check exists because a round-trip cannot see the defect: under `identity` normalization
/// both forms round-trip perfectly, since the tokenizer is faithfully reproducing an
/// inconsistency it was handed. The evidence is in the vocabulary instead.
///
/// The signal is one-directional. Non-NFC pieces prove the corpus carried decomposed text; their
/// absence is strong but not conclusive evidence that it did not.
pub fn nfc_vocabulary(vocabulary: &Vocabulary) -> Finding {
    let offenders: Vec<String> = vocabulary
        .inspectable()
        .filter(|piece| !is_nfc(&piece.text))
        .map(|piece| {
            let composed: String = piece.text.nfc().collect();
            // Escaped, because the two forms are visually identical — which is the problem.
            format!("{:?} should be {:?}", piece.text, composed)
        })
        .collect();

    if offenders.is_empty() {
        return Finding::passed("nfc_vocabulary", "every vocabulary piece is in NFC");
    }

    Finding::failed(
        "nfc_vocabulary",
        format!(
            "{} vocabulary pieces are not NFC — the corpus mixed NFC and NFD",
            offenders.len()
        ),
    )
    .with_evidence(offenders)
    .graded(Severity::Blocker, Remedy::FixCorpus)
}

/// Every declared `user_defined_symbols` entry must be present as a user-defined piece.
///
/// A LaTeX command that fragments inflates sequence length and multiplies the ways a decoder can
/// emit an invalid command. Declaring a symbol is not the same as it taking effect.
///
/// The symbols come from the model's own trainer spec rather than from the operator, so this
/// checks what was actually requested rather than what someone remembered to re-supply.
pub fn protected_symbols(vocabulary: &Vocabulary, trainer: &Trainer) -> Finding {
    if trainer.user_defined_symbols.is_empty() {
        return Finding::skipped(
            "protected_symbols",
            "the model declares no user_defined_symbols",
        )
        .graded(Severity::High, Remedy::RetrainConfig);
    }

    let missing: Vec<String> = trainer
        .user_defined_symbols
        .iter()
        .filter(|symbol| !vocabulary.has_user_defined(symbol))
        .map(|symbol| format!("{symbol:?} is declared but not in the vocabulary as one piece"))
        .collect();

    let declared = trainer.user_defined_symbols.len();

    if missing.is_empty() {
        return Finding::passed(
            "protected_symbols",
            format!("all {declared} protected symbols are atomic"),
        );
    }

    Finding::failed(
        "protected_symbols",
        format!(
            "{} of {declared} protected symbols did not survive training",
            missing.len()
        ),
    )
    .with_evidence(missing)
    .graded(Severity::High, Remedy::RetrainConfig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::artifact::{Kind, Piece};
    use crate::report::Status;

    fn vocabulary(pieces: &[(&str, Kind)]) -> Vocabulary {
        Vocabulary::new(
            pieces
                .iter()
                .map(|(text, kind)| Piece::new(*text, *kind))
                .collect(),
        )
    }

    fn trainer_with(symbols: &[&str]) -> Trainer {
        Trainer {
            model_type: sentencepiece_model::ModelType::Bpe,
            vocab_size: 40000,
            byte_fallback: true,
            split_digits: true,
            split_by_unicode_script: true,
            max_piece_length: 8,
            max_line_bytes: 4192,
            character_coverage: 0.9998,
            user_defined_symbols: symbols.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    #[test]
    fn a_merged_multi_digit_piece_fails() {
        let vocabulary = vocabulary(&[("100", Kind::Normal), ("7", Kind::Normal)]);
        let finding = digit_pieces(&vocabulary, DEFAULT_MAX_DIGIT_PIECE_LENGTH);

        assert_eq!(finding.status, Status::Failed);
        assert_eq!(finding.severity, Severity::High);
        assert_eq!(finding.evidence, vec!["\"100\""]);
    }

    #[test]
    fn a_digit_piece_is_reported_with_its_space_marker_intact() {
        // `250` and `▁250` are different entries; the evidence has to say which one.
        let vocabulary = vocabulary(&[("▁250", Kind::Normal)]);
        let finding = digit_pieces(&vocabulary, 1);
        assert_eq!(finding.evidence, vec!["\"▁250\""]);
    }

    #[test]
    fn byte_pieces_never_count_as_digit_pieces() {
        let vocabulary = vocabulary(&[("<0x30>", Kind::Byte)]);
        assert_eq!(digit_pieces(&vocabulary, 1).status, Status::Passed);
    }

    #[test]
    fn a_piece_spanning_two_scripts_fails() {
        let vocabulary = vocabulary(&[("a漢", Kind::Normal), ("hello", Kind::Normal)]);
        let finding = cross_script_pieces(&vocabulary, true);

        assert_eq!(finding.status, Status::Failed);
        assert!(finding.evidence[0].contains("Han+Latin"));
    }

    #[test]
    fn digit_letter_merges_are_optional() {
        let vocabulary = vocabulary(&[("3D", Kind::Normal)]);

        assert_eq!(
            cross_script_pieces(&vocabulary, true).status,
            Status::Failed
        );
        assert_eq!(
            cross_script_pieces(&vocabulary, false).status,
            Status::Passed,
            "opted out, so a digit fusing with a letter is allowed"
        );
    }

    #[test]
    fn a_decomposed_piece_fails_as_a_corpus_defect() {
        let vocabulary = vocabulary(&[("cafe\u{0301}", Kind::Normal)]);
        let finding = nfc_vocabulary(&vocabulary);

        assert_eq!(finding.status, Status::Failed);
        assert_eq!(finding.severity, Severity::Blocker);
        assert_eq!(
            finding.remedy,
            Remedy::FixCorpus,
            "retraining alone reproduces it exactly"
        );
    }

    #[test]
    fn composed_pieces_pass() {
        let vocabulary = vocabulary(&[("café", Kind::Normal), ("漢字", Kind::Normal)]);
        assert_eq!(nfc_vocabulary(&vocabulary).status, Status::Passed);
    }

    #[test]
    fn a_declared_symbol_that_did_not_survive_fails() {
        let vocabulary = vocabulary(&[("\\frac", Kind::UserDefined)]);
        let finding = protected_symbols(&vocabulary, &trainer_with(&["\\frac", "\\sum"]));

        assert_eq!(finding.status, Status::Failed);
        assert!(finding.evidence[0].contains("\\\\sum"));
    }

    #[test]
    fn declared_symbols_all_present_pass() {
        let vocabulary = vocabulary(&[("\\frac", Kind::UserDefined)]);
        let finding = protected_symbols(&vocabulary, &trainer_with(&["\\frac"]));
        assert_eq!(finding.status, Status::Passed);
    }

    #[test]
    fn a_skipped_symbol_check_keeps_its_severity() {
        // A skipped HIGH must not read as clean.
        let finding = protected_symbols(&vocabulary(&[]), &trainer_with(&[]));
        assert_eq!(finding.status, Status::Skipped);
    }
}
