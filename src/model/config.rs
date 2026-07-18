//! Checks that read the settings the model was trained with.
//!
//! Three of these replace checks the Python half performs by experiment — encoding samples and
//! inferring a flag from what comes back. The flag is recorded in the artifact, so reading it is
//! both exact and a statement about every possible input rather than about the samples on hand.
//!
//! Not every recommendation in the guide is checkable. Where the right answer depends on
//! something the tool cannot see — the normalization of *your* ground truth, your vocabulary
//! budget — the setting is reported rather than judged. A check that fails on a value it cannot
//! evaluate teaches its reader to ignore it.

use sentencepiece_model::ModelType;

use crate::model::artifact::{Kind, Normalizer, Trainer, Vocabulary};
use crate::report::{Finding, Remedy, Severity};

/// Byte fallback needs one piece per possible byte to be total.
const BYTE_PIECES: usize = 256;

/// `<unk>` must be unreachable for every possible input.
///
/// Failure mode #1: for OCR an `<unk>` is a *label*, so the character becomes permanently
/// unreadable rather than merely misread. This is the one check with no acceptable failure
/// threshold — which is exactly why it is answered structurally instead of by sampling. With
/// byte fallback complete, no input can produce an unknown token; without it, some input can,
/// whether or not any sample happened to find it.
pub fn no_unknown(vocabulary: &Vocabulary, trainer: &Trainer) -> Finding {
    let byte_pieces = vocabulary.count_of(Kind::Byte);

    if !trainer.byte_fallback {
        return Finding::failed(
            "no_unknown",
            "byte_fallback is off — any character outside the vocabulary encodes to <unk>",
        )
        .graded(Severity::Blocker, Remedy::RetrainConfig);
    }

    if byte_pieces != BYTE_PIECES {
        return Finding::failed(
            "no_unknown",
            format!(
                "byte_fallback is on but the vocabulary carries {byte_pieces} byte pieces, not {BYTE_PIECES}"
            ),
        )
        .with_evidence(vec![
            "the fallback is incomplete, so some bytes still have no representation".to_string(),
        ])
        .graded(Severity::Blocker, Remedy::RetrainConfig);
    }

    Finding::passed(
        "no_unknown",
        format!("byte_fallback is on with all {BYTE_PIECES} byte pieces — <unk> is unreachable"),
    )
}

/// No phantom leading space on text that did not begin with one.
///
/// `add_dummy_prefix` prepends a space marker to every line. A round-trip still passes, because
/// decoding strips it again, so this corruption is invisible to the highest-value check in the
/// guide and needs its own. For CJK and Thai, where lines carry no leading whitespace, it puts a
/// spurious token at the start of every single label.
pub fn no_phantom_prefix(normalizer: &Normalizer) -> Finding {
    if normalizer.add_dummy_prefix {
        return Finding::failed(
            "no_phantom_prefix",
            "add_dummy_prefix is on — every label gains a leading space marker it did not have",
        )
        .graded(Severity::High, Remedy::RetrainConfig);
    }

    Finding::passed(
        "no_phantom_prefix",
        "add_dummy_prefix is off — no phantom leading space",
    )
}

/// Whitespace in the ground truth must survive tokenization.
///
/// `remove_extra_whitespaces` folds runs of spaces and strips the edges. For a language model
/// that is harmless tidying; for OCR the whitespace is part of the label, and folding it makes
/// the tokenizer unable to reproduce the line it was asked to read.
pub fn whitespace_preserved(normalizer: &Normalizer) -> Finding {
    if normalizer.remove_extra_whitespaces {
        return Finding::failed(
            "whitespace_preserved",
            "remove_extra_whitespaces is on — runs of spaces are folded and edges stripped",
        )
        .graded(Severity::High, Remedy::RetrainConfig);
    }

    Finding::passed(
        "whitespace_preserved",
        "remove_extra_whitespaces is off — whitespace reaches the tokenizer intact",
    )
}

/// What normalization the tokenizer applies, reported rather than judged.
///
/// The guide recommends `identity`, but the actual rule is that tokenizer normalization must
/// *match* ground-truth normalization — and the tool cannot see your ground truth. A folding
/// rule is correct if your labels are folded the same way and a defect if they are not, so this
/// states what the model does and leaves the verdict to the reader.
pub fn normalization_rule(normalizer: &Normalizer) -> Finding {
    let rule = if normalizer.rule_name.is_empty() {
        "unnamed"
    } else {
        &normalizer.rule_name
    };

    let effect = if normalizer.has_charsmap {
        "folds characters before tokenizing"
    } else {
        "no character folding"
    };

    Finding::passed(
        "normalization_rule",
        format!("normalization is {rule:?} — {effect} (verify this matches your ground truth)"),
    )
}

/// BPE rather than Unigram.
///
/// Unigram's subword regularization is inert when the decoder trains against one canonical
/// ground-truth segmentation, and it costs determinism. A Unigram model is not broken, which is
/// why this is not a blocker — but it is the guide's headline recommendation.
pub fn algorithm(trainer: &Trainer) -> Finding {
    if trainer.model_type == ModelType::Bpe {
        return Finding::passed("algorithm", "model_type is BPE");
    }

    Finding::failed(
        "algorithm",
        format!(
            "model_type is {} — the guide recommends BPE for an OCR decoder",
            display_model_type(trainer.model_type)
        ),
    )
    .with_evidence(vec![
        "subword regularization is inert against one canonical segmentation, and costs determinism"
            .to_string(),
    ])
    .graded(Severity::Medium, Remedy::RetrainConfig)
}

/// Digits split into individual tokens.
///
/// Off by default, and ordinary corpus frequency is enough to merge `100` into one token — after
/// which the model reproduces familiar numbers instead of reading the ones on the page. This is
/// the setting; [`super::pieces::digit_pieces`] is the observed consequence in the vocabulary.
pub fn split_digits(trainer: &Trainer) -> Finding {
    if trainer.split_digits {
        return Finding::passed("split_digits", "split_digits is on");
    }

    Finding::failed(
        "split_digits",
        "split_digits is off — multi-digit tokens will form from corpus frequency",
    )
    .graded(Severity::High, Remedy::RetrainConfig)
}

/// Pieces may not form across script boundaries.
///
/// The setting behind [`super::pieces::cross_script_pieces`]. Both are reported: the flag says
/// what was asked for, the vocabulary says what happened.
pub fn script_splitting(trainer: &Trainer) -> Finding {
    if trainer.split_by_unicode_script {
        return Finding::passed("script_splitting", "split_by_unicode_script is on");
    }

    Finding::failed(
        "script_splitting",
        "split_by_unicode_script is off — pieces may merge across writing systems",
    )
    .graded(Severity::Medium, Remedy::RetrainConfig)
}

/// The tunables, reported for the record.
///
/// Vocabulary size, character coverage and maximum piece length are trade-offs against your
/// script mix and decoder budget. The guide argues values for them; none has a threshold this
/// tool can defend on its own, so they are stated rather than graded.
pub fn trainer_settings(trainer: &Trainer) -> Finding {
    Finding::passed(
        "trainer_settings",
        format!(
            "vocab_size={}, character_coverage={:.4}, max_sentencepiece_length={}",
            trainer.vocab_size, trainer.character_coverage, trainer.max_piece_length
        ),
    )
}

fn display_model_type(model_type: ModelType) -> &'static str {
    match model_type {
        ModelType::Bpe => "BPE",
        ModelType::Unigram => "Unigram",
        ModelType::Word => "word",
        ModelType::Char => "char",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::artifact::Piece;
    use crate::report::Status;

    fn trainer() -> Trainer {
        Trainer {
            model_type: ModelType::Bpe,
            vocab_size: 40000,
            byte_fallback: true,
            split_digits: true,
            split_by_unicode_script: true,
            max_piece_length: 8,
            character_coverage: 0.9998,
            user_defined_symbols: Vec::new(),
        }
    }

    fn normalizer() -> Normalizer {
        Normalizer {
            rule_name: "identity".to_string(),
            has_charsmap: false,
            add_dummy_prefix: false,
            remove_extra_whitespaces: false,
        }
    }

    fn all_byte_pieces() -> Vocabulary {
        Vocabulary::new(
            (0..BYTE_PIECES)
                .map(|byte| Piece::new(format!("<0x{byte:02X}>"), Kind::Byte))
                .collect(),
        )
    }

    #[test]
    fn complete_byte_fallback_makes_unknown_unreachable() {
        let finding = no_unknown(&all_byte_pieces(), &trainer());
        assert_eq!(finding.status, Status::Passed);
    }

    #[test]
    fn byte_fallback_off_is_a_blocker() {
        let mut trainer = trainer();
        trainer.byte_fallback = false;

        let finding = no_unknown(&all_byte_pieces(), &trainer);
        assert_eq!(finding.status, Status::Failed);
        assert_eq!(finding.severity, Severity::Blocker);
    }

    #[test]
    fn byte_fallback_on_but_incomplete_is_still_a_blocker() {
        // The flag alone is not the guarantee; the pieces have to be there.
        let partial = Vocabulary::new(vec![Piece::new("<0x00>", Kind::Byte)]);

        let finding = no_unknown(&partial, &trainer());
        assert_eq!(finding.status, Status::Failed);
        assert!(finding.summary.contains("1 byte pieces"));
    }

    #[test]
    fn add_dummy_prefix_on_fails() {
        let mut normalizer = normalizer();
        normalizer.add_dummy_prefix = true;

        let finding = no_phantom_prefix(&normalizer);
        assert_eq!(finding.status, Status::Failed);
        assert_eq!(finding.severity, Severity::High);
    }

    #[test]
    fn folding_whitespace_fails_for_ocr() {
        let mut normalizer = normalizer();
        normalizer.remove_extra_whitespaces = true;
        assert_eq!(whitespace_preserved(&normalizer).status, Status::Failed);
    }

    #[test]
    fn a_folding_normalizer_is_reported_not_failed() {
        // Correct or not depends on the ground truth, which this tool cannot see.
        let mut normalizer = normalizer();
        normalizer.rule_name = "nfkc".to_string();
        normalizer.has_charsmap = true;

        let finding = normalization_rule(&normalizer);
        assert_eq!(finding.status, Status::Passed);
        assert!(finding.summary.contains("folds characters"));
        assert!(finding.summary.contains("verify"));
    }

    #[test]
    fn unigram_is_flagged_but_is_not_a_blocker() {
        let mut trainer = trainer();
        trainer.model_type = ModelType::Unigram;

        let finding = algorithm(&trainer);
        assert_eq!(finding.status, Status::Failed);
        assert_eq!(finding.severity, Severity::Medium);
        assert!(finding.summary.contains("Unigram"));
    }

    #[test]
    fn split_digits_off_fails() {
        let mut trainer = trainer();
        trainer.split_digits = false;
        assert_eq!(split_digits(&trainer).status, Status::Failed);
    }

    #[test]
    fn script_splitting_off_fails() {
        let mut trainer = trainer();
        trainer.split_by_unicode_script = false;
        assert_eq!(script_splitting(&trainer).status, Status::Failed);
    }

    #[test]
    fn tunables_are_reported_never_graded() {
        let finding = trainer_settings(&trainer());
        assert_eq!(finding.status, Status::Passed);
        assert!(finding.summary.contains("40000"));
    }
}
