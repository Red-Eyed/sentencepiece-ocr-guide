//! What the corpus is made of, and what the trainer will refuse to read.
//!
//! Both of these are listed as manual steps in docs/08-validation.md. Neither has to be: the scan
//! already walks every line, so the share of each writing system and the count of over-length
//! lines cost nothing beyond the pass that was happening anyway.

use crate::corpus::scan::{Counts, DEFAULT_MAX_LINE_BYTES};
use crate::report::{Finding, MAX_EVIDENCE, Remedy, Severity};

/// Where the line-length limit came from, which changes what the finding means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitSource {
    /// No model was supplied, so SentencePiece's default stands in.
    SentencePieceDefault,
    /// Read from a trained model's `max_sentence_length`.
    Model,
}

impl LimitSource {
    fn describe(self) -> &'static str {
        match self {
            LimitSource::SentencePieceDefault => "SentencePiece's default",
            LimitSource::Model => "the model's max_sentence_length",
        }
    }
}

/// The share of the corpus written in each script.
///
/// Reported, never graded. A corpus that is 90% Latin is a defect if you meant to train 100
/// languages evenly and correct if you meant to train English — and this tool cannot tell which.
/// What it can do is put the number in front of you, which is the step docs/05-corpus-engineering
/// asks you to take before computing any α-smoothing.
pub fn script_balance(counts: &Counts) -> Finding {
    let total = counts.script_characters();

    if total == 0 {
        return Finding::skipped(
            "script_balance",
            "no characters belonging to any writing system",
        )
        .graded(Severity::Medium, Remedy::FixCorpus);
    }

    let mut ranked: Vec<(&str, u64)> = counts
        .per_script
        .iter()
        .map(|(name, count)| (*name, *count))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let shares: Vec<String> = ranked
        .iter()
        .take(MAX_EVIDENCE)
        .map(|(name, count)| {
            let share = 100.0 * *count as f64 / total as f64;
            format!("{name}: {share:.1}% ({count} chars)")
        })
        .collect();

    Finding::passed(
        "script_balance",
        format!(
            "{} writing systems across {total} classified characters",
            ranked.len()
        ),
    )
    .with_evidence(shares)
}

/// Lines the trainer will silently discard.
///
/// The protobuf's own comment on `max_sentence_length` is that a longer sentence "is simply
/// ignored" — no warning, no error, no count. For OCR that is training data disappearing, and
/// long lines are not randomly distributed: they are the dense scripts and the full-page
/// transcriptions, which is exactly the material a page-level model needs most.
pub fn long_lines(counts: &Counts, limit: usize, source: LimitSource) -> Finding {
    if counts.long_lines == 0 {
        return Finding::passed(
            "long_lines",
            format!("no line exceeds {limit} bytes ({})", source.describe()),
        );
    }

    let share = 100.0 * counts.long_lines as f64 / counts.lines.max(1) as f64;

    Finding::failed(
        "long_lines",
        format!(
            "{} of {} lines ({share:.2}%) exceed {limit} bytes and will be dropped by the trainer",
            counts.long_lines, counts.lines
        ),
    )
    .with_evidence(vec![
        format!("the limit is {} ({limit} bytes)", source.describe()),
        "raise max_sentence_length, or split these lines before training".to_string(),
    ])
    .graded(Severity::High, Remedy::FixCorpus)
}

/// The limit to measure against when no model says otherwise.
pub fn default_limit() -> (usize, LimitSource) {
    (DEFAULT_MAX_LINE_BYTES, LimitSource::SentencePieceDefault)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::Status;
    use std::collections::BTreeMap;

    fn counts(lines: u64, long: u64, scripts: &[(&'static str, u64)]) -> Counts {
        Counts {
            lines,
            invalid_utf8: 0,
            long_lines: long,
            per_script: scripts.iter().copied().collect::<BTreeMap<_, _>>(),
            per_axis: BTreeMap::new(),
        }
    }

    #[test]
    fn balance_reports_shares_worst_first() {
        let finding = script_balance(&counts(10, 0, &[("Latin", 900), ("Han", 100)]));

        assert_eq!(finding.status, Status::Passed);
        assert!(finding.evidence[0].starts_with("Latin: 90.0%"));
        assert!(finding.evidence[1].starts_with("Han: 10.0%"));
    }

    #[test]
    fn balance_is_never_a_failure() {
        // A 99% Latin corpus is correct if that is what you meant to train.
        let finding = script_balance(&counts(10, 0, &[("Latin", 9900), ("Han", 100)]));
        assert_eq!(finding.status, Status::Passed);
    }

    #[test]
    fn a_corpus_with_no_classified_characters_skips() {
        let finding = script_balance(&counts(10, 0, &[]));
        assert_eq!(finding.status, Status::Skipped);
        assert_eq!(finding.severity, Severity::Medium, "a skip keeps its grade");
    }

    #[test]
    fn long_lines_fail_and_name_where_the_limit_came_from() {
        let finding = long_lines(&counts(1000, 12, &[]), 4192, LimitSource::Model);

        assert_eq!(finding.status, Status::Failed);
        assert_eq!(finding.severity, Severity::High);
        assert!(finding.summary.contains("12 of 1000"));
        assert!(finding.evidence[0].contains("max_sentence_length"));
    }

    #[test]
    fn no_long_lines_passes() {
        let finding = long_lines(
            &counts(1000, 0, &[]),
            4192,
            LimitSource::SentencePieceDefault,
        );
        assert_eq!(finding.status, Status::Passed);
        assert!(finding.summary.contains("SentencePiece's default"));
    }
}
