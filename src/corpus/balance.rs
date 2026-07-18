//! What the corpus is made of, and what the trainer will refuse to read.
//!
//! Both of these are listed as manual steps in docs/08-validation.md. Neither has to be: the scan
//! already walks every line, so the share of each writing system and the count of over-length
//! lines cost nothing beyond the pass that was happening anyway.

use crate::corpus::scan::{Counts, DEFAULT_MAX_LINE_BYTES, Forms, Totals};
use crate::format::{count, percent};
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

/// Which normalization form the corpus is written in, per source.
///
/// `axis[nfc_composition]` already reports how many lines are not NFC, and fails on them. This
/// answers the question that count leaves open: whether those lines are a single source that is
/// uniformly NFD — one extractor to fix, and canonicalizing is clean — or sources that disagree
/// with each other. The shares are taken over the lines that *carry* the distinction, because a
/// corpus that is mostly ASCII would otherwise report itself as almost entirely composed and
/// mean nothing by it.
///
/// Reported, never graded: the defect is already failed by the axis, and this explains its shape.
pub fn normalization_forms(totals: &Totals) -> Finding {
    let mut combined = Forms::default();
    for (_, counts) in totals {
        combined.absorb(&counts.forms);
    }

    let decidable = combined.decidable();
    if decidable == 0 {
        return Finding::skipped(
            "normalization_forms",
            "no line carries a character that can be composed or decomposed",
        )
        .graded(Severity::Blocker, Remedy::FixCorpus);
    }

    Finding::passed(
        "normalization_forms",
        format!(
            "of {} lines that carry the distinction: {} composed, {} decomposed, {} mixed",
            count(decidable),
            percent(combined.composed, decidable),
            percent(combined.decomposed, decidable),
            percent(combined.mixed, decidable),
        ),
    )
    .with_evidence(per_source_forms(totals))
}

/// One line per source, the sources needing attention first.
///
/// A source that is uniformly one form is a single decision; one that is internally split is a
/// broken extractor, so ranking puts mixed lines above merely-decomposed ones.
fn per_source_forms(totals: &Totals) -> Vec<String> {
    let mut ranked: Vec<(f64, f64, String)> = totals
        .iter()
        .filter_map(|(name, counts)| {
            let forms = &counts.forms;
            let decidable = forms.decidable();
            if decidable == 0 {
                return None;
            }

            let line = format!(
                "{name}: {} composed, {} decomposed, {} mixed of {} lines",
                percent(forms.composed, decidable),
                percent(forms.decomposed, decidable),
                percent(forms.mixed, decidable),
                count(decidable),
            );
            Some((
                forms.mixed as f64 / decidable as f64,
                forms.decomposed as f64 / decidable as f64,
                line,
            ))
        })
        .collect();

    // Any mixed source outranks every unmixed one, however decomposed. Ordering the two signals
    // rather than weighting them keeps the rule the doc comment states, and avoids a weight that
    // nobody chose deciding which source you look at first.
    ranked.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    ranked.into_iter().map(|(_, _, line)| line).collect()
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
            forms: Forms::default(),
        }
    }

    fn with_forms(name: &str, composed: u64, decomposed: u64, mixed: u64) -> (String, Counts) {
        let mut c = counts(composed + decomposed + mixed, 0, &[]);
        c.forms = Forms {
            undecidable: 0,
            composed,
            decomposed,
            mixed,
        };
        (name.to_string(), c)
    }

    #[test]
    fn shares_are_taken_over_the_lines_that_carry_the_distinction() {
        // 900 ASCII lines say nothing about normalization and must not dilute the number.
        let mut ascii_heavy = counts(1000, 0, &[]);
        ascii_heavy.forms = Forms {
            undecidable: 900,
            composed: 50,
            decomposed: 50,
            mixed: 0,
        };

        let finding = normalization_forms(&vec![("shard.txt".to_string(), ascii_heavy)]);
        assert!(
            finding
                .summary
                .contains("of 100 lines that carry the distinction"),
            "{}",
            finding.summary
        );
        assert!(finding.summary.contains("50.0% composed"));
        assert!(finding.summary.contains("50.0% decomposed"));
    }

    #[test]
    fn a_corpus_with_nothing_composable_skips_rather_than_dividing_by_zero() {
        let finding = normalization_forms(&vec![("shard.txt".to_string(), counts(500, 0, &[]))]);
        assert_eq!(finding.status, Status::Skipped);
        assert_eq!(
            finding.severity,
            Severity::Blocker,
            "a skip keeps its grade"
        );
    }

    #[test]
    fn the_source_needing_attention_is_named_first() {
        // A source split within its own lines beats one that is merely uniformly decomposed.
        let totals = vec![
            with_forms("clean.txt", 100, 0, 0),
            with_forms("uniformly_nfd.txt", 0, 100, 0),
            with_forms("internally_split.txt", 50, 30, 20),
        ];

        let finding = normalization_forms(&totals);
        assert!(
            finding.evidence[0].starts_with("internally_split.txt"),
            "{:?}",
            finding.evidence
        );
        assert!(finding.evidence[2].starts_with("clean.txt"));
    }

    #[test]
    fn a_uniformly_decomposed_source_reads_as_one_decision() {
        let finding = normalization_forms(&vec![with_forms("vendor_a.txt", 0, 12000, 0)]);
        assert!(
            finding.evidence[0].contains("100.0% decomposed"),
            "{:?}",
            finding.evidence
        );
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
