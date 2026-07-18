//! What the corpus and the model say about each other.
//!
//! Everything else in this tool judges one artifact against the guide. These checks judge the two
//! against *each other*, which is the one thing neither checklist can do alone — and the reason
//! `spm-ocr all` is more than the two reports concatenated.
//!
//! Both findings here answer questions docs/08-validation.md leaves to the reader.

use crate::corpus::balance::{LimitSource, long_lines};
use crate::corpus::scan::Counts;
use crate::model::artifact::Artifact;
use crate::model::budget::pieces_per_script;
use crate::report::{Finding, MAX_EVIDENCE, Remedy, Report, Severity};

/// A script must be at least this much of the corpus before its vocabulary share is worth
/// judging. Below it, a small allocation is proportionate rather than a defect.
const MATERIAL_CORPUS_SHARE: f64 = 0.01;

/// Getting less than this fraction of proportional allocation is the signal `fertility` used to
/// catch. A tenth is not a threshold anyone derived — it is far enough below parity that the
/// alternative explanations run out, and the finding prints both numbers so the call stays yours.
const STARVED_RATIO: f64 = 0.1;

/// Run the checks that need both halves.
pub fn check(corpus: &Counts, artifact: &Artifact, limit: usize) -> Report {
    Report::new(vec![
        long_lines(corpus, limit, LimitSource::Model).about(15),
        script_coverage(corpus, artifact).about(13),
    ])
}

/// The line limit to scan a corpus against, taken from the model when it records one.
pub fn line_limit(artifact: &Artifact) -> Option<usize> {
    artifact
        .trainer
        .as_ref()
        .map(|trainer| trainer.max_line_bytes)
}

/// Scripts the corpus is written in, against the vocabulary they were given.
///
/// Failure mode #13: the vocabulary budget follows frequency, so an under-represented script gets
/// few merges and its text fragments into short pieces. `fertility` caught this by encoding text
/// and measuring tokens per character; this catches it by comparing what a script contributed to
/// the corpus against what it received in the vocabulary.
///
/// Fair shares are not proportional — CJK needs far more pieces than Latin for the same fluency —
/// so only the lopsided case is graded, and the numbers travel with the finding either way.
pub fn script_coverage(corpus: &Counts, artifact: &Artifact) -> Finding {
    let corpus_total = corpus.script_characters();
    let per_script = pieces_per_script(&artifact.vocabulary);
    let vocabulary_total: u64 = per_script.values().sum();

    if corpus_total == 0 || vocabulary_total == 0 {
        return Finding::skipped(
            "script_coverage",
            "the corpus or the vocabulary has no classified characters to compare",
        )
        .graded(Severity::Medium, Remedy::FixCorpus);
    }

    let mut starved: Vec<(f64, String)> = Vec::new();

    for (script, characters) in &corpus.per_script {
        let corpus_share = *characters as f64 / corpus_total as f64;
        if corpus_share < MATERIAL_CORPUS_SHARE {
            continue;
        }

        let pieces = per_script.get(script).copied().unwrap_or(0);
        let vocabulary_share = pieces as f64 / vocabulary_total as f64;

        if vocabulary_share >= corpus_share * STARVED_RATIO {
            continue;
        }

        let corpus_percent = 100.0 * corpus_share;
        let vocabulary_percent = 100.0 * vocabulary_share;

        starved.push((
            vocabulary_share / corpus_share,
            format!(
                "{script}: {corpus_percent:.1}% of corpus characters, {vocabulary_percent:.1}% of the vocabulary ({pieces} pieces)"
            ),
        ));
    }

    if starved.is_empty() {
        let floor = 100.0 * MATERIAL_CORPUS_SHARE;
        return Finding::passed(
            "script_coverage",
            format!("every script above {floor:.0}% of the corpus has a proportionate vocabulary"),
        );
    }

    // Worst-starved first: the smallest ratio is the script fragmenting hardest.
    starved.sort_by(|a, b| a.0.total_cmp(&b.0));

    let subject = if starved.len() == 1 {
        "1 script"
    } else {
        &format!("{} scripts", starved.len())
    };

    Finding::failed(
        "script_coverage",
        format!("{subject} received far less vocabulary than its share of the corpus"),
    )
    .with_evidence(starved.into_iter().take(MAX_EVIDENCE).map(|(_, e)| e))
    .graded(Severity::Medium, Remedy::FixCorpus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::artifact::{Kind, Piece, Trainer, Vocabulary};
    use crate::report::Status;
    use sentencepiece_model::ModelType;
    use std::collections::BTreeMap;

    fn corpus(scripts: &[(&'static str, u64)]) -> Counts {
        Counts {
            lines: 100,
            invalid_utf8: 0,
            long_lines: 0,
            per_script: scripts.iter().copied().collect::<BTreeMap<_, _>>(),
            per_axis: BTreeMap::new(),
        }
    }

    fn model(pieces: &[(&str, u32)]) -> Artifact {
        let mut inventory = Vec::new();
        for (text, count) in pieces {
            for index in 0..*count {
                inventory.push(Piece::new(format!("{text}{index}"), Kind::Normal));
            }
        }
        Artifact {
            vocabulary: Vocabulary::new(inventory),
            trainer: Some(Trainer {
                model_type: ModelType::Bpe,
                vocab_size: 400,
                byte_fallback: true,
                split_digits: true,
                split_by_unicode_script: true,
                max_piece_length: 8,
                max_line_bytes: 4192,
                character_coverage: 0.9998,
                user_defined_symbols: Vec::new(),
            }),
            normalizer: None,
        }
    }

    #[test]
    fn a_starved_script_is_caught_without_encoding_anything() {
        // Devanagari is a fifth of the corpus and got almost no pieces: it can only fragment.
        let report = script_coverage(
            &corpus(&[("Latin", 800), ("Devanagari", 200)]),
            &model(&[("word", 99), ("क", 1)]),
        );

        assert_eq!(report.status, Status::Failed);
        assert_eq!(report.severity, Severity::Medium);
        assert_eq!(report.remedy, Remedy::FixCorpus);
        assert!(report.evidence[0].contains("Devanagari"));
    }

    #[test]
    fn a_proportionate_split_passes() {
        let report = script_coverage(
            &corpus(&[("Latin", 500), ("Han", 500)]),
            &model(&[("word", 50), ("漢", 50)]),
        );
        assert_eq!(report.status, Status::Passed);
    }

    #[test]
    fn a_trace_script_is_not_judged() {
        // 0.5% of the corpus getting few pieces is proportionate, not starvation.
        let report = script_coverage(
            &corpus(&[("Latin", 1000), ("Han", 5)]),
            &model(&[("word", 100)]),
        );
        assert_eq!(report.status, Status::Passed);
    }

    #[test]
    fn the_worst_starved_script_is_reported_first() {
        let report = script_coverage(
            &corpus(&[("Latin", 500), ("Han", 300), ("Devanagari", 200)]),
            &model(&[("word", 199), ("漢", 1)]),
        );

        assert_eq!(report.status, Status::Failed);
        assert!(
            report.evidence[0].contains("Devanagari"),
            "zero pieces beats one piece: {:?}",
            report.evidence
        );
    }

    #[test]
    fn an_empty_side_skips_rather_than_dividing_by_zero() {
        let report = script_coverage(&corpus(&[]), &model(&[("word", 10)]));
        assert_eq!(report.status, Status::Skipped);
        assert_eq!(report.severity, Severity::Medium);
    }

    #[test]
    fn the_line_limit_comes_from_the_model() {
        assert_eq!(line_limit(&model(&[("word", 1)])), Some(4192));
    }
}
