//! Every failure mode the tool cites has to exist in the guide, and the guide has to number it
//! once.
//!
//! `docs/09-failure-modes.md` carries the numbers twice: a summary table near the top and the
//! detailed entries below it. They disagreed once — the table condensed twenty-one entries into
//! fifteen rows — and the cost was not cosmetic: code and comments cited whichever list the
//! author happened to read, so the same defect was called #11 in one place and #10 in another.
//!
//! These tests make that a build failure rather than a thing a reader has to notice.

#![allow(clippy::panic, clippy::expect_used)]

use std::collections::BTreeSet;
use std::path::PathBuf;

use spm_ocr::corpus::axis::default_axes;
use spm_ocr::corpus::balance::{self, LimitSource};
use spm_ocr::corpus::scan::{self, Counts};
use spm_ocr::crosscheck;
use spm_ocr::model::artifact;
use spm_ocr::model::suite::{self, Options};
use spm_ocr::report::{FailureMode, Report};

fn doc() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/09-failure-modes.md");
    std::fs::read_to_string(path).expect("the guide should be readable")
}

/// The numbers on the detailed entries, `**7. The decoder emits …**`.
fn documented_entries(text: &str) -> BTreeSet<u8> {
    text.lines()
        .filter_map(|line| line.strip_prefix("**"))
        .filter_map(|rest| rest.split_once('.'))
        .filter_map(|(number, _)| number.parse().ok())
        .collect()
}

/// The numbers in the summary table, `| 7 | … |`.
fn summary_rows(text: &str) -> BTreeSet<u8> {
    text.lines()
        .filter_map(|line| line.strip_prefix("| "))
        .filter_map(|rest| rest.split_once(" |"))
        .filter_map(|(number, _)| number.trim().parse().ok())
        .collect()
}

/// Every mode cited by a report.
fn cited(report: &Report) -> BTreeSet<u8> {
    report
        .findings
        .iter()
        .filter_map(|finding| finding.failure_mode)
        .map(|mode| mode.0)
        .collect()
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Every citation the tool can emit, from both halves and the cross-check.
fn every_citation() -> BTreeSet<u8> {
    let mut modes = BTreeSet::new();

    for name in ["guide_config.model", "stock_defaults.model"] {
        let parsed = artifact::load(&fixture(name)).expect("fixture should parse");
        modes.extend(cited(&suite::check(&parsed, &Options::default())));

        let corpus = Counts::default();
        modes.extend(cited(&crosscheck::check(&corpus, &parsed, 4192)));
    }

    // The corpus half: one finding per axis, plus the two balance findings.
    let axes = default_axes();
    let totals = vec![("shard.txt".to_string(), Counts::default())];
    modes.extend(cited(&scan::report(&totals, &axes)));

    let counts = Counts::default();
    modes.insert(13); // script_balance, cited at its call site in main
    modes.extend(cited(&Report::new(vec![
        balance::long_lines(&counts, 4192, LimitSource::SentencePieceDefault).about(15),
    ])));

    modes
}

#[test]
fn the_summary_table_and_the_detailed_entries_agree() {
    let text = doc();
    let entries = documented_entries(&text);
    let rows = summary_rows(&text);

    assert!(!entries.is_empty(), "no numbered entries found");
    assert_eq!(
        rows, entries,
        "the summary table and the detailed entries must number the same failures"
    );
}

#[test]
fn the_entries_are_numbered_without_gaps() {
    let entries = documented_entries(&doc());
    let expected: BTreeSet<u8> = (1..=entries.len() as u8).collect();

    assert_eq!(
        entries, expected,
        "failure modes should run 1..n with no gaps or repeats"
    );
}

#[test]
fn every_mode_the_tool_cites_exists_in_the_guide() {
    let documented = documented_entries(&doc());
    let cited = every_citation();

    assert!(!cited.is_empty(), "the tool cites nothing");

    let dangling: Vec<u8> = cited.difference(&documented).copied().collect();
    assert!(
        dangling.is_empty(),
        "cited but not in docs/09-failure-modes.md: {dangling:?}"
    );
}

#[test]
fn the_highest_mode_constant_matches_the_guide() {
    // The `about()` guard is only as good as this number.
    let documented = documented_entries(&doc());
    let highest = documented.iter().copied().max().unwrap_or(0);

    assert_eq!(
        highest,
        FailureMode::HIGHEST,
        "FailureMode::HIGHEST is out of step with the guide"
    );
}

#[test]
fn the_table_names_the_check_that_covers_each_mode() {
    // The "Checked by" column is what makes the guide navigable from a finding and back.
    //
    // Scoped to rows whose first cell is a number: the page carries other tables, and matching
    // every pipe-prefixed line swept in the mitigation table further down.
    let text = doc();
    let rows: Vec<&str> = text
        .lines()
        .filter(|line| {
            line.strip_prefix("| ")
                .and_then(|rest| rest.split_once(" |"))
                .is_some_and(|(first, _)| first.trim().parse::<u8>().is_ok())
        })
        .collect();

    assert_eq!(
        rows.len(),
        FailureMode::HIGHEST as usize,
        "expected one row per failure mode"
    );
    for row in rows {
        assert_eq!(
            row.matches('|').count(),
            6,
            "row is missing a column: {row}"
        );
    }
}
