//! Measuring which axes actually vary, and in which source.
//!
//! You cannot canonicalize what you have not characterised: the exception list beyond NFC is a
//! judgement about *your* extractors, and guessing at it is how a soft hyphen that was on the
//! page gets stripped. This produces the evidence that judgement needs.
//!
//! The mapping from [`Action`] to severity is the whole ranking. `Collapse` variation is a
//! blocker — the label space is split before training starts. `Decide` variation is high: it
//! needs a call, and the counts are how you make it. `Preserve` variation is informational, and
//! a non-zero count there is expected rather than a defect.
//!
//! # Memory
//!
//! A file is mapped, not read, and split into byte ranges at line boundaries. Nothing here
//! holds a growing buffer, so peak memory is the mapping (virtual, paged on demand) plus one
//! small [`Counts`] per worker — independent of both file size and line length.
//!
//! This is the property worth stating explicitly, because the obvious alternative is a trap:
//! batching *lines* bounds the batch in the one dimension that does not control memory. A
//! corpus whose lines are long, or which contains a single newline-free multi-gigabyte file,
//! turns a "20,000 line" budget into an unbounded one. Byte ranges cannot express that bug.
//!
//! # Invalid UTF-8
//!
//! Bytes that are not valid UTF-8 are corrupt data, not an encoding preference: normalizing
//! cannot recover the intended text, so they are counted and reported rather than transformed.
//! Chunks are decoded with [`str::from_utf8`], and a chunk that fails is walked line by line so
//! one bad line does not discard the rest of the block.

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::corpus::axis::{Action, Axis};
use crate::corpus::source::Source;
use crate::report::{Finding, MAX_EVIDENCE, Remedy, Report, Severity};

/// Target size of one unit of parallel work.
///
/// Large enough that per-chunk overhead is negligible against the Unicode work inside it, small
/// enough that a worker's share stays in cache. Chunks are extended to the next line boundary,
/// so this is a floor rather than a cap — a single line longer than this is one chunk.
const CHUNK_BYTES: usize = 1 << 20;

/// What one unit of work observed. Merged commutatively, so completion order cannot affect the
/// report — a parallel run and a serial one produce identical output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    pub lines: u64,
    pub invalid_utf8: u64,
    /// Affected-line count per axis name. Ordered so merging and rendering are deterministic.
    pub per_axis: BTreeMap<&'static str, u64>,
}

impl Counts {
    fn merge(mut self, other: Counts) -> Counts {
        self.lines += other.lines;
        self.invalid_utf8 += other.invalid_utf8;
        for (axis, count) in other.per_axis {
            *self.per_axis.entry(axis).or_default() += count;
        }
        self
    }
}

/// Count one line against every axis.
pub fn count_line(line: &str, axes: &[Axis], counts: &mut Counts) {
    counts.lines += 1;

    // No axis fires on ASCII, so the whole axis loop is skipped for what is typically most of a
    // corpus. `Axis::apply` re-checks this; doing it here as well skips the dispatch entirely.
    if line.is_ascii() {
        return;
    }

    for axis in axes {
        if axis.affects(line) {
            *counts.per_axis.entry(axis.name).or_default() += 1;
        }
    }
}

/// Count a block of complete lines.
fn count_block(block: &[u8], axes: &[Axis]) -> Counts {
    let mut counts = Counts::default();

    match std::str::from_utf8(block) {
        // The overwhelmingly common case: the whole block decodes, so it is split once.
        Ok(text) => {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                count_line(line, axes, &mut counts);
            }
        }
        // Something in this block is not UTF-8. Fall back to per-line decoding so the damage is
        // attributed to the lines that carry it rather than to the whole block.
        Err(_) => {
            for line in block.split(|&b| b == b'\n') {
                match std::str::from_utf8(line) {
                    Ok(text) if text.trim().is_empty() => {}
                    Ok(text) => count_line(text, axes, &mut counts),
                    Err(_) => {
                        counts.lines += 1;
                        counts.invalid_utf8 += 1;
                    }
                }
            }
        }
    }
    counts
}

/// Split a buffer into chunks of at least `CHUNK_BYTES` that each end on a line boundary.
///
/// Returned as ranges rather than slices so the caller keeps ownership of the mapping and the
/// split can be computed without touching the bytes it describes.
fn line_aligned_chunks(data: &[u8]) -> Vec<std::ops::Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < data.len() {
        let target = (start + CHUNK_BYTES).min(data.len());
        // Extend to the end of the line the target landed in, so no chunk splits a line.
        let end = match memchr::memchr(b'\n', &data[target..]) {
            Some(offset) => target + offset + 1,
            None => data.len(),
        };
        chunks.push(start..end);
        start = end;
    }
    chunks
}

/// Scan one source, counting every axis in a single pass over the file.
pub fn scan_source(source: &Source, axes: &[Axis]) -> std::io::Result<Counts> {
    let data = source.map()?;
    let bytes: &[u8] = &data;

    // A short file is not worth splitting; the chunk list would cost more than the work.
    if bytes.len() <= CHUNK_BYTES {
        return Ok(count_block(bytes, axes));
    }

    Ok(line_aligned_chunks(bytes)
        .into_par_iter()
        .map(|range| count_block(&bytes[range], axes))
        .reduce(Counts::default, Counts::merge))
}

/// Per-source totals, in the order the sources were supplied.
pub type Totals = Vec<(String, Counts)>;

/// Turn scan totals into the report the checklist renders.
pub fn report(totals: &Totals, axes: &[Axis]) -> Report {
    if totals.is_empty() {
        return Report::new(vec![
            Finding::skipped("corpus_scan", "no sources supplied")
                .graded(Severity::Blocker, Remedy::FixCorpus),
        ]);
    }

    let scanned: u64 = totals.iter().map(|(_, c)| c.lines).sum();
    let mut findings = vec![invalid_utf8_finding(totals, scanned)];
    findings.extend(axes.iter().map(|axis| axis_finding(axis, totals, scanned)));
    Report::new(findings)
}

fn invalid_utf8_finding(totals: &Totals, scanned: u64) -> Finding {
    let affected: u64 = totals.iter().map(|(_, c)| c.invalid_utf8).sum();

    if affected == 0 {
        return Finding::passed(
            "invalid_utf8",
            format!("every one of {scanned} lines decoded as valid UTF-8"),
        );
    }

    Finding::failed(
        "invalid_utf8",
        format!(
            "{affected} of {scanned} lines contain bytes that are not valid UTF-8 — \
             fix the extractor; these bytes cannot be recovered by normalizing"
        ),
    )
    .with_evidence(evidence(totals, |c| c.invalid_utf8))
    .graded(Severity::Blocker, Remedy::FixCorpus)
}

fn axis_finding(axis: &Axis, totals: &Totals, scanned: u64) -> Finding {
    let affected: u64 = totals
        .iter()
        .map(|(_, c)| c.per_axis.get(axis.name).copied().unwrap_or(0))
        .sum();
    let check = format!("axis[{}]", axis.name);

    if affected == 0 {
        return Finding::passed(check, format!("no variation across {scanned} lines"));
    }

    // A `Preserve` axis reports a measurement, not a defect, so it never fails however large
    // the count is. Its severity would otherwise make expected text look broken.
    if axis.action == Action::Preserve {
        return Finding::passed(
            check,
            format!(
                "{affected} of {scanned} lines — {} (expected — preserve, do not fold)",
                axis.rationale
            ),
        );
    }

    let severity = match axis.action {
        Action::Collapse => Severity::Blocker,
        Action::Decide => Severity::High,
        Action::Preserve => unreachable!("handled above"),
    };

    Finding::failed(
        check,
        format!("{affected} of {scanned} lines — {}", axis.rationale),
    )
    .with_evidence(evidence(totals, |c| {
        c.per_axis.get(axis.name).copied().unwrap_or(0)
    }))
    .graded(severity, Remedy::FixCorpus)
}

/// Worst source first — variation is usually one broken extractor, not a diffuse issue.
///
/// Ties break on source name so a report stays diffable between runs: ranking on count alone
/// would let two equally-affected sources swap places, and a report that changes with `--jobs`
/// is one you cannot trust.
fn evidence(totals: &Totals, count: impl Fn(&Counts) -> u64) -> Vec<String> {
    let mut ranked: Vec<(&str, u64, u64)> = totals
        .iter()
        .map(|(name, counts)| (name.as_str(), count(counts), counts.lines))
        .filter(|(_, affected, _)| *affected > 0)
        .collect();

    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked
        .into_iter()
        .take(MAX_EVIDENCE)
        .map(|(name, affected, lines)| format!("{name}: {affected} / {lines} lines"))
        .collect()
}

/// Scan every source, in parallel across sources as well as within each one.
pub fn scan_all(sources: &[Source], axes: &[Axis]) -> Totals {
    sources
        .par_iter()
        .map(|source| {
            let counts = scan_source(source, axes).unwrap_or_default();
            (source.label(), counts)
        })
        .collect()
}

/// Where a scan reads its bytes from — the seam that keeps [`scan_source`] off the filesystem
/// in tests.
pub fn scan_bytes(bytes: &[u8], axes: &[Axis]) -> Counts {
    count_block(bytes, axes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::axis::default_axes;

    fn counts_for(text: &str) -> Counts {
        scan_bytes(text.as_bytes(), &default_axes())
    }

    #[test]
    fn blank_lines_are_not_counted() {
        assert_eq!(counts_for("alpha\n\n   \nbeta\n").lines, 2);
    }

    #[test]
    fn ascii_text_shows_no_variation() {
        let counts = counts_for("plain ascii line\nanother one\n");
        assert_eq!(counts.lines, 2);
        assert!(counts.per_axis.is_empty());
    }

    #[test]
    fn decomposed_text_is_attributed_to_the_nfc_axis() {
        let counts = counts_for("cafe\u{0301}\n");
        assert_eq!(counts.per_axis.get("nfc_composition"), Some(&1));
    }

    #[test]
    fn invalid_utf8_is_counted_per_line_not_per_block() {
        // One bad line must not discard the good ones sharing its block.
        let mut bytes = Vec::new();
        bytes.extend_from_slice("good line\n".as_bytes());
        bytes.extend_from_slice(b"caf\xe9 bad\n");
        bytes.extend_from_slice("another good\n".as_bytes());

        let counts = scan_bytes(&bytes, &default_axes());
        assert_eq!(counts.invalid_utf8, 1);
        assert_eq!(counts.lines, 3, "the good lines still counted");
    }

    #[test]
    fn chunks_never_split_a_line() {
        let data = b"aaaa\nbbbb\ncccc\n";
        for range in line_aligned_chunks(data) {
            let chunk = &data[range];
            assert!(
                chunk.is_empty() || chunk.ends_with(b"\n") || chunk.last() == data.last(),
                "chunk {chunk:?} does not end on a line boundary"
            );
        }
    }

    #[test]
    fn a_newline_free_file_is_a_single_chunk() {
        // The shape that OOMs a line-batching scanner: no line boundary to batch on.
        let data = vec![b'x'; CHUNK_BYTES * 3];
        assert_eq!(line_aligned_chunks(&data).len(), 1);
    }

    #[test]
    fn merging_counts_is_commutative() {
        let a = counts_for("cafe\u{0301}\n");
        let b = counts_for("a\u{00A0}b\n");
        assert_eq!(
            a.clone().merge(b.clone()),
            b.merge(a),
            "completion order must not change the report"
        );
    }

    #[test]
    fn preserve_axes_never_fail_the_run() {
        let totals: Totals = vec![("src".into(), counts_for("\u{2018}quoted\u{2019}\n"))];
        let axes = default_axes();
        let report = report(&totals, &axes);

        let punctuation = report
            .findings
            .iter()
            .find(|f| f.check == "axis[typographic_punctuation]")
            .expect("axis reported");
        assert!(
            !punctuation.is_failure(),
            "a preserve axis reports, it does not fail"
        );
    }

    #[test]
    fn collapse_variation_blocks() {
        let totals: Totals = vec![("src".into(), counts_for("cafe\u{0301}\n"))];
        let axes = default_axes();
        let report = report(&totals, &axes);

        let nfc = report
            .findings
            .iter()
            .find(|f| f.check == "axis[nfc_composition]")
            .expect("axis reported");
        assert!(nfc.is_failure());
        assert_eq!(nfc.severity, Severity::Blocker);
    }

    #[test]
    fn an_empty_corpus_is_skipped_not_passed() {
        let report = report(&Totals::new(), &default_axes());
        assert_eq!(report.count(crate::report::Status::Skipped), 1);
        assert!(report.ok(), "a skip is not a failure");
    }
}
