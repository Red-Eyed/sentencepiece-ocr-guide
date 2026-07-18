//! How the vocabulary budget was spent, per writing system.
//!
//! `fertility` measured a script's fragmentation by encoding real text and counting tokens per
//! character. That needs a tokenizer runtime. The same imbalance is visible in the inventory: a
//! script the trainer allocated almost no pieces to has learned no merges, so its text can only
//! come out as short pieces — which is what high fertility *is*.
//!
//! This is the weaker half of that signal and is reported rather than graded. What counts as a
//! fair share is not proportional: CJK needs thousands of pieces to reach the same fluency Latin
//! reaches with hundreds. Comparing the split against the corpus that produced it is the part
//! that can be judged, and that lives in [`crate::crosscheck`].

use std::collections::BTreeMap;

use crate::model::artifact::{Vocabulary, surface};
use crate::report::{Finding, MAX_EVIDENCE, Remedy, Severity};
use crate::writing::writing_of;

/// Pieces attributed to each writing system.
///
/// A piece counts once for every system it draws on, so a cross-script piece is counted under
/// both. That is deliberate: it is the same double-count `cross_script_pieces` reports as a
/// defect, and hiding it here would make the budget look tidier than the vocabulary is.
pub fn pieces_per_script(vocabulary: &Vocabulary) -> BTreeMap<&'static str, u64> {
    let mut per_script: BTreeMap<&'static str, u64> = BTreeMap::new();

    for piece in vocabulary.inspectable() {
        let text = surface(&piece.text);
        let mut seen: Vec<&'static str> = Vec::new();

        for writing in text.chars().filter_map(writing_of) {
            let name = writing.name();
            if !seen.contains(&name) {
                seen.push(name);
                *per_script.entry(name).or_default() += 1;
            }
        }
    }

    per_script
}

/// The vocabulary split, reported for the record.
pub fn vocabulary_budget(vocabulary: &Vocabulary) -> Finding {
    let per_script = pieces_per_script(vocabulary);
    let total: u64 = per_script.values().sum();

    if total == 0 {
        return Finding::skipped(
            "vocabulary_budget",
            "no pieces belong to any writing system",
        )
        .graded(Severity::Medium, Remedy::FixCorpus);
    }

    let mut ranked: Vec<(&str, u64)> = per_script.iter().map(|(n, c)| (*n, *c)).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let shares: Vec<String> = ranked
        .iter()
        .take(MAX_EVIDENCE)
        .map(|(name, count)| {
            let share = 100.0 * *count as f64 / total as f64;
            format!("{name}: {share:.1}% ({count} pieces)")
        })
        .collect();

    Finding::passed(
        "vocabulary_budget",
        format!("{} writing systems across the vocabulary", ranked.len()),
    )
    .with_evidence(shares)
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

    #[test]
    fn pieces_are_attributed_to_their_script() {
        let counted = pieces_per_script(&vocabulary(&[
            ("hello", Kind::Normal),
            ("world", Kind::Normal),
            ("漢字", Kind::Normal),
        ]));

        assert_eq!(counted.get("Latin"), Some(&2));
        assert_eq!(counted.get("Han"), Some(&1));
    }

    #[test]
    fn a_piece_counts_once_per_script_not_once_per_character() {
        let counted = pieces_per_script(&vocabulary(&[("hello", Kind::Normal)]));
        assert_eq!(counted.get("Latin"), Some(&1), "one piece, not five chars");
    }

    #[test]
    fn a_cross_script_piece_counts_under_both() {
        let counted = pieces_per_script(&vocabulary(&[("a漢", Kind::Normal)]));
        assert_eq!(counted.get("Latin"), Some(&1));
        assert_eq!(counted.get("Han"), Some(&1));
    }

    #[test]
    fn machinery_pieces_are_excluded() {
        let counted = pieces_per_script(&vocabulary(&[
            ("<0x41>", Kind::Byte),
            ("<s>", Kind::Control),
        ]));
        assert!(counted.is_empty(), "byte and control pieces are not text");
    }

    #[test]
    fn the_budget_is_reported_never_graded() {
        let finding = vocabulary_budget(&vocabulary(&[
            ("hello", Kind::Normal),
            ("漢字", Kind::Normal),
        ]));
        assert_eq!(finding.status, Status::Passed);
        assert_eq!(finding.evidence.len(), 2);
    }

    #[test]
    fn an_empty_vocabulary_skips_with_its_grade() {
        let finding = vocabulary_budget(&vocabulary(&[]));
        assert_eq!(finding.status, Status::Skipped);
        assert_eq!(finding.severity, Severity::Medium);
    }
}
