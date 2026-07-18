//! How the vocabulary budget was spent, per writing system.
//!
//! `fertility` measured a script's fragmentation by encoding real text and counting tokens per
//! character. That needs a tokenizer runtime. The same imbalance is visible in the inventory: a
//! script the trainer allocated almost no pieces to has learned no merges, so its text can only
//! come out as short pieces — which is what high fertility *is*.
//!
//! The split on its own is not a finding. "Latin holds half the vocabulary" is unactionable
//! without knowing what the corpus asked for, and once the corpus is present
//! [`crate::crosscheck::script_coverage`] answers the same question with a verdict. So this
//! module supplies the tally and states no opinion.

use std::collections::BTreeMap;

use crate::model::artifact::{Vocabulary, surface};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::artifact::{Kind, Piece};

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
}
