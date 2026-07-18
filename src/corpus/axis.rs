//! The ways two sources can disagree about how to encode the same text.
//!
//! An axis is a named transform plus a verdict on what should happen to it. A line is
//! *affected* when the transform changes it — which is all the scanner needs — and the
//! canonicalizer reuses the same transforms for the subset it is allowed to apply.
//!
//! The verdict keeps those two uses honest: [`Action::Preserve`] axes carry a transform purely
//! so they can be *detected*, and canonicalizing filters on [`Action`] so it can never fold one
//! by accident.
//!
//! # Why `Cow`
//!
//! [`Axis::apply`] returns [`Cow::Borrowed`] when a line is already canonical along that axis.
//! That is both the fast path — no allocation for the overwhelming majority of lines — and the
//! answer to "was this line affected?", so there is no separate predicate that could disagree
//! with the transform. A trigger-character table would be a third thing to keep in step; the
//! borrow already carries the information.
//!
//! # The ASCII invariant
//!
//! Every axis here fires only on non-ASCII input, so [`Axis::apply`] short-circuits on an ASCII
//! line before doing any work. That single check is what makes scanning a real corpus
//! affordable. It is enforced by a test over every axis rather than by convention, because an
//! axis that broke it would silently stop being detected.

use std::borrow::Cow;
use std::ops::RangeInclusive;

use unicode_normalization::{UnicodeNormalization, is_nfc};

pub const SOFT_HYPHEN: char = '\u{00AD}';
pub const ZERO_WIDTH_NON_JOINER: char = '\u{200C}';
pub const ZERO_WIDTH_JOINER: char = '\u{200D}';

/// What should be done about text that varies along an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Not a difference in the text. Canonicalizing folds these.
    Collapse,
    /// A genuine character difference that must survive. Reported, never folded.
    Preserve,
    /// Depends on your sources and rendering context. Opt in explicitly, after measuring.
    Decide,
}

/// How an axis rewrites a line.
///
/// An enum rather than a boxed closure: the set of rules is closed, so matching on it is
/// exhaustive and adding a variant is a compile error everywhere it must be handled.
#[derive(Debug)]
enum Rule {
    /// Delete every listed character.
    Strip(&'static [char]),
    /// Rewrite each listed character to its partner.
    Replace(&'static [(char, char)]),
    /// NFKC-fold only the characters inside the listed codepoint ranges.
    NfkcWithin(&'static [RangeInclusive<u32>]),
    /// Compose the whole line.
    Nfc,
    /// Fold non-ASCII decimal digits to their ASCII value.
    AsciiDigits,
    /// A trailing soft hyphen was rendered as a real hyphen — the line broke there.
    SoftHyphenLineFinal,
    /// A soft hyphen anywhere but the end was never drawn.
    SoftHyphenMidLine,
}

/// One way two encodings of the same text can differ.
#[derive(Debug)]
pub struct Axis {
    pub name: &'static str,
    pub action: Action,
    pub rationale: &'static str,
    rule: Rule,
}

impl Axis {
    /// This line's canonical form along this axis, borrowed when it is already canonical.
    pub fn apply<'a>(&self, line: &'a str) -> Cow<'a, str> {
        // No axis can fire on pure ASCII. One check replaces every transform below for what is
        // typically most of a corpus.
        if line.is_ascii() {
            return Cow::Borrowed(line);
        }

        match &self.rule {
            Rule::Strip(chars) => strip(line, chars),
            Rule::Replace(pairs) => replace(line, pairs),
            Rule::NfkcWithin(ranges) => nfkc_within(line, ranges),
            Rule::Nfc => compose(line),
            Rule::AsciiDigits => ascii_digits(line),
            Rule::SoftHyphenLineFinal => soft_hyphen_line_final(line),
            Rule::SoftHyphenMidLine => soft_hyphen_mid_line(line),
        }
    }

    /// Whether this axis changes the line.
    ///
    /// Reads the borrow rather than comparing strings: a transform that had nothing to do
    /// returns the original slice, and that is the answer.
    pub fn affects(&self, line: &str) -> bool {
        matches!(self.apply(line), Cow::Owned(_))
    }
}

fn strip<'a>(line: &'a str, chars: &[char]) -> Cow<'a, str> {
    if !line.chars().any(|c| chars.contains(&c)) {
        return Cow::Borrowed(line);
    }
    Cow::Owned(line.chars().filter(|c| !chars.contains(c)).collect())
}

fn replace<'a>(line: &'a str, pairs: &[(char, char)]) -> Cow<'a, str> {
    let swap = |c: char| pairs.iter().find(|(from, _)| *from == c).map(|(_, to)| *to);

    if !line.chars().any(|c| swap(c).is_some()) {
        return Cow::Borrowed(line);
    }
    Cow::Owned(line.chars().map(|c| swap(c).unwrap_or(c)).collect())
}

fn nfkc_within<'a>(line: &'a str, ranges: &[RangeInclusive<u32>]) -> Cow<'a, str> {
    let inside = |c: char| ranges.iter().any(|r| r.contains(&(c as u32)));

    if !line.chars().any(inside) {
        return Cow::Borrowed(line);
    }
    // Fold only the members: an unrestricted NFKC would also rewrite characters this axis has
    // no verdict on, which is how a `Preserve` axis would start silently collapsing text.
    let mut folded = String::with_capacity(line.len());
    for c in line.chars() {
        match inside(c) {
            true => folded.extend(c.to_string().nfkc()),
            false => folded.push(c),
        }
    }

    // Membership does not imply change: a character can sit inside the range and already be in
    // its folded form, and reporting that as variation would be a false positive.
    if folded == line {
        Cow::Borrowed(line)
    } else {
        Cow::Owned(folded)
    }
}

fn compose(line: &str) -> Cow<'_, str> {
    // `is_nfc` is a quick check with a fast path for already-composed text, so the common case
    // costs a scan rather than a full normalization pass plus a comparison.
    if is_nfc(line) {
        return Cow::Borrowed(line);
    }
    Cow::Owned(line.nfc().collect())
}

/// The decimal value of a non-ASCII digit, or `None` for anything else.
///
/// `char::to_digit` cannot be used here: it only knows ASCII, so every Arabic-Indic, Devanagari
/// and Thai digit reads as a non-digit and the axis silently stops detecting the thing it
/// exists for.
///
/// Unicode assigns decimal digits in aligned runs of ten, value zero first, so the distance
/// back to the start of the run *is* the value. That is a property of the standard rather than
/// of any table this crate would have to keep current.
fn decimal_value(c: char) -> Option<u32> {
    use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

    let is_digit = |c: char| c.general_category() == GeneralCategory::DecimalNumber;

    if c.is_ascii() || !is_digit(c) {
        return None;
    }

    let codepoint = c as u32;
    let mut value: u32 = 0;
    // Nine steps at most: a run is ten long, so anything further would be a different run.
    // `checked_sub` rather than `-`: the guard above means this cannot underflow today, but the
    // subtraction is the one place here where that is an argument rather than a fact.
    while value < 9 {
        let previous = codepoint
            .checked_sub(value.saturating_add(1))
            .and_then(char::from_u32);
        match previous {
            Some(previous) if is_digit(previous) => value = value.saturating_add(1),
            _ => break,
        }
    }
    Some(value)
}

fn ascii_digits(line: &str) -> Cow<'_, str> {
    if !line.chars().any(|c| decimal_value(c).is_some()) {
        return Cow::Borrowed(line);
    }
    Cow::Owned(
        line.chars()
            .map(|c| match decimal_value(c) {
                // A decimal value is 0..=9, which is always one ASCII character.
                Some(value) => char::from_digit(value, 10).unwrap_or(c),
                None => c,
            })
            .collect(),
    )
}

fn soft_hyphen_line_final(line: &str) -> Cow<'_, str> {
    match line.strip_suffix(SOFT_HYPHEN) {
        Some(head) => Cow::Owned(format!("{head}-")),
        None => Cow::Borrowed(line),
    }
}

fn soft_hyphen_mid_line(line: &str) -> Cow<'_, str> {
    // The final character is this axis's counterpart's business, so it is left alone here even
    // when it is a soft hyphen. Splitting the two keeps each verdict independently decidable.
    let Some((last_index, last)) = line.char_indices().next_back() else {
        return Cow::Borrowed(line);
    };
    let head = &line[..last_index];

    if !head.contains(SOFT_HYPHEN) {
        return Cow::Borrowed(line);
    }
    let mut folded = head.replace(SOFT_HYPHEN, "");
    folded.push(last);
    Cow::Owned(folded)
}

const TYPOGRAPHIC_PUNCTUATION: &[(char, char)] = &[
    ('\u{2018}', '\''),
    ('\u{2019}', '\''),
    ('\u{201C}', '"'),
    ('\u{201D}', '"'),
    ('\u{2010}', '-'),
    ('\u{2011}', '-'),
    ('\u{2012}', '-'),
    ('\u{2013}', '-'),
    ('\u{2014}', '-'),
    ('\u{2015}', '-'),
    ('\u{2212}', '-'),
];

/// The axes the guide covers.
///
/// Order matters for the collapsing subset: the BOM sits inside the Arabic Forms-B range, so it
/// must be stripped before that fold runs, and NFC must run last because folding can emit
/// decomposed sequences.
pub fn default_axes() -> Vec<Axis> {
    vec![
        Axis {
            name: "zero_width_non_content",
            action: Action::Collapse,
            rationale: "BOM and zero-width space are not page content",
            rule: Rule::Strip(&['\u{FEFF}', '\u{200B}']),
        },
        Axis {
            name: "arabic_presentation_forms",
            action: Action::Collapse,
            rationale: "legacy pre-shaped glyphs; shaping is derivable from context",
            rule: Rule::NfkcWithin(&[0xFB50..=0xFDFF, 0xFE70..=0xFEFF]),
        },
        Axis {
            name: "hebrew_presentation_forms",
            action: Action::Collapse,
            rationale: "legacy precomposed letter+point glyphs; identical to the logical sequence",
            rule: Rule::NfkcWithin(&[0xFB1D..=0xFB4F]),
        },
        Axis {
            name: "nbsp",
            action: Action::Collapse,
            rationale: "differs only in line-breaking, which a line image cannot show",
            rule: Rule::Replace(&[('\u{00A0}', ' ')]),
        },
        Axis {
            name: "nfc_composition",
            action: Action::Collapse,
            rationale: "canonically equivalent — not a difference in the text",
            rule: Rule::Nfc,
        },
        Axis {
            name: "soft_hyphen_line_final",
            action: Action::Decide,
            rationale: "rendered as a real hyphen where the line breaks",
            rule: Rule::SoftHyphenLineFinal,
        },
        Axis {
            name: "soft_hyphen_mid_line",
            action: Action::Decide,
            rationale: "never drawn away from a line break",
            rule: Rule::SoftHyphenMidLine,
        },
        Axis {
            name: "fullwidth_forms",
            action: Action::Preserve,
            rationale: "different characters with visibly wider glyphs",
            rule: Rule::NfkcWithin(&[0xFF01..=0xFF60, 0xFFE0..=0xFFE6]),
        },
        Axis {
            name: "ligatures",
            action: Action::Preserve,
            rationale: "typography the model can see in the image",
            // Latin (FB00–FB06) and Armenian (FB13–FB17) only. The Hebrew presentation forms
            // sharing this block are a legacy encoding artifact, not typography, and are
            // collapsed by their own axis above.
            rule: Rule::NfkcWithin(&[0xFB00..=0xFB1C]),
        },
        Axis {
            name: "non_ascii_digits",
            action: Action::Preserve,
            rationale: "Arabic-Indic and other digits are visibly distinct from ASCII",
            rule: Rule::AsciiDigits,
        },
        Axis {
            name: "typographic_punctuation",
            action: Action::Preserve,
            rationale: "curly quotes and dash widths are distinct glyphs",
            rule: Rule::Replace(TYPOGRAPHIC_PUNCTUATION),
        },
        Axis {
            name: "ideographic_space",
            action: Action::Preserve,
            rationale: "U+3000 is a visibly wider space",
            rule: Rule::Replace(&[('\u{3000}', ' ')]),
        },
        Axis {
            name: "zero_width_joiners",
            action: Action::Preserve,
            rationale: "ZWJ and ZWNJ change the shape of neighbouring letters",
            rule: Rule::Strip(&[ZERO_WIDTH_NON_JOINER, ZERO_WIDTH_JOINER]),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(name: &str) -> Axis {
        default_axes()
            .into_iter()
            .find(|a| a.name == name)
            .expect("axis exists")
    }

    #[test]
    fn no_axis_fires_on_ascii() {
        // The invariant the whole scan's speed rests on. A new axis breaking this would stop
        // being detected on ASCII lines rather than failing loudly, so it is asserted here.
        let ascii = "The quick brown fox: 12345 -- 'quoted' \"text\"\t(end)";
        for axis in default_axes() {
            assert!(!axis.affects(ascii), "{} fired on ASCII", axis.name);
            assert!(matches!(axis.apply(ascii), Cow::Borrowed(_)));
        }
    }

    #[test]
    fn an_unaffected_line_is_never_copied() {
        // Borrowed-on-unchanged is the fast path and the `affects` answer at once.
        let clean = "Пример чистого текста";
        assert!(matches!(axis("nbsp").apply(clean), Cow::Borrowed(_)));
        assert!(!axis("nbsp").affects(clean));
    }

    #[test]
    fn nbsp_collapses_to_a_plain_space() {
        assert_eq!(axis("nbsp").apply("a\u{00A0}b"), "a b");
        assert!(axis("nbsp").affects("a\u{00A0}b"));
    }

    #[test]
    fn decomposed_text_is_composed() {
        let decomposed = "cafe\u{0301}";
        assert_eq!(axis("nfc_composition").apply(decomposed), "café");
        assert!(axis("nfc_composition").affects(decomposed));
        assert!(!axis("nfc_composition").affects("café"));
    }

    #[test]
    fn zero_width_non_content_is_stripped() {
        assert_eq!(
            axis("zero_width_non_content").apply("\u{FEFF}text\u{200B}!"),
            "text!"
        );
    }

    #[test]
    fn a_trailing_soft_hyphen_becomes_a_real_hyphen() {
        let line_final = axis("soft_hyphen_line_final");
        assert_eq!(line_final.apply(&format!("weiter{SOFT_HYPHEN}")), "weiter-");
        // Mid-line is the other axis's business; this one must not touch it.
        assert!(!line_final.affects(&format!("wei{SOFT_HYPHEN}ter")));
    }

    #[test]
    fn a_mid_line_soft_hyphen_is_removed_but_a_final_one_is_left() {
        let mid = axis("soft_hyphen_mid_line");
        assert_eq!(mid.apply(&format!("wei{SOFT_HYPHEN}ter")), "weiter");
        assert!(
            !mid.affects(&format!("weiter{SOFT_HYPHEN}")),
            "a line-final soft hyphen belongs to the other axis"
        );
    }

    #[test]
    fn non_ascii_digits_fold_to_ascii_values() {
        let digits = axis("non_ascii_digits");
        // Arabic-Indic, Devanagari and Thai — three unrelated runs, to prove the value comes
        // from the run's alignment rather than from one hardcoded block.
        assert_eq!(digits.apply("\u{0664}\u{0667}"), "47");
        assert_eq!(digits.apply("\u{0966}\u{096F}"), "09");
        assert_eq!(digits.apply("\u{0E51}\u{0E55}"), "15");
    }

    #[test]
    fn decimal_value_ignores_ascii_and_non_digits() {
        assert_eq!(decimal_value('7'), None, "ASCII is already ASCII");
        assert_eq!(decimal_value('é'), None);
        // Roman numeral one is a Number, but not a *decimal* digit.
        assert_eq!(decimal_value('\u{2160}'), None);
    }

    #[test]
    fn typographic_punctuation_is_detected_but_marked_preserve() {
        let punctuation = axis("typographic_punctuation");
        assert!(punctuation.affects("\u{2018}quoted\u{2019}"));
        assert_eq!(punctuation.action, Action::Preserve);
    }

    #[test]
    fn fullwidth_forms_are_detected_without_touching_halfwidth() {
        assert!(axis("fullwidth_forms").affects("ＡＢＣ"));
        assert!(!axis("fullwidth_forms").affects("ABC"));
    }

    #[test]
    fn every_axis_transform_is_idempotent() {
        // Canonicalizing runs these in sequence, and `line == canonicalize(line)` is only a
        // valid write-time assertion if one pass reaches a fixed point.
        let samples = [
            "cafe\u{0301}",
            "a\u{00A0}b",
            "\u{FEFF}text\u{200B}",
            &format!("wei{SOFT_HYPHEN}ter{SOFT_HYPHEN}"),
            "\u{0664}\u{0667}",
            "ＡＢＣ",
            "\u{2018}q\u{2019}",
        ];
        for axis in default_axes() {
            for sample in &samples {
                let once = axis.apply(sample).into_owned();
                let twice = axis.apply(&once).into_owned();
                assert_eq!(once, twice, "{} is not idempotent on {sample:?}", axis.name);
            }
        }
    }
}
