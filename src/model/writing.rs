//! Which writing systems a piece draws on.
//!
//! Script data comes from [`unicode_script`], which implements UAX #24 in full, rather than from
//! a tabulated block list — the same reason the corpus half takes its Unicode facts from the
//! standard.
//!
//! Digits are treated as a writing system of their own, which UAX #24 does not do: it assigns
//! ASCII and fullwidth digits to `Common`. That distinction is the point of the check, because a
//! digit fusing with a letter is exactly the cross-script merge the guide names.

use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};
use unicode_script::{Script, UnicodeScript};

/// A writing system a piece can draw characters from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Writing {
    /// Script-neutral digits — ASCII and fullwidth, which UAX #24 calls `Common`.
    Digit,
    Script(Script),
}

impl Writing {
    pub fn name(self) -> &'static str {
        match self {
            Writing::Digit => "Digit",
            Writing::Script(script) => script.full_name(),
        }
    }
}

/// The writing system a character belongs to, or `None` when it belongs to none.
///
/// Punctuation, whitespace and symbols are `Common`, and combining marks are `Inherited`. Both
/// appear legitimately inside pieces of every script, so counting them would make every ordinary
/// piece look like a cross-script merge.
fn writing_of(character: char) -> Option<Writing> {
    let script = character.script();

    // A digit carrying a real script — Devanagari's, say — is that script's text, and is left
    // to it. Only the script-neutral digits become `Digit`.
    if character.general_category() == GeneralCategory::DecimalNumber
        && matches!(script, Script::Common | Script::Inherited | Script::Unknown)
    {
        return Some(Writing::Digit);
    }

    match script {
        Script::Common | Script::Inherited | Script::Unknown => None,
        script => Some(Writing::Script(script)),
    }
}

/// Every writing system present in `text`, ordered by name so evidence is stable between runs.
pub fn writing_systems_in(text: &str) -> Vec<Writing> {
    let mut present: Vec<Writing> = Vec::new();

    for writing in text.chars().filter_map(writing_of) {
        if !present.contains(&writing) {
            present.push(writing);
        }
    }

    present.sort_unstable_by_key(|w| w.name());
    present
}

/// Whether every character is a decimal digit, and there is at least one.
///
/// The general category rather than `is_ascii_digit`, so a fullwidth or Devanagari numeral
/// counts — those merge into multi-digit pieces exactly as ASCII ones do.
pub fn is_all_digits(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.general_category() == GeneralCategory::DecimalNumber)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(text: &str) -> Vec<&'static str> {
        writing_systems_in(text)
            .into_iter()
            .map(Writing::name)
            .collect()
    }

    #[test]
    fn a_single_script_piece_reports_one_system() {
        assert_eq!(names("hello"), vec!["Latin"]);
        assert_eq!(names("漢字"), vec!["Han"]);
    }

    #[test]
    fn punctuation_and_marks_belong_to_no_writing_system() {
        // Otherwise every piece containing a hyphen would read as cross-script.
        assert_eq!(names("-.,!"), Vec::<&str>::new());
        assert_eq!(
            names("café"),
            vec!["Latin"],
            "a combining mark is Inherited"
        );
    }

    #[test]
    fn a_piece_spanning_two_scripts_reports_both() {
        assert_eq!(names("aあ"), vec!["Hiragana", "Latin"]);
    }

    #[test]
    fn script_neutral_digits_are_their_own_system() {
        assert_eq!(names("123"), vec!["Digit"]);
        assert_eq!(names("３"), vec!["Digit"], "fullwidth digits too");
        assert_eq!(names("3D"), vec!["Digit", "Latin"], "the documented merge");
    }

    #[test]
    fn a_digit_belonging_to_a_script_stays_with_it() {
        // U+0966 DEVANAGARI DIGIT ZERO is Devanagari text, not a neutral numeral.
        assert_eq!(names("\u{0966}"), vec!["Devanagari"]);
    }

    #[test]
    fn ordering_is_stable_regardless_of_character_order() {
        assert_eq!(names("aあ"), names("あa"));
    }

    #[test]
    fn digits_are_recognised_beyond_ascii() {
        assert!(is_all_digits("100"));
        assert!(is_all_digits("１００"), "fullwidth");
        assert!(!is_all_digits("10a"));
        assert!(!is_all_digits(""), "no digits is not all digits");
    }
}
