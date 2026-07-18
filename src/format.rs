//! Turning numbers and text into something a reader can act on.
//!
//! Counts in a corpus report run to seven digits, and the question a reader actually has is never
//! "how many" on its own — it is "how much of it". Both forms are shown, because the count says
//! how much work a fix is and the share says whether it matters.

/// A count with thousands separators: `2104556` reads as `2,104,556`.
pub fn count(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);

    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// `part` of `whole` as a percentage, to one decimal place.
///
/// A zero denominator yields `0.0%` rather than a NaN: the caller has already established there
/// is nothing to report on, and a report is not the place to surface a division by zero.
pub fn percent(part: u64, whole: u64) -> String {
    if whole == 0 {
        return "0.0%".to_string();
    }
    format!("{:.1}%", 100.0 * part as f64 / whole as f64)
}

/// `part` of `whole` as `1,234 of 5,678 (21.7%)`.
pub fn ratio(part: u64, whole: u64) -> String {
    format!(
        "{} of {} ({})",
        count(part),
        count(whole),
        percent(part, whole)
    )
}

/// Text as it renders, followed by the code points that do not render distinctly.
///
/// The two forms are both necessary and neither is sufficient. Printing only the escape makes a
/// reader decode `\u{301}` by hand; printing only the glyph is worse, because the whole class of
/// defect this tool reports is text that *looks identical* and is not — `café` composed against
/// `café` decomposed is the same picture and a different token sequence.
///
/// Only non-ASCII characters get a code point, so an ordinary Latin piece stays readable.
pub fn literal(text: &str) -> String {
    let points: Vec<String> = text
        .chars()
        .filter(|c| !c.is_ascii())
        .map(|c| format!("U+{:04X}", c as u32))
        .collect();

    // Quoted by hand rather than with `{:?}`, which escapes exactly the characters this is here
    // to display: a combining acute comes out as `\u{301}` and the reader is back to decoding
    // escapes. The code points in brackets do that job without hiding the glyph.
    if points.is_empty() {
        return format!("\"{text}\"");
    }
    format!("\"{text}\" [{}]", points.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_get_thousands_separators() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1000), "1,000");
        assert_eq!(count(2104556), "2,104,556");
    }

    #[test]
    fn percentages_are_one_decimal() {
        assert_eq!(percent(1, 5), "20.0%");
        assert_eq!(percent(412883, 2104556), "19.6%");
    }

    #[test]
    fn a_zero_denominator_is_not_a_nan() {
        assert_eq!(percent(0, 0), "0.0%");
    }

    #[test]
    fn a_ratio_carries_both_the_count_and_the_share() {
        assert_eq!(ratio(1, 5), "1 of 5 (20.0%)");
        assert_eq!(ratio(412883, 2104556), "412,883 of 2,104,556 (19.6%)");
    }

    #[test]
    fn ascii_text_needs_no_code_points() {
        assert_eq!(literal("plain"), "\"plain\"");
    }

    #[test]
    fn non_ascii_text_carries_its_code_points() {
        // Written with escapes on both sides, because the two forms are indistinguishable in
        // source too — an expectation typed as a literal `é` silently tests the wrong string.
        assert_eq!(literal("caf\u{e9}"), "\"caf\u{e9}\" [U+00E9]");
        assert_eq!(literal("cafe\u{301}"), "\"cafe\u{301}\" [U+0301]");

        // Same picture, different text. This is the defect `nfc_vocabulary` reports, and the
        // reason the code points have to be printed alongside the glyphs.
        assert_ne!(literal("caf\u{e9}"), literal("cafe\u{301}"));
    }

    #[test]
    fn the_space_marker_is_shown_as_itself_and_as_a_code_point() {
        assert_eq!(literal("▁70"), "\"▁70\" [U+2581]");
    }
}
