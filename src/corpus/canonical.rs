//! The single chokepoint every source passes through on the way into the corpus.
//!
//! Applies the [`Action::Collapse`] axes and nothing else. [`Action::Decide`] axes are opt-in,
//! because the right answer depends on how your extractor behaves and you should have measured
//! it first. [`Action::Preserve`] axes are structurally unreachable from here — the filter is
//! on the verdict, so no amount of misconfiguration can fold one.
//!
//! The result is idempotent, which is what turns `line == canonicalize(line)` from an
//! instruction into an enforced invariant at corpus-write time.

use std::borrow::Cow;

use crate::corpus::axis::{Action, Axis};

/// The axes a canonicalizing run will apply, in order.
#[derive(Debug)]
pub struct Canonicalizer {
    applied: Vec<Axis>,
}

/// A `Decide` axis was named that does not exist.
///
/// An error rather than a silent no-op: a typo here would quietly disable a transform the
/// operator believed was running, and they would find out from the corpus.
#[derive(Debug, thiserror::Error)]
#[error("unknown DECIDE axis: {name}. Available: {available}")]
pub struct UnknownAxis {
    pub name: String,
    pub available: String,
}

impl Canonicalizer {
    pub fn new(axes: Vec<Axis>, decide: &[String]) -> Result<Self, UnknownAxis> {
        let available = |axes: &[Axis]| {
            let mut names: Vec<&str> = axes
                .iter()
                .filter(|a| a.action == Action::Decide)
                .map(|a| a.name)
                .collect();
            names.sort_unstable();
            names.join(", ")
        };

        for name in decide {
            let known = axes
                .iter()
                .any(|a| a.action == Action::Decide && a.name == name);
            if !known {
                return Err(UnknownAxis {
                    name: name.clone(),
                    available: available(&axes),
                });
            }
        }

        let applied = axes
            .into_iter()
            .filter(|axis| match axis.action {
                Action::Collapse => true,
                Action::Decide => decide.iter().any(|n| n == axis.name),
                Action::Preserve => false,
            })
            .collect();

        Ok(Self { applied })
    }

    /// The canonical form of a line, borrowed when it is already canonical.
    pub fn apply<'a>(&self, line: &'a str) -> Cow<'a, str> {
        let mut current = Cow::Borrowed(line);

        for axis in &self.applied {
            // Re-borrowing each step keeps the whole chain allocation-free until some axis
            // actually changes something.
            current = match axis.apply(&current) {
                Cow::Borrowed(_) => current,
                Cow::Owned(changed) => Cow::Owned(changed),
            };
        }
        current
    }

    pub fn is_canonical(&self, line: &str) -> bool {
        matches!(self.apply(line), Cow::Borrowed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::axis::{SOFT_HYPHEN, default_axes};

    fn plain() -> Canonicalizer {
        Canonicalizer::new(default_axes(), &[]).unwrap()
    }

    #[test]
    fn collapse_axes_are_applied() {
        assert_eq!(plain().apply("cafe\u{0301}"), "café");
        assert_eq!(plain().apply("a\u{00A0}b"), "a b");
    }

    #[test]
    fn preserve_axes_are_structurally_unreachable() {
        // Curly quotes and fullwidth forms are real glyph differences; folding them would be
        // data loss, so no configuration may reach them.
        assert_eq!(
            plain().apply("\u{2018}quoted\u{2019}"),
            "\u{2018}quoted\u{2019}"
        );
        assert_eq!(plain().apply("ＡＢＣ"), "ＡＢＣ");
    }

    #[test]
    fn decide_axes_are_opt_in() {
        let line = format!("weiter{SOFT_HYPHEN}");
        assert_eq!(plain().apply(&line), line, "not opted in, so untouched");

        let opted = Canonicalizer::new(default_axes(), &["soft_hyphen_line_final".into()]).unwrap();
        assert_eq!(opted.apply(&line), "weiter-");
    }

    #[test]
    fn an_unknown_decide_axis_is_an_error_not_a_no_op() {
        let error = Canonicalizer::new(default_axes(), &["sofft_hyphen".into()]).unwrap_err();
        assert_eq!(error.name, "sofft_hyphen");
        assert!(error.available.contains("soft_hyphen_line_final"));
    }

    #[test]
    fn naming_a_preserve_axis_as_decide_is_rejected() {
        // It is a real axis, but not one anybody may opt into folding.
        let error = Canonicalizer::new(default_axes(), &["ligatures".into()]).unwrap_err();
        assert_eq!(error.name, "ligatures");
    }

    #[test]
    fn canonicalizing_is_idempotent() {
        // The property that makes `line == canonicalize(line)` a valid write-time assertion.
        let samples = [
            "cafe\u{0301}",
            "a\u{00A0}b",
            "\u{FEFF}text",
            "ＡＢＣ",
            "plain ascii",
        ];
        for sample in samples {
            let once = plain().apply(sample).into_owned();
            assert_eq!(plain().apply(&once), once, "not a fixed point: {sample:?}");
        }
    }

    #[test]
    fn already_canonical_text_is_never_copied() {
        assert!(matches!(plain().apply("plain ascii"), Cow::Borrowed(_)));
        assert!(plain().is_canonical("café"));
        assert!(!plain().is_canonical("cafe\u{0301}"));
    }
}
