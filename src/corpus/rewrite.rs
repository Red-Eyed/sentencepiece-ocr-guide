//! Writing a corpus back out in canonical form.
//!
//! [`super::canonical::Canonicalizer`] decides what a canonical line is; this module is the part
//! that gets a whole file through it and onto disk without loading it into memory and without
//! leaving a half-written shard behind.
//!
//! The core takes bytes and an [`std::io::Write`], so it is drivable with a `Vec<u8>` and needs
//! no filesystem to test. Only [`rewrite_file`] touches disk.

use std::io::Write;
use std::path::Path;

use crate::corpus::canonical::Canonicalizer;
use crate::corpus::source::Source;

/// What to do with a line that is not valid UTF-8.
///
/// An enum rather than a `bool` because the two answers are not "on and off" — refusing keeps
/// the corpus intact and stops, dropping silently loses data. The names say which is which at
/// every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnInvalidUtf8 {
    /// Stop and report. The default: a corpus is expensive to reassemble.
    #[default]
    Refuse,
    /// Skip the line and carry on, counting what was lost.
    Drop,
}

/// What a rewrite did to one source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tally {
    /// Non-blank lines seen, counted the way [`crate::corpus::scan`] counts them so the two
    /// halves of the tool report the same denominator.
    pub lines: u64,
    pub changed: u64,
    pub dropped: u64,
}

impl Tally {
    /// One line for an operator watching a run.
    pub fn summary(&self) -> String {
        let plural = if self.lines == 1 { "line" } else { "lines" };
        let mut summary = format!("{} {plural}, {} canonicalized", self.lines, self.changed);
        if self.dropped > 0 {
            summary.push_str(&format!(", {} dropped as invalid UTF-8", self.dropped));
        }
        summary
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RewriteError {
    #[error("{path}: line {line} is not valid UTF-8 (pass --drop-invalid to skip such lines)")]
    InvalidUtf8 { path: String, line: u64 },

    #[error("{path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Canonicalize `bytes` into `out`, one line at a time.
///
/// Line structure is preserved exactly: a file that does not end in a newline does not gain one,
/// and blank lines survive even though they are not counted.
pub fn canonicalize_into(
    bytes: &[u8],
    canon: &Canonicalizer,
    on_invalid: OnInvalidUtf8,
    out: &mut impl Write,
) -> Result<Tally, InvalidLine> {
    let mut tally = Tally::default();
    let mut segments = bytes.split(|&b| b == b'\n').peekable();
    let mut ordinal = 0u64;

    while let Some(segment) = segments.next() {
        ordinal += 1;
        // A trailing newline makes the final segment empty; it is a terminator, not a line.
        let is_last = segments.peek().is_none();

        let Ok(text) = std::str::from_utf8(segment) else {
            match on_invalid {
                OnInvalidUtf8::Refuse => return Err(InvalidLine { line: ordinal }),
                // Dropping takes the line's newline with it, or the gap would become a blank line.
                OnInvalidUtf8::Drop => {
                    tally.dropped += 1;
                    continue;
                }
            }
        };

        let canonical = canon.apply(text);
        if !text.trim().is_empty() {
            tally.lines += 1;
            if canonical != text {
                tally.changed += 1;
            }
        }

        out.write_all(canonical.as_bytes())
            .map_err(io_at(ordinal))?;
        if !is_last {
            out.write_all(b"\n").map_err(io_at(ordinal))?;
        }
    }

    Ok(tally)
}

/// A line that could not be decoded, or a write that failed while handling it. Carries the
/// ordinal so the operator can go look at it.
#[derive(Debug)]
pub struct InvalidLine {
    pub line: u64,
}

/// Writes into a `Vec<u8>` cannot fail, and writes into a file are re-reported against the path
/// by [`rewrite_file`], which is the only caller that knows one.
fn io_at(line: u64) -> impl Fn(std::io::Error) -> InvalidLine {
    move |_| InvalidLine { line }
}

/// Canonicalize `source` to `target`, replacing it only once the whole file is written.
///
/// The write goes to a temporary file alongside the target and is renamed over it at the end, so
/// an interrupted run leaves the original intact rather than a truncated shard. That is also what
/// makes `--in-place` safe while the input is mapped: the rename swaps the directory entry, and
/// the mapping this function is still reading from stays valid on the old inode.
pub fn rewrite_file(
    source: &Source,
    target: &Path,
    canon: &Canonicalizer,
    on_invalid: OnInvalidUtf8,
) -> Result<Tally, RewriteError> {
    let at = |error| RewriteError::Io {
        path: source.label(),
        source: error,
    };

    let mapped = source.map().map_err(at)?;
    let directory = target.parent().unwrap_or(Path::new("."));
    let mut scratch = tempfile::NamedTempFile::new_in(directory).map_err(at)?;

    let tally = {
        let mut writer = std::io::BufWriter::new(scratch.as_file_mut());
        let tally =
            canonicalize_into(&mapped, canon, on_invalid, &mut writer).map_err(|invalid| {
                RewriteError::InvalidUtf8 {
                    path: source.label(),
                    line: invalid.line,
                }
            })?;
        writer.flush().map_err(at)?;
        tally
    };

    scratch.persist(target).map_err(|error| RewriteError::Io {
        path: target.display().to_string(),
        source: error.error,
    })?;
    Ok(tally)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::axis::default_axes;

    fn plain() -> Canonicalizer {
        Canonicalizer::new(default_axes(), &[]).unwrap()
    }

    fn run(bytes: &[u8], on_invalid: OnInvalidUtf8) -> (Vec<u8>, Tally) {
        let mut out = Vec::new();
        let tally = canonicalize_into(bytes, &plain(), on_invalid, &mut out).unwrap();
        (out, tally)
    }

    #[test]
    fn collapse_axes_are_applied_to_the_bytes() {
        let (out, tally) = run("cafe\u{0301}\n".as_bytes(), OnInvalidUtf8::Refuse);
        assert_eq!(out, "café\n".as_bytes());
        assert_eq!(tally.changed, 1);
        assert_eq!(tally.lines, 1);
    }

    #[test]
    fn a_trailing_newline_is_neither_added_nor_removed() {
        let (with, _) = run(b"plain\n", OnInvalidUtf8::Refuse);
        assert_eq!(with, b"plain\n");

        let (without, _) = run(b"plain", OnInvalidUtf8::Refuse);
        assert_eq!(
            without, b"plain",
            "a file that lacked one must not gain one"
        );
    }

    #[test]
    fn blank_lines_survive_even_though_they_are_not_counted() {
        let (out, tally) = run(b"first\n\nsecond\n", OnInvalidUtf8::Refuse);
        assert_eq!(out, b"first\n\nsecond\n");
        assert_eq!(tally.lines, 2, "the blank line is structure, not content");
    }

    #[test]
    fn invalid_utf8_is_refused_by_default() {
        let mut out = Vec::new();
        let error = canonicalize_into(
            b"good\n\xff\xfe bad\ngood\n",
            &plain(),
            OnInvalidUtf8::Refuse,
            &mut out,
        )
        .unwrap_err();
        assert_eq!(error.line, 2, "the ordinal points at the offending line");
    }

    #[test]
    fn dropping_invalid_utf8_takes_the_line_and_its_newline() {
        let (out, tally) = run(b"good\n\xff\xfe bad\nalso good\n", OnInvalidUtf8::Drop);
        assert_eq!(out, b"good\nalso good\n", "no blank line left behind");
        assert_eq!(tally.dropped, 1);
        assert_eq!(tally.lines, 2);
    }

    #[test]
    fn rewriting_is_idempotent() {
        // The property the re-scan after a run is asserting: canonical output is a fixed point.
        let source = "cafe\u{0301}\na\u{00A0}b\n\u{FEFF}text\nＡＢＣ\n".as_bytes();
        let (once, _) = run(source, OnInvalidUtf8::Refuse);
        let (twice, tally) = run(&once, OnInvalidUtf8::Refuse);
        assert_eq!(once, twice);
        assert_eq!(tally.changed, 0, "already canonical, so nothing to do");
    }

    #[test]
    fn preserve_axes_are_not_folded_on_the_way_out() {
        let (out, _) = run(
            "ＡＢＣ \u{2018}q\u{2019}\n".as_bytes(),
            OnInvalidUtf8::Refuse,
        );
        assert_eq!(out, "ＡＢＣ \u{2018}q\u{2019}\n".as_bytes());
    }

    #[test]
    fn a_file_is_replaced_only_once_it_is_fully_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.txt");
        std::fs::write(&path, "cafe\u{0301}\n").unwrap();

        let source = Source::new(&path, dir.path());
        let tally = rewrite_file(&source, &path, &plain(), OnInvalidUtf8::Refuse).unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "café\n");
        assert_eq!(tally.changed, 1);
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            1,
            "the scratch file is gone, not left beside the shard"
        );
    }

    #[test]
    fn a_refused_file_is_left_exactly_as_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.txt");
        std::fs::write(&path, b"good\n\xff\xfe\n").unwrap();

        let source = Source::new(&path, dir.path());
        let error = rewrite_file(&source, &path, &plain(), OnInvalidUtf8::Refuse).unwrap_err();

        assert!(matches!(error, RewriteError::InvalidUtf8 { line: 2, .. }));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"good\n\xff\xfe\n",
            "the original survives a refusal"
        );
    }
}
