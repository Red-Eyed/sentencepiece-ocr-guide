//! Finding the text files under a corpus directory, and getting bytes out of them.
//!
//! A corpus directory is rarely only corpus: it collects trained `.model` and `.vocab`
//! artifacts, editor droppings, archives and checkpoints. Scanning those is at best noise and
//! at worst a report full of spurious findings from binary data that happened to decode.
//!
//! Detection uses the same heuristic as `git` and `grep` — a NUL byte in the leading chunk
//! means binary — via [`content_inspector`]. That beats an extension allowlist, because corpus
//! shards are frequently extensionless, and beats trusting the extension, because a `.txt` can
//! still be binary.
//!
//! A path named explicitly by the user is always accepted. An explicit path is a decision, and
//! second-guessing it would make the tool argue with its operator.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use memmap2::Mmap;

/// How much of a file to look at before deciding it is not text.
const SNIFF_BYTES: usize = 8192;

/// A file the scanner will read, and the root it was found under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    path: PathBuf,
    root: PathBuf,
}

impl Source {
    pub fn new(path: impl Into<PathBuf>, root: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            root: root.into(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Report name. The full path, because shard names repeat across subdirectories.
    pub fn label(&self) -> String {
        self.path.display().to_string()
    }

    /// Position within its root, so an output tree can mirror the input one.
    pub fn relative(&self) -> &Path {
        self.path.strip_prefix(&self.root).unwrap_or(&self.path)
    }

    /// Map the file for reading.
    ///
    /// Mapping rather than reading is what lets a file larger than memory be scanned at all:
    /// the kernel pages in what the scan touches and evicts what it has passed.
    pub fn map(&self) -> std::io::Result<Mapped> {
        let file = std::fs::File::open(&self.path)?;

        // Mapping a zero-length file is an error on most platforms, and an empty corpus shard
        // is ordinary rather than exceptional.
        if file.metadata()?.len() == 0 {
            return Ok(Mapped::Empty);
        }
        // SAFETY: the usual mmap caveat — another process truncating this file while the scan
        // is reading it is undefined behaviour. A corpus being rewritten underneath a scan is
        // already a lost cause, and the alternative is reading gigabytes into the heap.
        let mapping = unsafe { Mmap::map(&file)? };
        Ok(Mapped::File(mapping))
    }

    /// Size on disk, for sizing a progress bar without reading anything.
    pub fn size_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }
}

/// A mapped file, or nothing when the file was empty.
#[derive(Debug)]
pub enum Mapped {
    File(Mmap),
    Empty,
}

impl std::ops::Deref for Mapped {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Mapped::File(mapping) => mapping,
            Mapped::Empty => &[],
        }
    }
}

/// A path that was not scanned, and why. The reason travels to the caller rather than being
/// dropped, so a surprising file count is always explainable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    pub path: PathBuf,
    pub reason: &'static str,
}

#[derive(Debug, Default)]
pub struct Discovery {
    pub sources: Vec<Source>,
    pub skipped: Vec<Skipped>,
}

impl Discovery {
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Total bytes about to be read.
    pub fn total_bytes(&self) -> u64 {
        self.sources.iter().map(Source::size_bytes).sum()
    }

    /// One line naming what was passed over.
    pub fn summarize_skipped(&self, limit: usize) -> Option<String> {
        if self.skipped.is_empty() {
            return None;
        }
        let shown: Vec<String> = self
            .skipped
            .iter()
            .take(limit)
            .map(|s| {
                let name = s.path.file_name().unwrap_or(s.path.as_os_str());
                format!("{} ({})", name.to_string_lossy(), s.reason)
            })
            .collect();

        let mut note = format!("skipped {}: {}", self.skipped.len(), shown.join(", "));
        if self.skipped.len() > limit {
            note.push_str(&format!(", and {} more", self.skipped.len() - limit));
        }
        Some(note)
    }
}

/// Expand `roots` into scannable files, walking any directory recursively.
///
/// Results are sorted so a report over the same tree is byte-identical between runs, whatever
/// order the walk happened to produce.
pub fn discover(roots: &[PathBuf]) -> Discovery {
    let mut found = Discovery::default();

    for root in roots {
        if root.is_dir() {
            walk_directory(root, &mut found);
        } else if root.exists() {
            let parent = root.parent().unwrap_or(Path::new(".")).to_path_buf();
            found.sources.push(Source::new(root.clone(), parent));
        } else {
            found.skipped.push(Skipped {
                path: root.clone(),
                reason: "does not exist",
            });
        }
    }

    found.sources.sort_by(|a, b| a.path.cmp(&b.path));
    found.skipped.sort_by(|a, b| a.path.cmp(&b.path));
    found
}

fn walk_directory(root: &Path, found: &mut Discovery) {
    // `ignore` gives the hidden-file and `.gitignore` handling that a corpus directory living
    // inside a repository wants, and does the walk in parallel.
    let walker = WalkBuilder::new(root).hidden(true).git_ignore(true).build();

    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        match classify(entry.path()) {
            None => found.sources.push(Source::new(entry.path(), root)),
            Some(reason) => found.skipped.push(Skipped {
                path: entry.path().to_path_buf(),
                reason,
            }),
        }
    }
}

/// Why `path` should not be scanned, or `None` when it looks like text.
fn classify(path: &Path) -> Option<&'static str> {
    use std::io::Read;

    let Ok(mut file) = std::fs::File::open(path) else {
        return Some("unreadable");
    };
    let mut head = vec![0u8; SNIFF_BYTES];
    let Ok(read) = file.read(&mut head) else {
        return Some("unreadable");
    };
    head.truncate(read);

    match content_inspector::inspect(&head) {
        content_inspector::ContentType::BINARY => Some("binary"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn binary_files_are_skipped_with_a_reason() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "shard.txt", b"hello corpus\n");
        write(dir.path(), "trained.model", b"\x00\x01\x02binary");

        let found = discover(&[dir.path().to_path_buf()]);

        assert_eq!(found.sources.len(), 1);
        assert!(found.sources[0].path().ends_with("shard.txt"));
        assert_eq!(found.skipped.len(), 1);
        assert_eq!(found.skipped[0].reason, "binary");
    }

    #[test]
    fn an_explicitly_named_file_is_never_second_guessed() {
        // An explicit path is a decision; the tool must not argue with it.
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "weird.model", b"\x00\x01\x02binary");

        let found = discover(&[path]);
        assert_eq!(found.sources.len(), 1, "named directly, so scanned");
    }

    #[test]
    fn a_missing_path_is_reported_rather_than_ignored() {
        let found = discover(&[PathBuf::from("/nonexistent/corpus")]);
        assert!(found.is_empty());
        assert_eq!(found.skipped[0].reason, "does not exist");
    }

    #[test]
    fn an_empty_file_maps_to_no_bytes_rather_than_failing() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "empty.txt", b"");

        let source = Source::new(path, dir.path());
        let mapped = source.map().expect("an empty shard is ordinary");
        assert!(mapped.is_empty());
    }

    #[test]
    fn mapped_bytes_match_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "shard.txt", "café\nsecond\n".as_bytes());

        let source = Source::new(path, dir.path());
        assert_eq!(&*source.map().unwrap(), "café\nsecond\n".as_bytes());
    }

    #[test]
    fn discovery_is_sorted_so_reports_are_diffable() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["c.txt", "a.txt", "b.txt"] {
            write(dir.path(), name, b"text\n");
        }
        let found = discover(&[dir.path().to_path_buf()]);
        let names: Vec<String> = found
            .sources
            .iter()
            .map(|s| s.relative().display().to_string())
            .collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);
    }
}
