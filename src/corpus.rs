use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredCorpus {
    pub root: PathBuf,
    pub files: Vec<PathBuf>,
}

#[derive(Debug, Error)]
pub enum CorpusError {
    #[error("corpus path does not exist: {0}")]
    MissingPath(PathBuf),
    #[error("corpus path is neither a file nor a directory: {0}")]
    UnsupportedPath(PathBuf),
    #[error("failed to inspect corpus path: {0}")]
    Walk(#[from] ignore::Error),
}

pub fn discover_text_files(path: &Path) -> Result<DiscoveredCorpus, CorpusError> {
    if !path.exists() {
        return Err(CorpusError::MissingPath(path.to_path_buf()));
    }

    if path.is_file() {
        return Ok(DiscoveredCorpus {
            root: path.to_path_buf(),
            files: vec![path.to_path_buf()],
        });
    }

    if !path.is_dir() {
        return Err(CorpusError::UnsupportedPath(path.to_path_buf()));
    }

    let mut files = Vec::new();
    let walker = WalkBuilder::new(path)
        .hidden(true)
        .parents(true)
        .ignore(true)
        .git_ignore(true)
        .git_exclude(true)
        .standard_filters(true)
        .build();

    for entry in walker {
        let entry = entry?;
        if entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
            && has_text_extension(entry.path())
        {
            files.push(entry.path().to_path_buf());
        }
    }

    files.sort();
    Ok(DiscoveredCorpus {
        root: path.to_path_buf(),
        files,
    })
}

fn has_text_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("txt"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn accepts_single_file_without_extension_filter() {
        let temp = TempDir::new().expect("temp dir");
        let corpus_file = temp.path().join("corpus.data");
        fs::write(&corpus_file, "hello").expect("write corpus");

        let corpus = discover_text_files(&corpus_file).expect("discover corpus file");

        assert_eq!(corpus.files, vec![corpus_file]);
    }

    #[test]
    fn recursively_discovers_txt_files_in_directory() {
        let temp = TempDir::new().expect("temp dir");
        let nested = temp.path().join("nested");
        fs::create_dir(&nested).expect("create nested dir");

        let first = temp.path().join("a.txt");
        let second = nested.join("b.TXT");
        fs::write(&first, "a").expect("write first");
        fs::write(&second, "b").expect("write second");
        fs::write(temp.path().join("ignore.md"), "ignored").expect("write ignored file");

        let corpus = discover_text_files(temp.path()).expect("discover corpus dir");

        assert_eq!(corpus.files, vec![first, second]);
    }
}
