use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use ignore::WalkBuilder;
use serde::Serialize;
use tar::Archive as TarArchive;
use thiserror::Error;
use xz2::read::XzDecoder;
use zip::ZipArchive;

const MAX_RECURSION_DEPTH: usize = 16;
const SNIFF_BYTES: usize = 8192;
const TAR_MAGIC_OFFSET: usize = 257;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredCorpus {
    pub root: PathBuf,
    pub files: Vec<PathBuf>,
    pub issues: Vec<CorpusSourceIssue>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CorpusSourceIssue {
    pub id: CorpusSourceIssueId,
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorpusSourceIssueId {
    UnsupportedFile,
    ArchiveReadFailed,
    ArchiveDepthExceeded,
    ArchiveCycle,
}

#[derive(Debug, Error)]
pub enum CorpusError {
    #[error("corpus path does not exist: {0}")]
    MissingPath(PathBuf),
    #[error("corpus path is neither a file nor a directory: {0}")]
    UnsupportedPath(PathBuf),
    #[error("failed to create unpack directory {path}: {source}")]
    CreateUnpackDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to inspect corpus path: {0}")]
    Walk(#[from] ignore::Error),
}

pub fn discover_text_files(
    corpus_path: &Path,
    unpack_dir: &Path,
) -> Result<DiscoveredCorpus, CorpusError> {
    if !corpus_path.exists() {
        return Err(CorpusError::MissingPath(corpus_path.to_path_buf()));
    }

    fs::create_dir_all(unpack_dir).map_err(|source| CorpusError::CreateUnpackDir {
        path: unpack_dir.to_path_buf(),
        source,
    })?;

    let mut discovery = Discovery::new(corpus_path.to_path_buf(), unpack_dir.to_path_buf());

    if corpus_path.is_file() {
        discovery.discover_file(corpus_path, 0);
        return Ok(discovery.finish());
    }

    if !corpus_path.is_dir() {
        return Err(CorpusError::UnsupportedPath(corpus_path.to_path_buf()));
    }

    let walker = WalkBuilder::new(corpus_path)
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
        {
            discovery.discover_file(entry.path(), 0);
        }
    }

    Ok(discovery.finish())
}

struct Discovery {
    root: PathBuf,
    unpack_dir: PathBuf,
    files: Vec<PathBuf>,
    issues: Vec<CorpusSourceIssue>,
    seen_containers: HashSet<blake3::Hash>,
}

impl Discovery {
    fn new(root: PathBuf, unpack_dir: PathBuf) -> Self {
        Self {
            root,
            unpack_dir,
            files: Vec::new(),
            issues: Vec::new(),
            seen_containers: HashSet::new(),
        }
    }

    fn finish(mut self) -> DiscoveredCorpus {
        self.files.sort();
        self.files.dedup();
        DiscoveredCorpus {
            root: self.root,
            files: self.files,
            issues: self.issues,
        }
    }

    fn discover_file(&mut self, path: &Path, depth: usize) {
        if depth > MAX_RECURSION_DEPTH {
            self.issue(
                CorpusSourceIssueId::ArchiveDepthExceeded,
                path,
                format!("archive nesting exceeded depth limit {MAX_RECURSION_DEPTH}"),
            );
            return;
        }

        let kind = match sniff_path(path) {
            Ok(kind) => kind,
            Err(error) => {
                self.issue(
                    CorpusSourceIssueId::UnsupportedFile,
                    path,
                    error.to_string(),
                );
                return;
            }
        };

        match kind {
            SourceKind::Text => self.files.push(path.to_path_buf()),
            SourceKind::Zip => self.unpack_zip(path, depth),
            SourceKind::Tar => self.unpack_tar(path, depth),
            SourceKind::Gzip => self.decompress(path, depth, Compression::Gzip),
            SourceKind::Bzip2 => self.decompress(path, depth, Compression::Bzip2),
            SourceKind::Xz => self.decompress(path, depth, Compression::Xz),
            SourceKind::Unsupported(reason) => {
                self.issue(CorpusSourceIssueId::UnsupportedFile, path, reason);
            }
        }
    }

    fn unpack_zip(&mut self, path: &Path, depth: usize) {
        if self.seen_container(path).is_some_and(|seen| seen) {
            self.issue(
                CorpusSourceIssueId::ArchiveCycle,
                path,
                "archive content was already seen",
            );
            return;
        }

        let result = (|| -> Result<(), ArchiveIoError> {
            let file = File::open(path)?;
            let mut archive = ZipArchive::new(file)?;
            for index in 0..archive.len() {
                let mut member = archive.by_index(index)?;
                if member.is_dir() {
                    continue;
                }

                let member_path = self.unpack_member_path(path, index, member.name());
                write_reader_to_path(&mut member, &member_path)?;
                self.discover_file(&member_path, depth + 1);
            }
            Ok(())
        })();

        if let Err(error) = result {
            self.issue(
                CorpusSourceIssueId::ArchiveReadFailed,
                path,
                error.to_string(),
            );
        }
    }

    fn unpack_tar(&mut self, path: &Path, depth: usize) {
        if self.seen_container(path).is_some_and(|seen| seen) {
            self.issue(
                CorpusSourceIssueId::ArchiveCycle,
                path,
                "archive content was already seen",
            );
            return;
        }

        let result = (|| -> Result<(), ArchiveIoError> {
            let file = File::open(path)?;
            let mut archive = TarArchive::new(file);
            for (index, entry) in archive.entries()?.enumerate() {
                let mut entry = entry?;
                if !entry.header().entry_type().is_file() {
                    continue;
                }

                let name = entry.path()?.display().to_string();
                let member_path = self.unpack_member_path(path, index, &name);
                write_reader_to_path(&mut entry, &member_path)?;
                self.discover_file(&member_path, depth + 1);
            }
            Ok(())
        })();

        if let Err(error) = result {
            self.issue(
                CorpusSourceIssueId::ArchiveReadFailed,
                path,
                error.to_string(),
            );
        }
    }

    fn decompress(&mut self, path: &Path, depth: usize, compression: Compression) {
        if self.seen_container(path).is_some_and(|seen| seen) {
            self.issue(
                CorpusSourceIssueId::ArchiveCycle,
                path,
                "compressed content was already seen",
            );
            return;
        }

        let result = (|| -> Result<PathBuf, ArchiveIoError> {
            let file = File::open(path)?;
            let output_path = self.unpack_member_path(path, 0, compression.output_label());
            match compression {
                Compression::Gzip => write_reader_to_path(&mut GzDecoder::new(file), &output_path)?,
                Compression::Bzip2 => {
                    write_reader_to_path(&mut BzDecoder::new(file), &output_path)?;
                }
                Compression::Xz => write_reader_to_path(&mut XzDecoder::new(file), &output_path)?,
            }
            Ok(output_path)
        })();

        match result {
            Ok(output_path) => self.discover_file(&output_path, depth + 1),
            Err(error) => {
                self.issue(
                    CorpusSourceIssueId::ArchiveReadFailed,
                    path,
                    error.to_string(),
                );
            }
        }
    }

    fn seen_container(&mut self, path: &Path) -> Option<bool> {
        let bytes = fs::read(path).ok()?;
        let hash = blake3::hash(&bytes);
        Some(!self.seen_containers.insert(hash))
    }

    fn unpack_member_path(&self, container: &Path, index: usize, member_name: &str) -> PathBuf {
        let mut hash_input = container.display().to_string();
        hash_input.push(':');
        hash_input.push_str(&index.to_string());
        hash_input.push(':');
        hash_input.push_str(member_name);
        let hash = blake3::hash(hash_input.as_bytes()).to_hex();
        self.unpack_dir.join(hash.as_str())
    }

    fn issue(&mut self, id: CorpusSourceIssueId, path: &Path, reason: impl Into<String>) {
        self.issues.push(CorpusSourceIssue {
            id,
            path: path.to_path_buf(),
            reason: reason.into(),
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SourceKind {
    Text,
    Zip,
    Tar,
    Gzip,
    Bzip2,
    Xz,
    Unsupported(String),
}

#[derive(Debug, Clone, Copy)]
enum Compression {
    Gzip,
    Bzip2,
    Xz,
}

impl Compression {
    fn output_label(self) -> &'static str {
        match self {
            Compression::Gzip => "gzip",
            Compression::Bzip2 => "bzip2",
            Compression::Xz => "xz",
        }
    }
}

#[derive(Debug, Error)]
enum ArchiveIoError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
}

fn sniff_path(path: &Path) -> Result<SourceKind, std::io::Error> {
    let mut file = File::open(path)?;
    let mut bytes = vec![0; SNIFF_BYTES];
    let read = file.read(&mut bytes)?;
    bytes.truncate(read);
    Ok(sniff_bytes(&bytes))
}

fn sniff_bytes(bytes: &[u8]) -> SourceKind {
    if bytes.is_empty() {
        return SourceKind::Unsupported("empty file".to_owned());
    }

    if let Some(kind) = infer_archive_kind(bytes) {
        return kind;
    }

    if is_tar(bytes) {
        return SourceKind::Tar;
    }

    if is_probably_utf8_text(bytes) {
        return SourceKind::Text;
    }

    SourceKind::Unsupported("not recognized as text, archive, or compressed text".to_owned())
}

fn infer_archive_kind(bytes: &[u8]) -> Option<SourceKind> {
    if let Some(kind) = infer::get(bytes) {
        match kind.mime_type() {
            "application/zip" => return Some(SourceKind::Zip),
            "application/gzip" | "application/x-gzip" => return Some(SourceKind::Gzip),
            "application/x-bzip2" => return Some(SourceKind::Bzip2),
            "application/x-xz" => return Some(SourceKind::Xz),
            _ => {}
        }
    }

    fallback_magic_kind(bytes)
}

fn fallback_magic_kind(bytes: &[u8]) -> Option<SourceKind> {
    if bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
    {
        return Some(SourceKind::Zip);
    }
    if bytes.starts_with(&[0x1f, 0x8b]) {
        return Some(SourceKind::Gzip);
    }
    if bytes.starts_with(b"BZh") {
        return Some(SourceKind::Bzip2);
    }
    if bytes.starts_with(&[0xfd, b'7', b'z', b'X', b'Z', 0x00]) {
        return Some(SourceKind::Xz);
    }
    None
}

fn is_tar(bytes: &[u8]) -> bool {
    bytes
        .get(TAR_MAGIC_OFFSET..TAR_MAGIC_OFFSET + 5)
        .is_some_and(|magic| magic == b"ustar")
}

fn is_probably_utf8_text(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    if text.contains('\0') {
        return false;
    }

    let chars = text.chars().count();
    if chars == 0 {
        return false;
    }

    let controls = text
        .chars()
        .filter(|char| char.is_control() && !matches!(char, '\n' | '\r' | '\t'))
        .count();
    controls * 20 <= chars
}

fn write_reader_to_path<R>(reader: &mut R, path: &Path) -> Result<(), std::io::Error>
where
    R: Read,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = File::create(path)?;
    std::io::copy(reader, &mut output)?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use flate2::write::GzEncoder;
    use flate2::Compression as GzipLevel;
    use tempfile::TempDir;
    use zip::write::SimpleFileOptions;

    use super::*;

    #[test]
    fn accepts_single_text_file_without_extension() {
        let temp = TempDir::new().expect("temp dir");
        let corpus_file = temp.path().join("corpus.data");
        fs::write(&corpus_file, "hello").expect("write corpus");

        let corpus =
            discover_text_files(&corpus_file, &temp.path().join("unpacked")).expect("discover");

        assert_eq!(corpus.files, vec![corpus_file]);
    }

    #[test]
    fn recursively_discovers_text_by_content_not_name() {
        let temp = TempDir::new().expect("temp dir");
        let nested = temp.path().join("nested");
        fs::create_dir(&nested).expect("create nested dir");

        let first = temp.path().join("a.bin");
        let second = nested.join("b.weird");
        fs::write(&first, "a").expect("write first");
        fs::write(&second, "b").expect("write second");
        fs::write(temp.path().join("binary"), [0, 159, 146, 150]).expect("write binary");

        let corpus = discover_text_files(temp.path(), &temp.path().join("unpacked"))
            .expect("discover corpus dir");

        assert_eq!(corpus.files, vec![first, second]);
        assert_eq!(corpus.issues.len(), 1);
    }

    #[test]
    fn unpacks_zip_by_magic_and_discovers_member_text() {
        let temp = TempDir::new().expect("temp dir");
        let archive_path = temp.path().join("corpus.noext");
        write_zip(&archive_path, "member.data", b"inside").expect("write zip");

        let corpus =
            discover_text_files(&archive_path, &temp.path().join("unpacked")).expect("discover");

        assert_eq!(corpus.files.len(), 1);
        assert_eq!(
            fs::read_to_string(&corpus.files[0]).expect("read member"),
            "inside"
        );
    }

    #[test]
    fn unpacks_archive_inside_archive() {
        let temp = TempDir::new().expect("temp dir");
        let inner_zip = temp.path().join("inner");
        write_zip(&inner_zip, "member", b"nested").expect("write inner zip");

        let archive_path = temp.path().join("outer");
        write_zip(
            &archive_path,
            "inner_payload",
            &fs::read(&inner_zip).expect("read inner"),
        )
        .expect("write outer zip");

        let corpus =
            discover_text_files(&archive_path, &temp.path().join("unpacked")).expect("discover");

        assert_eq!(corpus.files.len(), 1);
        assert_eq!(
            fs::read_to_string(&corpus.files[0]).expect("read member"),
            "nested"
        );
    }

    #[test]
    fn decompresses_gzip_text_by_magic() {
        let temp = TempDir::new().expect("temp dir");
        let compressed = temp.path().join("compressed");
        let mut encoder = GzEncoder::new(Vec::new(), GzipLevel::default());
        encoder.write_all(b"hello gzip").expect("write gzip");
        fs::write(&compressed, encoder.finish().expect("finish gzip")).expect("write compressed");

        let corpus =
            discover_text_files(&compressed, &temp.path().join("unpacked")).expect("discover");

        assert_eq!(corpus.files.len(), 1);
        assert_eq!(
            fs::read_to_string(&corpus.files[0]).expect("read text"),
            "hello gzip"
        );
    }

    fn write_zip(path: &Path, member: &str, bytes: &[u8]) -> zip::result::ZipResult<()> {
        let file = File::create(path)?;
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file(member, SimpleFileOptions::default())?;
        zip.write_all(bytes)?;
        zip.finish()?;
        Ok(())
    }
}
