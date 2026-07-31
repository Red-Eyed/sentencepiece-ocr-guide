use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

use crate::balance::{classify_line, shard_file_name, BucketKey};
use crate::config::EffectiveConfig;
use crate::corpus::{CorpusSourceIssue, CorpusSourceIssueId, DiscoveredCorpus};
use crate::normalize::canonicalize_line;
use crate::progress::ProgressReporter;

const FIXED_CORPUS_NAME: &str = "fixed_corpus.txt";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RepairSummary {
    pub fixed_corpus: PathBuf,
    pub issue_log: PathBuf,
    pub files_read: usize,
    pub lines_read: u64,
    pub lines_written: u64,
    pub lines_fixed: u64,
    pub lines_skipped: u64,
    pub source_issues: usize,
    pub shards: Vec<ShardSummary>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ShardSummary {
    pub bucket: BucketKey,
    pub path: PathBuf,
    pub lines: u64,
}

#[derive(Debug, Error)]
pub enum RepairError {
    #[error("failed to create directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to open {path}: {source}")]
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[derive(Debug, Serialize)]
struct CorpusIssue {
    id: IssueId,
    action: IssueAction,
    path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_number: Option<u64>,
    reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_text: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum IssueId {
    InvalidUtf8,
    Canonicalized,
    NonIdempotentCanonicalization,
    MaxSentenceLength,
    UnsupportedFile,
    ArchiveReadFailed,
    ArchiveDepthExceeded,
    ArchiveCycle,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum IssueAction {
    Fixed,
    Skipped,
}

pub fn repair_corpus(
    config: &EffectiveConfig,
    corpus: &DiscoveredCorpus,
    progress: &ProgressReporter,
) -> Result<RepairSummary, RepairError> {
    let paths = RepairPaths::from_config(config);
    paths.create_dirs()?;
    write_effective_config(config, &paths.effective_config)?;

    let output = open_writer(&paths.fixed_corpus)?;
    let issues = open_writer(&paths.issue_log)?;
    let mut writer = RepairWriter::new(output, issues, config, corpus.root.clone(), &paths);
    writer.write_source_issues(&corpus.issues)?;

    let stage = progress.stage("fixing corpus");
    for file in &corpus.files {
        writer.repair_file(file)?;
        stage.set_message(format!(
            "fixing corpus: {} line(s), {} fixed, {} skipped",
            writer.summary.lines_read, writer.summary.lines_fixed, writer.summary.lines_skipped
        ));
    }
    stage.finish(format!(
        "fixed corpus: {} written, {} fixed, {} skipped",
        writer.summary.lines_written, writer.summary.lines_fixed, writer.summary.lines_skipped
    ));

    let mut summary = writer.finish()?;
    summary.files_read = corpus.files.len();
    summary.fixed_corpus = paths.fixed_corpus;
    summary.issue_log = paths.issue_log;
    Ok(summary)
}

struct RepairPaths {
    fixed_corpus: PathBuf,
    issue_log: PathBuf,
    effective_config: PathBuf,
    shard_dir: PathBuf,
}

impl RepairPaths {
    fn from_config(config: &EffectiveConfig) -> Self {
        let fixed_corpus = config.output.work_dir.join(FIXED_CORPUS_NAME);
        let issue_log = resolve_output_path(&config.output.work_dir, &config.validation.issue_log);
        let effective_config = config.output.work_dir.join("effective_config.json");
        let shard_dir = config.output.work_dir.join("shards");

        Self {
            fixed_corpus,
            issue_log,
            effective_config,
            shard_dir,
        }
    }

    fn create_dirs(&self) -> Result<(), RepairError> {
        create_parent_dir(&self.fixed_corpus)?;
        create_parent_dir(&self.issue_log)?;
        create_parent_dir(&self.effective_config)?;
        fs::create_dir_all(&self.shard_dir).map_err(|source| RepairError::CreateDir {
            path: self.shard_dir.clone(),
            source,
        })
    }
}

struct RepairWriter<'a> {
    output: BufWriter<File>,
    issues: BufWriter<File>,
    config: &'a EffectiveConfig,
    corpus_root: PathBuf,
    summary: RepairSummary,
    shard_dir: PathBuf,
    shard_writers: HashMap<BucketKey, BufWriter<File>>,
}

impl<'a> RepairWriter<'a> {
    fn new(
        output: BufWriter<File>,
        issues: BufWriter<File>,
        config: &'a EffectiveConfig,
        corpus_root: PathBuf,
        paths: &RepairPaths,
    ) -> Self {
        Self {
            output,
            issues,
            config,
            corpus_root,
            summary: RepairSummary {
                fixed_corpus: paths.fixed_corpus.clone(),
                issue_log: paths.issue_log.clone(),
                files_read: 0,
                lines_read: 0,
                lines_written: 0,
                lines_fixed: 0,
                lines_skipped: 0,
                source_issues: 0,
                shards: Vec::new(),
            },
            shard_dir: paths.shard_dir.clone(),
            shard_writers: HashMap::new(),
        }
    }

    fn write_source_issues(&mut self, issues: &[CorpusSourceIssue]) -> Result<(), RepairError> {
        for issue in issues {
            self.summary.source_issues += 1;
            self.write_issue(CorpusIssue {
                id: IssueId::from_source(issue.id),
                action: IssueAction::Skipped,
                path: issue.path.clone(),
                line_number: None,
                reason: issue.reason.clone(),
                line_text: None,
            })?;
        }
        Ok(())
    }

    fn repair_file(&mut self, path: &Path) -> Result<(), RepairError> {
        let file = File::open(path).map_err(|source| RepairError::Open {
            path: path.to_path_buf(),
            source,
        })?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        let mut line_number = 0;

        loop {
            bytes.clear();
            let read =
                reader
                    .read_until(b'\n', &mut bytes)
                    .map_err(|source| RepairError::Read {
                        path: path.to_path_buf(),
                        source,
                    })?;
            if read == 0 {
                return Ok(());
            }

            line_number += 1;
            self.summary.lines_read += 1;
            trim_line_ending(&mut bytes);
            self.repair_line(path, line_number, &bytes)?;
        }
    }

    fn repair_line(
        &mut self,
        path: &Path,
        line_number: u64,
        bytes: &[u8],
    ) -> Result<(), RepairError> {
        let text = match std::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(error) => {
                self.skip(
                    path,
                    line_number,
                    IssueId::InvalidUtf8,
                    error.to_string(),
                    None,
                )?;
                return Ok(());
            }
        };

        let canonicalized = canonicalize_line(text, &self.config.canonicalization);
        if canonicalize_line(&canonicalized.text, &self.config.canonicalization).text
            != canonicalized.text
        {
            self.skip(
                path,
                line_number,
                IssueId::NonIdempotentCanonicalization,
                "line changed after a second canonicalization pass",
                maybe_line_text(text, self.config.validation.include_line_text_in_log),
            )?;
            return Ok(());
        }

        if canonicalized.text.len() > self.config.sentencepiece.max_sentence_length as usize {
            self.skip(
                path,
                line_number,
                IssueId::MaxSentenceLength,
                format!(
                    "canonicalized line is {} byte(s), over max_sentence_length {}",
                    canonicalized.text.len(),
                    self.config.sentencepiece.max_sentence_length
                ),
                maybe_line_text(text, self.config.validation.include_line_text_in_log),
            )?;
            return Ok(());
        }

        if canonicalized.changed {
            self.fix(
                path,
                line_number,
                "line changed under configured canonicalization",
                maybe_line_text(text, self.config.validation.include_line_text_in_log),
            )?;
        }

        self.write_line(path, &canonicalized.text)
    }

    fn fix(
        &mut self,
        path: &Path,
        line_number: u64,
        reason: impl Into<String>,
        line_text: Option<String>,
    ) -> Result<(), RepairError> {
        self.summary.lines_fixed += 1;
        self.write_issue(CorpusIssue {
            id: IssueId::Canonicalized,
            action: IssueAction::Fixed,
            path: path.to_path_buf(),
            line_number: Some(line_number),
            reason: reason.into(),
            line_text,
        })
    }

    fn skip(
        &mut self,
        path: &Path,
        line_number: u64,
        id: IssueId,
        reason: impl Into<String>,
        line_text: Option<String>,
    ) -> Result<(), RepairError> {
        self.summary.lines_skipped += 1;
        self.write_issue(CorpusIssue {
            id,
            action: IssueAction::Skipped,
            path: path.to_path_buf(),
            line_number: Some(line_number),
            reason: reason.into(),
            line_text,
        })
    }

    fn write_line(&mut self, source_path: &Path, line: &str) -> Result<(), RepairError> {
        self.output
            .write_all(line.as_bytes())
            .and_then(|_| self.output.write_all(b"\n"))
            .map_err(|source| RepairError::Write {
                path: self.summary.fixed_corpus.clone(),
                source,
            })?;
        self.summary.lines_written += 1;
        self.write_shard_line(source_path, line)
    }

    fn write_shard_line(&mut self, source_path: &Path, line: &str) -> Result<(), RepairError> {
        let bucket = classify_line(&classification_path(&self.corpus_root, source_path), line);
        let shard_path = self.shard_dir.join(shard_file_name(&bucket));
        if !self.shard_writers.contains_key(&bucket) {
            let writer = open_writer(&shard_path)?;
            self.shard_writers.insert(bucket.clone(), writer);
            self.summary.shards.push(ShardSummary {
                bucket: bucket.clone(),
                path: shard_path,
                lines: 0,
            });
        }

        let writer = self
            .shard_writers
            .get_mut(&bucket)
            .expect("shard writer was inserted");
        writer
            .write_all(line.as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
            .map_err(|source| RepairError::Write {
                path: self.shard_dir.join(shard_file_name(&bucket)),
                source,
            })?;

        let shard = self
            .summary
            .shards
            .iter_mut()
            .find(|shard| shard.bucket == bucket)
            .expect("shard summary was inserted");
        shard.lines += 1;
        Ok(())
    }

    fn write_issue(&mut self, issue: CorpusIssue) -> Result<(), RepairError> {
        serde_json::to_writer(&mut self.issues, &issue).map_err(|source| RepairError::Write {
            path: self.summary.issue_log.clone(),
            source: std::io::Error::other(source),
        })?;
        self.issues
            .write_all(b"\n")
            .map_err(|source| RepairError::Write {
                path: self.summary.issue_log.clone(),
                source,
            })
    }

    fn finish(mut self) -> Result<RepairSummary, RepairError> {
        self.output.flush().map_err(|source| RepairError::Write {
            path: self.summary.fixed_corpus.clone(),
            source,
        })?;
        self.issues.flush().map_err(|source| RepairError::Write {
            path: self.summary.issue_log.clone(),
            source,
        })?;
        for (bucket, mut writer) in self.shard_writers {
            writer.flush().map_err(|source| RepairError::Write {
                path: self.shard_dir.join(shard_file_name(&bucket)),
                source,
            })?;
        }
        self.summary
            .shards
            .sort_by(|left, right| left.bucket.cmp(&right.bucket));
        Ok(self.summary)
    }
}

impl IssueId {
    fn from_source(id: CorpusSourceIssueId) -> Self {
        match id {
            CorpusSourceIssueId::UnsupportedFile => Self::UnsupportedFile,
            CorpusSourceIssueId::ArchiveReadFailed => Self::ArchiveReadFailed,
            CorpusSourceIssueId::ArchiveDepthExceeded => Self::ArchiveDepthExceeded,
            CorpusSourceIssueId::ArchiveCycle => Self::ArchiveCycle,
        }
    }
}

fn trim_line_ending(bytes: &mut Vec<u8>) {
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
}

fn resolve_output_path(work_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        work_dir.join(path)
    }
}

fn classification_path(corpus_root: &Path, source_path: &Path) -> PathBuf {
    match source_path.strip_prefix(corpus_root) {
        Ok(relative) if !relative.as_os_str().is_empty() => relative.to_path_buf(),
        _ => source_path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| source_path.to_path_buf()),
    }
}

fn create_parent_dir(path: &Path) -> Result<(), RepairError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };

    fs::create_dir_all(parent).map_err(|source| RepairError::CreateDir {
        path: parent.to_path_buf(),
        source,
    })
}

fn open_writer(path: &Path) -> Result<BufWriter<File>, RepairError> {
    File::create(path)
        .map(BufWriter::new)
        .map_err(|source| RepairError::Open {
            path: path.to_path_buf(),
            source,
        })
}

fn write_effective_config(config: &EffectiveConfig, path: &Path) -> Result<(), RepairError> {
    let writer = open_writer(path)?;
    serde_json::to_writer_pretty(writer, config).map_err(|source| RepairError::Write {
        path: path.to_path_buf(),
        source: std::io::Error::other(source),
    })
}

fn maybe_line_text(text: &str, include: bool) -> Option<String> {
    include.then(|| text.to_owned())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use crate::config::{
        BalanceAxis, BalancingConfig, BalancingMode, CanonicalizationConfig, CorpusConfig,
        EffectiveConfig, LinePolicy, ModelType, NormalizationRuleName, OutputConfig,
        PythonTrainerConfig, SentencePieceConfig, SoftHyphenPolicy, StripRule, TrainerKind,
        UnicodeForm, ValidationConfig, ValidationMode,
    };
    use crate::corpus::{CorpusSourceIssue, CorpusSourceIssueId};

    use super::*;

    fn config(work_dir: &Path, corpus_path: &Path) -> EffectiveConfig {
        EffectiveConfig {
            preset: "ocr_multilingual".to_owned(),
            corpus: CorpusConfig {
                path: corpus_path.to_path_buf(),
            },
            output: OutputConfig {
                work_dir: work_dir.to_path_buf(),
                model_prefix: "ocr_tokenizer".to_owned(),
            },
            canonicalization: CanonicalizationConfig {
                unicode_form: UnicodeForm::Nfc,
                strip: vec![StripRule::Bom, StripRule::ZeroWidthSpace],
                map_nbsp_to_space: true,
                fold_arabic_presentation_forms: true,
                soft_hyphen: SoftHyphenPolicy::LineFinalToHyphenMidlineStrip,
                preserve_zwj_zwnj: true,
                preserve_compatibility_chars: true,
            },
            balancing: BalancingConfig {
                enabled: true,
                mode: BalancingMode::Conservative,
                total_lines: 20_000_000,
                alpha: 0.7,
                hierarchy: vec![
                    BalanceAxis::Domain,
                    BalanceAxis::Script,
                    BalanceAxis::LanguageHint,
                    BalanceAxis::SourceGroup,
                    BalanceAxis::LengthBin,
                ],
                min_keep_fraction: 0.5,
                max_downsample_ratio: 4.0,
                collapse_buckets_below_lines: 1000,
                max_part_lines: 1_000_000,
                shuffle_seed: 1337,
            },
            sentencepiece: SentencePieceConfig {
                trainer: TrainerKind::PythonSentencepiece,
                python: PythonTrainerConfig {
                    runner: "uv".to_owned(),
                    args: vec!["run".to_owned(), "python".to_owned()],
                    module: "spm_ocr_train_bridge".to_owned(),
                },
                model_type: ModelType::Bpe,
                vocab_size: 40_000,
                character_coverage: 0.9998,
                byte_fallback: true,
                normalization_rule_name: NormalizationRuleName::Identity,
                add_dummy_prefix: false,
                remove_extra_whitespaces: false,
                split_by_unicode_script: true,
                split_by_whitespace: true,
                split_digits: true,
                max_sentencepiece_length: 8,
                max_sentence_length: 16_384,
                input_sentence_size: 20_000_000,
                shuffle_input_sentence: true,
                train_extremely_large_corpus: true,
                user_defined_symbols: vec![],
                num_threads: 16,
            },
            validation: ValidationConfig {
                mode: ValidationMode::Report,
                line_policy: LinePolicy::FixOrSkip,
                issue_log: "reports/corpus_issues.jsonl".into(),
                include_line_text_in_log: false,
                round_trip_sample_per_bucket: 1000,
            },
        }
    }

    #[test]
    fn writes_fixed_corpus_and_logs_repairs() {
        let temp = TempDir::new().expect("temp dir");
        let input = temp.path().join("raw.txt");
        fs::write(&input, "cafe\u{0301}\u{00a0}\nplain\r\n").expect("write input");
        let output = temp.path().join("run");
        let config = config(&output, &input);
        let corpus = DiscoveredCorpus {
            root: input.clone(),
            files: vec![input],
            issues: vec![],
        };
        let progress = ProgressReporter::new(true);

        let summary = repair_corpus(&config, &corpus, &progress).expect("repair corpus");

        assert_eq!(summary.lines_read, 2);
        assert_eq!(summary.lines_written, 2);
        assert_eq!(summary.lines_fixed, 1);
        assert_eq!(summary.lines_skipped, 0);
        assert_eq!(
            fs::read_to_string(output.join(FIXED_CORPUS_NAME)).expect("read fixed corpus"),
            "café \nplain\n"
        );
        assert!(
            fs::read_to_string(output.join("reports/corpus_issues.jsonl"))
                .expect("read issue log")
                .contains("\"action\":\"fixed\"")
        );
    }

    #[test]
    fn skips_invalid_utf8_lines() {
        let temp = TempDir::new().expect("temp dir");
        let input = temp.path().join("raw.txt");
        fs::write(&input, [b'o', b'k', b'\n', 0xff, b'\n']).expect("write input");
        let output = temp.path().join("run");
        let config = config(&output, &input);
        let corpus = DiscoveredCorpus {
            root: input.clone(),
            files: vec![input],
            issues: vec![],
        };
        let progress = ProgressReporter::new(true);

        let summary = repair_corpus(&config, &corpus, &progress).expect("repair corpus");

        assert_eq!(summary.lines_read, 2);
        assert_eq!(summary.lines_written, 1);
        assert_eq!(summary.lines_skipped, 1);
        assert_eq!(
            fs::read_to_string(output.join(FIXED_CORPUS_NAME)).expect("read fixed corpus"),
            "ok\n"
        );
        assert!(
            fs::read_to_string(output.join("reports/corpus_issues.jsonl"))
                .expect("read issue log")
                .contains("\"id\":\"invalid_utf8\"")
        );
    }

    #[test]
    fn logs_source_level_discovery_issues() {
        let temp = TempDir::new().expect("temp dir");
        let output = temp.path().join("run");
        let config = config(&output, temp.path());
        let corpus = DiscoveredCorpus {
            root: temp.path().to_path_buf(),
            files: vec![],
            issues: vec![CorpusSourceIssue {
                id: CorpusSourceIssueId::UnsupportedFile,
                path: temp.path().join("binary"),
                reason: "not text".to_owned(),
            }],
        };
        let progress = ProgressReporter::new(true);

        let summary = repair_corpus(&config, &corpus, &progress).expect("repair corpus");

        assert_eq!(summary.source_issues, 1);
        let issue_log =
            fs::read_to_string(output.join("reports/corpus_issues.jsonl")).expect("read issue log");
        assert!(issue_log.contains("\"id\":\"unsupported_file\""));
        assert!(issue_log.contains("\"action\":\"skipped\""));
    }

    #[test]
    fn classification_path_uses_corpus_relative_source() {
        let root = Path::new("/corpus/vendor/es");
        let source = Path::new("/corpus/vendor/es/books/page.txt");

        assert_eq!(
            classification_path(root, source),
            PathBuf::from("books/page.txt")
        );
    }
}
