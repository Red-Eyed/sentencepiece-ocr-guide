use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::cancel::{CancellationToken, Cancelled};
use crate::config::{EffectiveConfig, SentencePieceConfig};
use crate::progress::ProgressReporter;

const TRAINER_REQUEST_NAME: &str = "trainer_request.json";
const TRAINER_OUTPUT_NAME: &str = "trainer_output.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainerRequest {
    pub sentencepiece: SentencePieceArgs,
    pub output: TrainerOutputPaths,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SentencePieceArgs {
    pub input: Vec<PathBuf>,
    pub model_prefix: PathBuf,
    pub model_type: String,
    pub vocab_size: u32,
    pub character_coverage: f64,
    pub byte_fallback: bool,
    pub normalization_rule_name: String,
    pub add_dummy_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub split_by_unicode_script: bool,
    pub split_by_whitespace: bool,
    pub split_digits: bool,
    pub max_sentencepiece_length: u32,
    pub max_sentence_length: u32,
    pub input_sentence_size: u64,
    pub shuffle_input_sentence: bool,
    pub train_extremely_large_corpus: bool,
    pub user_defined_symbols: Vec<String>,
    pub num_threads: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainerOutputPaths {
    pub model: PathBuf,
    pub vocab: PathBuf,
    pub trainer_output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainerOutput {
    pub status: TrainerStatus,
    pub model: PathBuf,
    pub vocab: PathBuf,
    pub stdout: Vec<String>,
    pub stderr: Vec<String>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainerStatus {
    Succeeded,
    Failed,
}

#[derive(Debug, Error)]
pub enum TrainerError {
    #[error("failed to create directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to spawn trainer `{program}`: {source}")]
    Spawn {
        program: String,
        source: std::io::Error,
    },
    #[error("failed to wait for trainer `{program}`: {source}")]
    Wait {
        program: String,
        source: std::io::Error,
    },
    #[error("failed to stop trainer `{program}`: {source}")]
    Stop {
        program: String,
        source: std::io::Error,
    },
    #[error("trainer failed with status {status}")]
    Failed { status: String },
    #[error("interrupted")]
    Interrupted(#[from] Cancelled),
}

pub fn train_sentencepiece(
    config: &EffectiveConfig,
    corpus_paths: &[PathBuf],
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<TrainerOutput, TrainerError> {
    let paths = TrainerPaths::from_config(config);
    paths.create_dirs()?;

    let request = build_trainer_request(config, corpus_paths, &paths);
    write_json(&paths.request, &request)?;

    let stage = progress.stage("training SentencePiece");
    let started = Instant::now();
    let mut child = Command::new(&config.sentencepiece.python.runner)
        .args(&config.sentencepiece.python.args)
        .arg("-m")
        .arg(&config.sentencepiece.python.module)
        .arg(&paths.request)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| TrainerError::Spawn {
            program: config.sentencepiece.python.runner.clone(),
            source,
        })?;

    let (sender, receiver) = mpsc::channel();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = capture_lines(stdout, StreamName::Stdout, sender.clone());
    let stderr_reader = capture_lines(stderr, StreamName::Stderr, sender);

    let mut captured = CapturedOutput::default();
    let exit_status = loop {
        if cancellation.is_cancelled() {
            stage.finish("training SentencePiece interrupted; stopping trainer");
            stop_child(&mut child, &config.sentencepiece.python.runner)?;
            drain_lines(&receiver, &mut captured);
            join_reader(stdout_reader);
            join_reader(stderr_reader);
            return Err(TrainerError::Interrupted(Cancelled));
        }

        drain_lines(&receiver, &mut captured);
        stage.set_message(format!(
            "training SentencePiece: {}s elapsed{}",
            started.elapsed().as_secs(),
            captured.latest_status()
        ));

        match child.try_wait().map_err(|source| TrainerError::Wait {
            program: config.sentencepiece.python.runner.clone(),
            source,
        })? {
            Some(status) => break status,
            None => thread::sleep(Duration::from_millis(250)),
        }
    };

    drain_lines(&receiver, &mut captured);
    join_reader(stdout_reader);
    join_reader(stderr_reader);
    drain_lines(&receiver, &mut captured);

    let elapsed_ms = started.elapsed().as_millis();
    let output = TrainerOutput {
        status: if exit_status.success() {
            TrainerStatus::Succeeded
        } else {
            TrainerStatus::Failed
        },
        model: paths.model.clone(),
        vocab: paths.vocab.clone(),
        stdout: captured.stdout,
        stderr: captured.stderr,
        elapsed_ms,
    };
    write_json(&paths.output, &output)?;

    if !exit_status.success() {
        stage.finish(format!("SentencePiece failed after {elapsed_ms}ms"));
        return Err(TrainerError::Failed {
            status: exit_status.to_string(),
        });
    }

    stage.finish(format!("SentencePiece trained in {elapsed_ms}ms"));
    Ok(output)
}

fn stop_child(child: &mut Child, program: &str) -> Result<(), TrainerError> {
    if child
        .try_wait()
        .map_err(|source| TrainerError::Wait {
            program: program.to_owned(),
            source,
        })?
        .is_some()
    {
        return Ok(());
    }

    child.kill().map_err(|source| TrainerError::Stop {
        program: program.to_owned(),
        source,
    })?;
    child.wait().map_err(|source| TrainerError::Wait {
        program: program.to_owned(),
        source,
    })?;
    Ok(())
}

pub fn build_trainer_request(
    config: &EffectiveConfig,
    corpus_paths: &[PathBuf],
    paths: &TrainerPaths,
) -> TrainerRequest {
    TrainerRequest {
        sentencepiece: SentencePieceArgs::from_config(config, corpus_paths, paths),
        output: TrainerOutputPaths {
            model: paths.model.clone(),
            vocab: paths.vocab.clone(),
            trainer_output: paths.output.clone(),
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainerPaths {
    pub request: PathBuf,
    pub output: PathBuf,
    pub model_prefix: PathBuf,
    pub model: PathBuf,
    pub vocab: PathBuf,
}

impl TrainerPaths {
    pub fn from_config(config: &EffectiveConfig) -> Self {
        let model_prefix = config.output.work_dir.join(&config.output.model_prefix);

        Self {
            request: config.output.work_dir.join(TRAINER_REQUEST_NAME),
            output: config.output.work_dir.join(TRAINER_OUTPUT_NAME),
            model: with_extension(&model_prefix, "model"),
            vocab: with_extension(&model_prefix, "vocab"),
            model_prefix,
        }
    }

    fn create_dirs(&self) -> Result<(), TrainerError> {
        create_parent_dir(&self.request)?;
        create_parent_dir(&self.output)?;
        create_parent_dir(&self.model_prefix)
    }
}

impl SentencePieceArgs {
    fn from_config(
        config: &EffectiveConfig,
        corpus_paths: &[PathBuf],
        paths: &TrainerPaths,
    ) -> Self {
        let sentencepiece = &config.sentencepiece;
        Self {
            input: corpus_paths.to_vec(),
            model_prefix: paths.model_prefix.clone(),
            model_type: model_type(sentencepiece),
            vocab_size: sentencepiece.vocab_size,
            character_coverage: sentencepiece.character_coverage,
            byte_fallback: sentencepiece.byte_fallback,
            normalization_rule_name: normalization_rule_name(sentencepiece),
            add_dummy_prefix: sentencepiece.add_dummy_prefix,
            remove_extra_whitespaces: sentencepiece.remove_extra_whitespaces,
            split_by_unicode_script: sentencepiece.split_by_unicode_script,
            split_by_whitespace: sentencepiece.split_by_whitespace,
            split_digits: sentencepiece.split_digits,
            max_sentencepiece_length: sentencepiece.max_sentencepiece_length,
            max_sentence_length: sentencepiece.max_sentence_length,
            input_sentence_size: sentencepiece.input_sentence_size,
            shuffle_input_sentence: sentencepiece.shuffle_input_sentence,
            train_extremely_large_corpus: sentencepiece.train_extremely_large_corpus,
            user_defined_symbols: sentencepiece.user_defined_symbols.clone(),
            num_threads: config.num_threads,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum StreamName {
    Stdout,
    Stderr,
}

#[derive(Debug)]
struct StreamLine {
    stream: StreamName,
    line: String,
}

#[derive(Debug, Default)]
struct CapturedOutput {
    stdout: Vec<String>,
    stderr: Vec<String>,
}

impl CapturedOutput {
    fn push(&mut self, line: StreamLine) {
        match line.stream {
            StreamName::Stdout => self.stdout.push(line.line),
            StreamName::Stderr => self.stderr.push(line.line),
        }
    }

    fn latest_status(&self) -> String {
        self.stderr
            .last()
            .or_else(|| self.stdout.last())
            .map(|line| format!(" - {line}"))
            .unwrap_or_default()
    }
}

fn capture_lines<T>(
    stream: Option<T>,
    stream_name: StreamName,
    sender: mpsc::Sender<StreamLine>,
) -> Option<thread::JoinHandle<()>>
where
    T: std::io::Read + Send + 'static,
{
    stream.map(|stream| {
        thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                if sender
                    .send(StreamLine {
                        stream: stream_name,
                        line,
                    })
                    .is_err()
                {
                    return;
                }
            }
        })
    })
}

fn drain_lines(receiver: &mpsc::Receiver<StreamLine>, captured: &mut CapturedOutput) {
    while let Ok(line) = receiver.try_recv() {
        captured.push(line);
    }
}

fn join_reader(reader: Option<thread::JoinHandle<()>>) {
    if let Some(reader) = reader {
        let _ = reader.join();
    }
}

fn write_json<T>(path: &Path, value: &T) -> Result<(), TrainerError>
where
    T: Serialize,
{
    let file = File::create(path).map_err(|source| TrainerError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::to_writer_pretty(BufWriter::new(file), value).map_err(|source| {
        TrainerError::Write {
            path: path.to_path_buf(),
            source: std::io::Error::other(source),
        }
    })
}

fn create_parent_dir(path: &Path) -> Result<(), TrainerError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };

    fs::create_dir_all(parent).map_err(|source| TrainerError::CreateDir {
        path: parent.to_path_buf(),
        source,
    })
}

fn with_extension(path: &Path, extension: &str) -> PathBuf {
    let mut path = path.to_path_buf();
    path.set_extension(extension);
    path
}

fn model_type(config: &SentencePieceConfig) -> String {
    serde_json::to_value(config.model_type)
        .expect("model type is serializable")
        .as_str()
        .expect("model type serializes as a string")
        .to_owned()
}

fn normalization_rule_name(config: &SentencePieceConfig) -> String {
    serde_json::to_value(config.normalization_rule_name)
        .expect("normalization rule name is serializable")
        .as_str()
        .expect("normalization rule name serializes as a string")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::config::{
        BalanceAxis, BalancingConfig, BalancingMode, CanonicalizationConfig, CorpusConfig,
        EffectiveConfig, LinePolicy, ModelType, NormalizationRuleName, OutputConfig,
        PythonTrainerConfig, SentencePieceConfig, SoftHyphenPolicy, StripRule, TrainerKind,
        UnicodeForm, ValidationConfig, ValidationMode,
    };

    use super::*;

    fn config(work_dir: &Path) -> EffectiveConfig {
        EffectiveConfig {
            preset: "ocr_multilingual".to_owned(),
            num_threads: 16,
            corpus: CorpusConfig { path: "raw".into() },
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
                user_defined_symbols: vec!["\\frac".to_owned()],
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
    fn builds_replayable_trainer_request() {
        let config = config(Path::new("runs/demo"));
        let paths = TrainerPaths::from_config(&config);

        let corpus_paths = vec![
            PathBuf::from("runs/demo/train_corpus/text-latin-en.txt"),
            PathBuf::from("runs/demo/train_corpus/text-cyrillic-uk.txt"),
        ];
        let request = build_trainer_request(&config, &corpus_paths, &paths);

        assert_eq!(request.sentencepiece.input, corpus_paths);
        assert_eq!(
            request.sentencepiece.model_prefix,
            PathBuf::from("runs/demo/ocr_tokenizer")
        );
        assert_eq!(request.sentencepiece.model_type, "bpe");
        assert_eq!(request.sentencepiece.normalization_rule_name, "identity");
        assert_eq!(request.sentencepiece.num_threads, 16);
        assert_eq!(
            request.output.model,
            PathBuf::from("runs/demo/ocr_tokenizer.model")
        );
        assert_eq!(
            request.output.vocab,
            PathBuf::from("runs/demo/ocr_tokenizer.vocab")
        );
    }
}
