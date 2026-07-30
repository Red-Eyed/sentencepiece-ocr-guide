//! `spm-ocr` — the command line edge.
//!
//! Everything app-specific lives here: where to read, what to print, what to exit with. The
//! library below it takes bytes and returns findings.
//!
//! Stdout carries the report and nothing else, so `--json` is always pipeable. Progress,
//! announcements and notes go to stderr.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json::json;

use spm_ocr::corpus::axis::{Action, default_axes};
use spm_ocr::corpus::balance;
use spm_ocr::corpus::canonical::Canonicalizer;
use spm_ocr::corpus::rewrite::{OnInvalidUtf8, Tally, rewrite_file};
use spm_ocr::corpus::scan;
use spm_ocr::corpus::source::{Discovery, Source, discover};
use spm_ocr::crosscheck;
use spm_ocr::format::count;
use spm_ocr::model::{artifact, pieces, suite};
use spm_ocr::render;
use spm_ocr::report::{Finding, Remedy, Report, Severity};
use spm_ocr::train::{self, PrepareOptions};

#[derive(Parser)]
#[command(name = "spm-ocr", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan corpus files for encoding axes that vary between sources.
    Corpus {
        /// Corpus files or directories (directories are walked recursively).
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        #[command(flatten)]
        output: OutputArgs,
    },

    /// Rewrite corpus files into canonical form, then re-scan to verify the result.
    Canonicalize {
        /// Corpus files or directories (directories are walked recursively).
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        #[command(flatten)]
        destination: DestinationArgs,

        /// DECIDE axes to apply, e.g. soft_hyphen_line_final. Measure with `corpus` first.
        #[arg(long)]
        decide: Vec<String>,

        /// Skip lines that are not valid UTF-8 instead of refusing. Loses data.
        #[arg(long)]
        drop_invalid: bool,

        #[command(flatten)]
        output: OutputArgs,
    },

    /// Check a trained tokenizer against the guide's checklist.
    Model {
        /// The trained `.model` file.
        model: PathBuf,

        #[command(flatten)]
        tuning: ModelArgs,

        #[command(flatten)]
        output: OutputArgs,
    },

    /// Both checklists at once, corpus findings first.
    All {
        /// The trained `.model` file.
        model: PathBuf,

        /// Corpus files or directories the model was trained on.
        #[arg(long, required = true)]
        corpus: Vec<PathBuf>,

        #[command(flatten)]
        tuning: ModelArgs,

        #[command(flatten)]
        output: OutputArgs,
    },

    /// Stream a canonical, balanced training sample into the official SentencePiece trainer.
    Train {
        /// Corpus files or directories (directories are walked recursively).
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        #[command(flatten)]
        args: TrainArgs,

        #[command(flatten)]
        output: OutputArgs,
    },
}

#[derive(clap::Args)]
struct ModelArgs {
    /// Longest digit-only piece to allow.
    #[arg(long, default_value_t = pieces::DEFAULT_MAX_DIGIT_PIECE_LENGTH)]
    max_digit_piece_length: usize,

    /// Allow a digit fusing with a letter, e.g. `3D`, instead of calling it a cross-script merge.
    #[arg(long)]
    allow_digit_letter_pieces: bool,
}

impl ModelArgs {
    fn options(&self) -> suite::Options {
        suite::Options {
            max_digit_piece_length: self.max_digit_piece_length,
            digits_are_a_script: !self.allow_digit_letter_pieces,
        }
    }
}

/// Where canonicalized output goes.
///
/// A clap group rather than two loose flags, so "both" and "neither" are rejected at parse time.
/// Overwriting the input is never the default: a corpus is expensive to reassemble, and a
/// canonicalizer configured with the wrong `--decide` axes is not obviously wrong afterwards.
#[derive(clap::Args)]
#[group(required = true, multiple = false)]
struct DestinationArgs {
    /// Directory to write canonicalized copies into, mirroring the input tree.
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,

    /// Overwrite the input files in place.
    #[arg(long)]
    in_place: bool,
}

enum Destination {
    Out(PathBuf),
    InPlace,
}

impl DestinationArgs {
    fn resolve(&self) -> Destination {
        match &self.out {
            Some(directory) => Destination::Out(directory.clone()),
            // The group above guarantees `--in-place` when `--out` is absent.
            None => Destination::InPlace,
        }
    }
}

impl Destination {
    /// Where `source` should be written, creating any directory it needs.
    fn target(&self, source: &Source) -> Result<PathBuf> {
        match self {
            Destination::InPlace => Ok(source.path().to_path_buf()),
            Destination::Out(directory) => {
                // Mirror the input tree, so a recursive run does not flatten shards together.
                let path = directory.join(source.relative());
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                Ok(path)
            }
        }
    }
}

#[derive(clap::Args)]
struct OutputArgs {
    /// Emit the report as JSON instead of text.
    #[arg(long)]
    json: bool,

    /// Exit non-zero when a failure reaches this severity.
    #[arg(long, value_enum, default_value_t = FailOn::High)]
    fail_on: FailOn,

    /// Worker threads. Defaults to the machine's parallelism.
    #[arg(long, short)]
    jobs: Option<usize>,
}

#[derive(clap::Args)]
struct TrainArgs {
    /// Output prefix. SentencePiece writes `<prefix>.model` and `<prefix>.vocab`.
    #[arg(long)]
    model_prefix: PathBuf,

    /// Target number of sampled training lines to stream.
    #[arg(long, default_value_t = 20_000_000)]
    lines: u64,

    /// Alpha for smoothing bucket shares: P'(bucket) ∝ P(bucket)^alpha.
    #[arg(long, default_value_t = 0.3)]
    alpha: f64,

    /// Deterministic seed for sampling and bounded shuffling.
    #[arg(long, default_value_t = 13)]
    seed: u64,

    /// SentencePiece vocabulary size.
    #[arg(long, default_value_t = 40_000)]
    vocab_size: u32,

    /// Character coverage for SentencePiece.
    #[arg(long, default_value_t = 0.9998)]
    character_coverage: f64,

    /// Maximum line length in bytes after canonicalization.
    #[arg(long, default_value_t = 16_384)]
    max_sentence_length: usize,

    /// Maximum learned piece length.
    #[arg(long, default_value_t = 8)]
    max_sentencepiece_length: usize,

    /// Lines held by Rust's bounded shuffle buffer.
    #[arg(long, default_value_t = 100_000)]
    shuffle_buffer_lines: usize,

    /// Hard upper bound for Rust-side shuffle memory, in GiB.
    #[arg(long, default_value_t = 32)]
    memory_budget_gb: u64,

    /// Directory for the temporary prepared training file. Defaults to the system temp dir.
    #[arg(long)]
    training_temp_dir: Option<PathBuf>,

    /// Keep the prepared SentencePiece input file instead of deleting it after training.
    #[arg(long)]
    keep_training_file: bool,

    /// Give math-like lines their own balancing bucket. Off by default for broad OCR.
    #[arg(long)]
    balance_math: bool,

    /// Maximum selected-line share for the math bucket when `--balance-math` is enabled.
    #[arg(long, default_value_t = 0.10)]
    math_max_share: f64,

    /// Worker threads for SentencePiece. Defaults to Rust worker count when set, otherwise 16.
    #[arg(long)]
    spm_threads: Option<usize>,

    /// SentencePiece trainer backend.
    #[arg(long, value_enum, default_value_t = TrainerBackend::UvPython)]
    trainer_backend: TrainerBackend,

    /// Path to the official `spm_train` binary, when using `--trainer-backend spm-train`.
    #[arg(long, default_value = "spm_train")]
    spm_train: PathBuf,

    /// DECIDE axes to apply while preparing training data.
    #[arg(long)]
    decide: Vec<String>,

    /// Skip lines that are not valid UTF-8 instead of refusing. Loses data.
    #[arg(long)]
    drop_invalid: bool,

    /// Skip lines above `--max-sentence-length` instead of refusing. Loses hard examples.
    #[arg(long)]
    drop_long_lines: bool,

    /// User-defined symbol to pass to SentencePiece. Repeat for multiple symbols.
    #[arg(long)]
    user_defined_symbol: Vec<String>,

    /// File containing one user-defined symbol per line.
    #[arg(long)]
    user_defined_symbols_file: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum TrainerBackend {
    /// Use `uv run python` and the project `sentencepiece` dependency.
    UvPython,
    /// Use an external `spm_train` executable.
    SpmTrain,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum FailOn {
    Blocker,
    High,
    Medium,
    Info,
}

impl From<FailOn> for Severity {
    fn from(value: FailOn) -> Self {
        match value {
            FailOn::Blocker => Severity::Blocker,
            FailOn::High => Severity::High,
            FailOn::Medium => Severity::Medium,
            FailOn::Info => Severity::Info,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Corpus { paths, output } => {
            configure_threads(output.jobs)?;
            let report = scan_corpus(&paths);
            emit(&report, &output);
            std::process::exit(report.exit_code(output.fail_on.into()));
        }

        Command::Canonicalize {
            paths,
            destination,
            decide,
            drop_invalid,
            output,
        } => {
            configure_threads(output.jobs)?;
            let report =
                canonicalize_corpus(&paths, &destination.resolve(), &decide, drop_invalid)?;
            emit(&report, &output);
            std::process::exit(report.exit_code(output.fail_on.into()));
        }

        Command::Model {
            model,
            tuning,
            output,
        } => {
            let report = check_model(&model, &tuning.options())?;
            emit(&report, &output);
            std::process::exit(report.exit_code(output.fail_on.into()));
        }

        Command::All {
            model,
            corpus,
            tuning,
            output,
        } => {
            configure_threads(output.jobs)?;
            let report = check_both(&model, &corpus, &tuning.options())?;

            emit(&report, &output);
            std::process::exit(report.exit_code(output.fail_on.into()));
        }

        Command::Train {
            paths,
            args,
            output,
        } => {
            configure_threads(output.jobs)?;
            let report = train_tokenizer(&paths, &args, output.jobs)?;
            emit(&report, &output);
            std::process::exit(report.exit_code(output.fail_on.into()));
        }
    }
}

fn check_model(path: &Path, options: &suite::Options) -> Result<Report> {
    step("reading the model…");
    let artifact = artifact::load(path)?;
    Ok(suite::check(&artifact, options))
}

/// Both checklists, plus the checks that need them together.
///
/// The model is read *first*, even though its findings are reported last. It records the
/// `max_sentence_length` the corpus must be measured against, and scanning before knowing it
/// would mean measuring against a default the trainer will not use.
fn check_both(model: &Path, corpus: &[PathBuf], options: &suite::Options) -> Result<Report> {
    step("reading the model…");
    let artifact = artifact::load(model)?;

    let axes = default_axes();
    let limit = crosscheck::line_limit(&artifact).unwrap_or(scan::DEFAULT_MAX_LINE_BYTES);
    let config = scan::Config::new(&axes).with_max_line_bytes(limit);

    // Corpus first in the report: several model defects originate in the data, and leading with
    // the model would invite a retrain which reproduces the defect exactly.
    let (mut report, combined) = scan_corpus_with(corpus, &config);
    report.extend(suite::check(&artifact, options));
    report.extend(crosscheck::check(&combined, &artifact, limit));

    Ok(report)
}

/// Size rayon's pool once, up front. Left to rayon's own default when unset — its work-stealing
/// handles a heterogeneous machine better than a core-count heuristic can.
fn configure_threads(jobs: Option<usize>) -> Result<()> {
    if let Some(jobs) = jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs.max(1))
            .build_global()?;
    }
    Ok(())
}

/// Scan a corpus, returning the report and the combined counts the cross-check needs.
///
/// `long_lines` is deliberately not added here: on its own the scan can only measure against
/// SentencePiece's default, while `all` measures against the model's actual limit. Whichever
/// caller knows the better number emits the finding, so it is never reported twice.
fn scan_corpus_with(paths: &[PathBuf], config: &scan::Config) -> (Report, scan::Counts) {
    step("discovering files…");
    let found = discover(paths);

    if let Some(note) = found.summarize_skipped(3) {
        note_line(&note);
    }

    let totals = scan_with_progress(&found, config);
    let combined = scan::combined(&totals);

    let mut report = scan::report(&totals, config.axes);
    report
        .findings
        .push(balance::script_balance(&combined).about(13));
    // Per source rather than combined: which extractor disagrees is the actionable part.
    report
        .findings
        .push(balance::normalization_forms(&totals).about(4));

    (report, combined)
}

/// The `corpus` subcommand: no model, so the line limit is SentencePiece's default.
fn scan_corpus(paths: &[PathBuf]) -> Report {
    let axes = default_axes();
    let (limit, source) = balance::default_limit();
    let config = scan::Config::new(&axes).with_max_line_bytes(limit);

    let (mut report, combined) = scan_corpus_with(paths, &config);
    report
        .findings
        .push(balance::long_lines(&combined, limit, source).about(15));
    report
}

/// Rewrite every source, then scan what was written.
///
/// The re-scan is the point rather than a courtesy: canonicalizing is only established as having
/// worked if the output is observed to be clean, and a wrong `--decide` set is not otherwise
/// visible afterwards.
fn canonicalize_corpus(
    paths: &[PathBuf],
    destination: &Destination,
    decide: &[String],
    drop_invalid: bool,
) -> Result<Report> {
    step("discovering files…");
    let found = discover(paths);

    if let Some(note) = found.summarize_skipped(3) {
        note_line(&note);
    }

    let canonicalizer = Canonicalizer::new(default_axes(), decide)?;
    let on_invalid = if drop_invalid {
        OnInvalidUtf8::Drop
    } else {
        OnInvalidUtf8::Refuse
    };

    let written = rewrite_with_progress(&found, destination, &canonicalizer, on_invalid)?;

    let axes = default_axes();
    let config = scan::Config::new(&axes);
    let totals = scan_with_progress(&discover(&written), &config);
    Ok(scan::report(&totals, &axes))
}

fn train_tokenizer(
    paths: &[PathBuf],
    args: &TrainArgs,
    rust_jobs: Option<usize>,
) -> Result<Report> {
    step("discovering files…");
    let found = discover(paths);
    if found.is_empty() {
        bail!("no training sources found");
    }
    if let Some(note) = found.summarize_skipped(3) {
        note_line(&note);
    }

    let canonicalizer = Canonicalizer::new(default_axes(), &args.decide)?;
    let options = prepare_options(args)?;

    let axes = default_axes();
    let scan_config = scan::Config::new(&axes).with_max_line_bytes(args.max_sentence_length);
    let totals = scan_with_progress(&found, &scan_config);
    let combined = scan::combined(&totals);
    let mut preflight = scan::report(&totals, &axes);
    preflight
        .findings
        .push(balance::script_balance(&combined).about(13));
    preflight
        .findings
        .push(balance::normalization_forms(&totals).about(4));
    preflight.findings.push(
        balance::long_lines(
            &combined,
            args.max_sentence_length,
            balance::LimitSource::Model,
        )
        .about(15),
    );
    let unresolved = unresolved_preflight_failures(&preflight, args);
    let mut report = report_preflight_for_training(preflight, args);

    if unresolved > 0 {
        report.findings.push(
            Finding::skipped(
                "training_stream",
                "preflight corpus checks failed; fix the corpus before spending trainer compute",
            )
            .graded(Severity::High, Remedy::FixCorpus),
        );
        return Ok(report);
    }

    step("counting training buckets…");
    let plan = train::plan_corpus(&found.sources, &canonicalizer, &options)?;
    report.extend(training_report(&plan, args, &options));
    note_line(&format!(
        "eligible lines: {}, streaming: {}",
        count(plan.eligible_lines),
        count(plan.selected_lines),
    ));
    for (bucket, quota) in &plan.bucket_quotas {
        let available = plan.bucket_counts.get(bucket).copied().unwrap_or(0);
        note_line(&format!(
            "{}: {} of {} lines",
            bucket.label(),
            count(*quota),
            count(available),
        ));
    }

    let user_symbols = user_defined_symbols(args)?;
    let settings = trainer_settings(
        Path::new("<fifo>"),
        args,
        &options,
        &user_symbols,
        rust_jobs,
    );
    report
        .findings
        .push(Finding::passed("spm_train_args", trainer_summary(args, &settings)).about(6));

    run_sentencepiece(
        &found.sources,
        &plan,
        &canonicalizer,
        &options,
        args,
        &user_symbols,
        rust_jobs,
    )?;

    step("reading the trained model…");
    let artifact = artifact::load(&sentencepiece_model_path(&args.model_prefix))?;
    report.extend(suite::check(&artifact, &suite::Options::default()));
    report
        .findings
        .push(crosscheck::script_coverage(&combined, &artifact).about(13));

    Ok(report)
}

fn training_report(plan: &train::CorpusPlan, args: &TrainArgs, options: &PrepareOptions) -> Report {
    let quota_evidence = plan.bucket_quotas.iter().map(|(bucket, quota)| {
        let available = plan.bucket_counts.get(bucket).copied().unwrap_or(0);
        format!(
            "{}: {} selected of {} eligible",
            bucket.label(),
            count(*quota),
            count(available),
        )
    });

    Report::new(vec![
        Finding::passed(
            "training_buckets",
            format!(
                "{} eligible lines across {} buckets; {} selected after alpha={}; math lines: {} ({})",
                count(plan.eligible_lines),
                plan.bucket_counts.len(),
                count(plan.selected_lines),
                args.alpha,
                count(plan.math_lines),
                math_policy_summary(args),
            ),
        )
        .with_evidence(quota_evidence)
        .about(13),
        Finding::passed(
            "training_stream",
            format!(
                "bounded shuffle buffer: {} lines, max line: {} bytes, Rust memory budget: {} bytes; prepared input file is {}",
                count(options.shuffle_buffer_lines as u64),
                options.max_line_bytes,
                options.memory_budget_bytes,
                training_file_policy(args),
            ),
        ),
    ])
}

fn training_file_policy(args: &TrainArgs) -> &'static str {
    if args.keep_training_file {
        "kept after training"
    } else {
        "deleted after training"
    }
}

fn math_policy_summary(args: &TrainArgs) -> String {
    if args.balance_math {
        format!(
            "balanced with max share {:.1}%",
            args.math_max_share * 100.0
        )
    } else {
        "reported, not balanced separately".to_string()
    }
}

fn unresolved_preflight_failures(report: &Report, args: &TrainArgs) -> usize {
    report
        .findings
        .iter()
        .filter(|finding| finding.is_failure())
        .filter(|finding| !training_handles_preflight_failure(finding, args))
        .count()
}

fn report_preflight_for_training(report: Report, args: &TrainArgs) -> Report {
    report
        .findings
        .into_iter()
        .map(|finding| {
            if finding.is_failure() && training_handles_preflight_failure(&finding, args) {
                return handled_preflight_finding(finding);
            }
            finding
        })
        .collect()
}

fn handled_preflight_finding(finding: Finding) -> Finding {
    let mut handled = Finding::passed(
        format!("preflight[{}]", finding.check),
        format!("{}; handled by the training stream", finding.summary),
    )
    .with_evidence(finding.evidence);
    handled.failure_mode = finding.failure_mode;
    handled
}

fn training_handles_preflight_failure(finding: &Finding, args: &TrainArgs) -> bool {
    if finding.check == "invalid_utf8" {
        return args.drop_invalid;
    }
    if finding.check == "long_lines" {
        return args.drop_long_lines;
    }
    axis_handled_by_training(&finding.check, args)
}

fn axis_handled_by_training(check: &str, args: &TrainArgs) -> bool {
    let Some(axis_name) = check
        .strip_prefix("axis[")
        .and_then(|rest| rest.strip_suffix(']'))
    else {
        return false;
    };

    default_axes().into_iter().any(|axis| {
        axis.name == axis_name
            && match axis.action {
                Action::Collapse => true,
                Action::Decide => args.decide.iter().any(|name| name == axis.name),
                Action::Preserve => false,
            }
    })
}

fn prepare_options(args: &TrainArgs) -> Result<PrepareOptions> {
    let gib = 1024_u64 * 1024 * 1024;
    let options = PrepareOptions {
        target_lines: args.lines,
        alpha: args.alpha,
        seed: args.seed,
        max_line_bytes: args.max_sentence_length,
        shuffle_buffer_lines: args.shuffle_buffer_lines,
        memory_budget_bytes: args.memory_budget_gb.saturating_mul(gib),
        drop_invalid_utf8: args.drop_invalid,
        drop_long_lines: args.drop_long_lines,
        math_policy: math_policy(args),
    };
    options.validate()?;
    Ok(options)
}

fn math_policy(args: &TrainArgs) -> train::MathPolicy {
    if args.balance_math {
        train::MathPolicy::Balanced {
            max_share: args.math_max_share,
        }
    } else {
        train::MathPolicy::ReportOnly
    }
}

fn run_sentencepiece(
    sources: &[Source],
    plan: &train::CorpusPlan,
    canonicalizer: &Canonicalizer,
    options: &PrepareOptions,
    args: &TrainArgs,
    user_symbols: &[String],
    rust_jobs: Option<usize>,
) -> Result<()> {
    let mut input = prepared_training_file(args)?;
    let input_path = input.path().to_path_buf();
    {
        let mut writer = std::io::BufWriter::new(input.as_file_mut());
        step("writing temporary balanced corpus for spm_train…");
        train::stream_prepared(sources, plan, canonicalizer, options, &mut writer)?;
        writer.flush().context("flushing temporary training file")?;
    }

    let settings = trainer_settings(&input_path, args, options, user_symbols, rust_jobs);
    let mut command = trainer_command(args, &settings)?;

    step("starting spm_train…");
    let status = command
        .status()
        .with_context(|| format!("starting {}", args.spm_train.display()))?;

    if !status.success() {
        bail!("spm_train exited with {status}");
    }

    if args.keep_training_file {
        let kept = input.keep().map_err(|error| error.error)?;
        note_line(&format!("kept training input: {}", kept.1.display()));
    }
    Ok(())
}

fn prepared_training_file(args: &TrainArgs) -> Result<tempfile::NamedTempFile> {
    match &args.training_temp_dir {
        Some(directory) => tempfile::NamedTempFile::new_in(directory)
            .with_context(|| format!("creating temporary file in {}", directory.display())),
        None => tempfile::NamedTempFile::new().context("creating temporary training file"),
    }
}

fn trainer_command(
    args: &TrainArgs,
    settings: &SentencePieceSettings,
) -> Result<std::process::Command> {
    let mut command = match args.trainer_backend {
        TrainerBackend::UvPython => uv_python_trainer(settings)?,
        TrainerBackend::SpmTrain => spm_train_command(&args.spm_train, settings),
    };
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    Ok(command)
}

fn uv_python_trainer(settings: &SentencePieceSettings) -> Result<std::process::Command> {
    const TRAINER: &str = r#"
import json
import sys

import sentencepiece as spm

spm.SentencePieceTrainer.train(**json.loads(sys.argv[1]))
"#;

    let mut command = std::process::Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("-c")
        .arg(TRAINER)
        .arg(settings_json(settings)?);
    Ok(command)
}

fn spm_train_command(binary: &Path, settings: &SentencePieceSettings) -> std::process::Command {
    let mut command = std::process::Command::new(binary);
    command.args(sentencepiece_args(settings));
    command
}

#[derive(Debug, Clone)]
struct SentencePieceSettings {
    input: PathBuf,
    model_prefix: PathBuf,
    vocab_size: u32,
    character_coverage: f64,
    max_sentencepiece_length: usize,
    max_sentence_length: usize,
    input_sentence_size: u64,
    num_threads: usize,
    user_defined_symbols: Vec<String>,
}

fn trainer_settings(
    input: &Path,
    args: &TrainArgs,
    options: &PrepareOptions,
    user_symbols: &[String],
    rust_jobs: Option<usize>,
) -> SentencePieceSettings {
    SentencePieceSettings {
        input: input.to_path_buf(),
        model_prefix: args.model_prefix.clone(),
        vocab_size: args.vocab_size,
        character_coverage: args.character_coverage,
        max_sentencepiece_length: args.max_sentencepiece_length,
        max_sentence_length: args.max_sentence_length,
        input_sentence_size: options.target_lines,
        num_threads: args.spm_threads.or(rust_jobs).unwrap_or(16).max(1),
        user_defined_symbols: user_symbols.to_vec(),
    }
}

fn sentencepiece_args(settings: &SentencePieceSettings) -> Vec<String> {
    let mut spm_args = vec![
        format!("--input={}", settings.input.display()),
        format!("--model_prefix={}", settings.model_prefix.display()),
        "--model_type=bpe".to_string(),
        format!("--vocab_size={}", settings.vocab_size),
        format!("--character_coverage={}", settings.character_coverage),
        "--byte_fallback=true".to_string(),
        "--normalization_rule_name=identity".to_string(),
        "--add_dummy_prefix=false".to_string(),
        "--remove_extra_whitespaces=false".to_string(),
        "--split_by_unicode_script=true".to_string(),
        "--split_by_whitespace=true".to_string(),
        "--split_digits=true".to_string(),
        format!(
            "--max_sentencepiece_length={}",
            settings.max_sentencepiece_length
        ),
        format!("--max_sentence_length={}", settings.max_sentence_length),
        format!("--input_sentence_size={}", settings.input_sentence_size),
        "--shuffle_input_sentence=false".to_string(),
        "--train_extremely_large_corpus=true".to_string(),
        format!("--num_threads={}", settings.num_threads),
    ];

    if !settings.user_defined_symbols.is_empty() {
        spm_args.push(format!(
            "--user_defined_symbols={}",
            settings.user_defined_symbols.join(",")
        ));
    }

    spm_args
}

fn settings_json(settings: &SentencePieceSettings) -> Result<String> {
    let mut value = json!({
        "input": settings.input,
        "model_prefix": settings.model_prefix,
        "model_type": "bpe",
        "vocab_size": settings.vocab_size,
        "character_coverage": settings.character_coverage,
        "byte_fallback": true,
        "normalization_rule_name": "identity",
        "add_dummy_prefix": false,
        "remove_extra_whitespaces": false,
        "split_by_unicode_script": true,
        "split_by_whitespace": true,
        "split_digits": true,
        "max_sentencepiece_length": settings.max_sentencepiece_length,
        "max_sentence_length": settings.max_sentence_length,
        "input_sentence_size": settings.input_sentence_size,
        "shuffle_input_sentence": false,
        "train_extremely_large_corpus": true,
        "num_threads": settings.num_threads,
    });

    if !settings.user_defined_symbols.is_empty() {
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "user_defined_symbols".to_string(),
                json!(settings.user_defined_symbols),
            );
        }
    }

    serde_json::to_string(&value).context("serializing SentencePiece trainer settings")
}

fn trainer_summary(args: &TrainArgs, settings: &SentencePieceSettings) -> String {
    match args.trainer_backend {
        TrainerBackend::UvPython => format!(
            "uv run python SentencePieceTrainer.train with {}",
            settings_json(settings).unwrap_or_else(|_| "<unserializable settings>".to_string())
        ),
        TrainerBackend::SpmTrain => {
            format!(
                "{} {}",
                args.spm_train.display(),
                sentencepiece_args(settings).join(" ")
            )
        }
    }
}

fn user_defined_symbols(args: &TrainArgs) -> Result<Vec<String>> {
    let mut symbols = args.user_defined_symbol.clone();
    if let Some(path) = &args.user_defined_symbols_file {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        symbols.extend(
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string),
        );
    }
    Ok(symbols)
}

fn sentencepiece_model_path(prefix: &Path) -> PathBuf {
    let mut path = PathBuf::from(prefix);
    let Some(name) = path.file_name() else {
        return PathBuf::from(format!("{}.model", prefix.display()));
    };
    let mut file_name = name.to_os_string();
    file_name.push(".model");
    path.set_file_name(file_name);
    path
}

fn rewrite_with_progress(
    found: &Discovery,
    destination: &Destination,
    canonicalizer: &Canonicalizer,
    on_invalid: OnInvalidUtf8,
) -> Result<Vec<PathBuf>> {
    // Targets first, and sequentially: this is the step that creates directories, and settling
    // them up front leaves the parallel pass below writing to distinct, already-existing places.
    let targets: Vec<PathBuf> = found
        .sources
        .iter()
        .map(|source| destination.target(source))
        .collect::<Result<_>>()?;

    let bar = byte_bar(found.total_bytes(), "canonicalizing");
    let tallies: Vec<(String, Tally)> = found
        .sources
        .par_iter()
        .zip(targets.par_iter())
        .map(|(source, target)| -> Result<(String, Tally)> {
            let tally = rewrite_file(source, target, canonicalizer, on_invalid)?;
            bar.inc(source.size_bytes());
            Ok((source.label(), tally))
        })
        .collect::<Result<_>>()?;
    bar.finish_and_clear();

    for (label, tally) in &tallies {
        note_line(&format!("{label}: {}", tally.summary()));
    }
    Ok(targets)
}

fn scan_with_progress(found: &Discovery, config: &scan::Config) -> scan::Totals {
    let bar = byte_bar(found.total_bytes(), "scanning");

    let totals: scan::Totals = found
        .sources
        .par_iter()
        .map(|source| {
            let counts = scan::scan_source(source, config).unwrap_or_default();
            bar.inc(source.size_bytes());
            (source.label(), counts)
        })
        .collect();

    bar.finish_and_clear();
    totals
}

/// A byte-denominated bar, because the total is knowable from file sizes before anything is
/// read. It draws only on a terminal, so a redirected run stays clean.
fn byte_bar(total: u64, what: &str) -> ProgressBar {
    if !std::io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template("{msg} {wide_bar} {bytes}/{total_bytes} ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    bar.set_message(what.to_string());
    bar
}

/// Commentary, which is never the report.
fn note_line(message: &str) {
    eprintln!("{message}");
}

/// Names a phase that runs long enough to look hung but cannot show a percentage — the walk
/// cannot say how many files it will find without walking twice.
fn step(message: &str) {
    if std::io::stderr().is_terminal() {
        eprintln!("{message}");
    }
}

fn emit(report: &Report, output: &OutputArgs) {
    let rendered = if output.json {
        render::as_json(report)
    } else {
        render::as_text(report)
    };
    println!("{rendered}");
}
