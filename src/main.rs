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
use serde::Deserialize;
use serde_json::json;

use spm_ocr::corpus::axis::{Action, default_axes};
use spm_ocr::corpus::balance;
use spm_ocr::corpus::canonical::Canonicalizer;
use spm_ocr::corpus::scan;
use spm_ocr::corpus::source::{Discovery, Source, discover};
use spm_ocr::crosscheck;
use spm_ocr::format::count;
use spm_ocr::model::{artifact, suite};
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
    /// Train with preprocessing, balancing, SentencePiece, and post-train checks.
    Train {
        #[command(flatten)]
        args: TrainArgs,

        #[command(flatten)]
        output: OutputArgs,
    },
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
    /// Strict JSON training config. Unknown or missing keys fail.
    #[arg(long, value_name = "FILE")]
    config: PathBuf,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize, ValueEnum)]
enum TrainerBackend {
    /// Use `uv run python` and the project `sentencepiece` dependency.
    #[serde(rename = "uv-python", alias = "uv_python")]
    UvPython,
    /// Use an external `spm_train` executable.
    #[serde(rename = "spm-train", alias = "spm_train")]
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
        Command::Train { args, output } => {
            configure_threads(output.jobs)?;
            let args = TrainConfig::read(&args.config)?.into_options()?;
            let report = train_tokenizer(&args, output.jobs)?;
            emit(&report, &output);
            std::process::exit(report.exit_code(output.fail_on.into()));
        }
    }
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

fn train_tokenizer(args: &TrainOptions, rust_jobs: Option<usize>) -> Result<Report> {
    step("discovering files…");
    let found = discover(&args.paths);
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

    let artifacts = run_sentencepiece(
        &found.sources,
        &plan,
        &canonicalizer,
        &options,
        args,
        &user_symbols,
        rust_jobs,
    )?;
    report.findings.push(training_artifact_report(&artifacts));

    step("reading the trained model…");
    let artifact = artifact::load(&artifacts.model_path)?;
    report.extend(suite::check(&artifact, &suite::Options::default()));
    report
        .findings
        .push(crosscheck::script_coverage(&combined, &artifact).about(13));

    Ok(report)
}

#[derive(Debug, Clone)]
struct TrainOptions {
    paths: Vec<PathBuf>,
    model_prefix: PathBuf,
    lines: u64,
    alpha: f64,
    seed: u64,
    vocab_size: u32,
    character_coverage: f64,
    max_sentence_length: usize,
    max_sentencepiece_length: usize,
    shuffle_buffer_lines: usize,
    memory_budget_gb: u64,
    training_temp_dir: Option<PathBuf>,
    keep_training_file: bool,
    balance_math: bool,
    math_max_share: f64,
    spm_threads: Option<usize>,
    trainer_backend: TrainerBackend,
    spm_train: PathBuf,
    decide: Vec<String>,
    drop_invalid: bool,
    drop_long_lines: bool,
    user_defined_symbol: Vec<String>,
    user_defined_symbols_file: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainConfig {
    paths: Vec<PathBuf>,
    model_prefix: PathBuf,
    lines: u64,
    alpha: f64,
    seed: u64,
    vocab_size: u32,
    character_coverage: f64,
    max_sentence_length: usize,
    max_sentencepiece_length: usize,
    shuffle_buffer_lines: usize,
    memory_budget_gb: u64,
    #[serde(deserialize_with = "required_option")]
    training_temp_dir: Option<PathBuf>,
    keep_training_file: bool,
    balance_math: bool,
    math_max_share: f64,
    #[serde(deserialize_with = "required_option")]
    spm_threads: Option<usize>,
    trainer_backend: TrainerBackend,
    spm_train: PathBuf,
    decide: Vec<String>,
    drop_invalid: bool,
    drop_long_lines: bool,
    user_defined_symbols: Vec<String>,
    #[serde(deserialize_with = "required_option")]
    user_defined_symbols_file: Option<PathBuf>,
}

impl TrainConfig {
    fn read(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    fn into_options(self) -> Result<TrainOptions> {
        if self.paths.is_empty() {
            bail!("train config must include at least one path");
        }

        Ok(TrainOptions {
            paths: self.paths,
            model_prefix: self.model_prefix,
            lines: self.lines,
            alpha: self.alpha,
            seed: self.seed,
            vocab_size: self.vocab_size,
            character_coverage: self.character_coverage,
            max_sentence_length: self.max_sentence_length,
            max_sentencepiece_length: self.max_sentencepiece_length,
            shuffle_buffer_lines: self.shuffle_buffer_lines,
            memory_budget_gb: self.memory_budget_gb,
            training_temp_dir: self.training_temp_dir,
            keep_training_file: self.keep_training_file,
            balance_math: self.balance_math,
            math_max_share: self.math_max_share,
            spm_threads: self.spm_threads,
            trainer_backend: self.trainer_backend,
            spm_train: self.spm_train,
            decide: self.decide,
            drop_invalid: self.drop_invalid,
            drop_long_lines: self.drop_long_lines,
            user_defined_symbol: self.user_defined_symbols,
            user_defined_symbols_file: self.user_defined_symbols_file,
        })
    }
}

fn required_option<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn training_report(
    plan: &train::CorpusPlan,
    args: &TrainOptions,
    options: &PrepareOptions,
) -> Report {
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

fn training_file_policy(args: &TrainOptions) -> &'static str {
    if args.keep_training_file {
        "kept after training"
    } else {
        "deleted after training"
    }
}

fn math_policy_summary(args: &TrainOptions) -> String {
    if args.balance_math {
        format!(
            "balanced with max share {:.1}%",
            args.math_max_share * 100.0
        )
    } else {
        "reported, not balanced separately".to_string()
    }
}

fn unresolved_preflight_failures(report: &Report, args: &TrainOptions) -> usize {
    report
        .findings
        .iter()
        .filter(|finding| finding.is_failure())
        .filter(|finding| !training_handles_preflight_failure(finding, args))
        .count()
}

fn report_preflight_for_training(report: Report, args: &TrainOptions) -> Report {
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

fn training_handles_preflight_failure(finding: &Finding, args: &TrainOptions) -> bool {
    if finding.check == "invalid_utf8" {
        return args.drop_invalid;
    }
    if finding.check == "long_lines" {
        return args.drop_long_lines;
    }
    axis_handled_by_training(&finding.check, args)
}

fn axis_handled_by_training(check: &str, args: &TrainOptions) -> bool {
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

fn prepare_options(args: &TrainOptions) -> Result<PrepareOptions> {
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

fn math_policy(args: &TrainOptions) -> train::MathPolicy {
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
    args: &TrainOptions,
    user_symbols: &[String],
    rust_jobs: Option<usize>,
) -> Result<TrainingArtifacts> {
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

    let training_input = if args.keep_training_file {
        let kept = input.keep().map_err(|error| error.error)?;
        note_line(&format!("kept training input: {}", kept.1.display()));
        Some(kept.1)
    } else {
        None
    };

    let artifacts = TrainingArtifacts {
        model_path: sentencepiece_model_path(&args.model_prefix),
        vocab_path: sentencepiece_vocab_path(&args.model_prefix),
        training_input,
    };
    note_line(&format!(
        "wrote tokenizer model: {}",
        artifacts.model_path.display()
    ));
    note_line(&format!(
        "wrote tokenizer vocab: {}",
        artifacts.vocab_path.display()
    ));
    Ok(artifacts)
}

fn prepared_training_file(args: &TrainOptions) -> Result<tempfile::NamedTempFile> {
    match &args.training_temp_dir {
        Some(directory) => tempfile::NamedTempFile::new_in(directory)
            .with_context(|| format!("creating temporary file in {}", directory.display())),
        None => tempfile::NamedTempFile::new().context("creating temporary training file"),
    }
}

#[derive(Debug, Clone)]
struct TrainingArtifacts {
    model_path: PathBuf,
    vocab_path: PathBuf,
    training_input: Option<PathBuf>,
}

fn training_artifact_report(artifacts: &TrainingArtifacts) -> Finding {
    let mut evidence = vec![
        format!("model: {}", artifacts.model_path.display()),
        format!("vocab: {}", artifacts.vocab_path.display()),
    ];
    if let Some(path) = &artifacts.training_input {
        evidence.push(format!("prepared corpus: {}", path.display()));
    }

    Finding::passed(
        "tokenizer_artifacts",
        "SentencePiece wrote tokenizer artifacts",
    )
    .with_evidence(evidence)
}

fn trainer_command(
    args: &TrainOptions,
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
    args: &TrainOptions,
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

fn trainer_summary(args: &TrainOptions, settings: &SentencePieceSettings) -> String {
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

fn user_defined_symbols(args: &TrainOptions) -> Result<Vec<String>> {
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
    sentencepiece_output_path(prefix, "model")
}

fn sentencepiece_vocab_path(prefix: &Path) -> PathBuf {
    sentencepiece_output_path(prefix, "vocab")
}

fn sentencepiece_output_path(prefix: &Path, extension: &str) -> PathBuf {
    let mut path = PathBuf::from(prefix);
    let Some(name) = path.file_name() else {
        return PathBuf::from(format!("{}.{extension}", prefix.display()));
    };
    let mut file_name = name.to_os_string();
    file_name.push(".");
    file_name.push(extension);
    path.set_file_name(file_name);
    path
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_train_config_loads() {
        let config: TrainConfig =
            serde_json::from_str(include_str!("../cfg.json.example")).expect("valid config");
        let options = config.into_options().expect("valid options");

        assert_eq!(options.paths, vec![PathBuf::from("corpus/")]);
        assert_eq!(options.model_prefix, PathBuf::from("ocr_tokenizer"));
        assert_eq!(options.vocab_size, 40_000);
    }

    #[test]
    fn train_config_rejects_unknown_keys() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../cfg.json.example")).expect("valid json");
        value
            .as_object_mut()
            .expect("object")
            .insert("vocab_szie".to_string(), json!(40_000));

        let error = serde_json::from_value::<TrainConfig>(value)
            .expect_err("unknown keys must fail")
            .to_string();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn train_config_rejects_missing_nullable_keys() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../cfg.json.example")).expect("valid json");
        value
            .as_object_mut()
            .expect("object")
            .remove("user_defined_symbols_file");

        let error = serde_json::from_value::<TrainConfig>(value)
            .expect_err("missing keys must fail")
            .to_string();
        assert!(error.contains("missing field `user_defined_symbols_file`"));
    }

    #[test]
    fn train_requires_config() {
        let parsed = Cli::try_parse_from(["spm-ocr", "train"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn train_rejects_positional_paths() {
        let parsed = Cli::try_parse_from(["spm-ocr", "train", "corpus/", "--config", "cfg.json"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn only_train_is_exposed() {
        let parsed = Cli::try_parse_from(["spm-ocr", "corpus", "corpus/"]);
        assert!(parsed.is_err());
    }
}
