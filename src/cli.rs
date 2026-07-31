use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use crate::balance::{self, BalanceSummary};
use crate::config::{self, EffectiveConfig};
use crate::corpus::{self, DiscoveredCorpus};
use crate::progress::ProgressReporter;
use crate::repair::{self, RepairSummary};
use crate::trainer::{self, TrainerOutput};

#[derive(Debug, Parser)]
#[command(name = "spm-ocr", version, about)]
pub struct Cli {
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train(TrainArgs),
}

#[derive(Debug, Args)]
struct TrainArgs {
    #[arg(long, default_value = "cfg.json")]
    config: PathBuf,
    #[arg(long, default_value = "cfg.json.ocr")]
    preset_config: PathBuf,
}

#[derive(Debug, Serialize)]
struct TrainSummary {
    preset: String,
    corpus_path: PathBuf,
    text_files: usize,
    fixed_corpus: PathBuf,
    issue_log: PathBuf,
    lines_read: u64,
    lines_written: u64,
    lines_fixed: u64,
    lines_skipped: u64,
    source_issues: usize,
    balanced_lines: u64,
    balance_report: PathBuf,
    model: PathBuf,
    vocab: PathBuf,
    trainer_output: PathBuf,
    work_dir: PathBuf,
    model_prefix: String,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => train(args, cli.json),
    }
}

fn train(args: TrainArgs, json_output: bool) -> Result<()> {
    let progress = ProgressReporter::new(json_output);

    let config = load_config(&args, &progress)?;
    let corpus = discover_corpus(&config, &progress)?;
    let repair = repair_corpus(&config, &corpus, &progress)?;
    if repair.lines_written == 0 {
        bail!(
            "no usable text lines found under {}; see {}",
            config.corpus.path.display(),
            repair.issue_log.display()
        );
    }
    let balance = balance_corpus(&config, &repair, &progress)?;
    let trainer = train_sentencepiece(&config, &balance, &progress)?;
    let summary = TrainSummary::from_outputs(&config, &corpus, &repair, &balance, &trainer);

    if json_output {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        let source_issue_suffix = if summary.source_issues == 0 {
            String::new()
        } else {
            format!(", {} source issue(s) logged", summary.source_issues)
        };
        println!(
            "Trained: preset `{}`, {} text file(s), {} repaired line(s), {} balanced line(s), {} fixed, {} skipped{}, model `{}`",
            summary.preset,
            summary.text_files,
            summary.lines_written,
            summary.balanced_lines,
            summary.lines_fixed,
            summary.lines_skipped,
            source_issue_suffix,
            summary.model.display()
        );
    }

    Ok(())
}

fn load_config(args: &TrainArgs, progress: &ProgressReporter) -> Result<EffectiveConfig> {
    let stage = progress.stage("loading config");
    let config = config::load_effective_config(&args.config, &args.preset_config)
        .with_context(|| format!("could not load {}", args.config.display()))?;
    stage.finish("loaded config");
    Ok(config)
}

fn discover_corpus(
    config: &EffectiveConfig,
    progress: &ProgressReporter,
) -> Result<DiscoveredCorpus> {
    let stage = progress.stage("discovering corpus text files");
    let unpack_dir = config.output.work_dir.join("unpacked");
    let corpus = corpus::discover_text_files(&config.corpus.path, &unpack_dir)
        .with_context(|| format!("could not discover {}", config.corpus.path.display()))?;

    stage.finish(format!("found {} text file(s)", corpus.files.len()));
    Ok(corpus)
}

fn repair_corpus(
    config: &EffectiveConfig,
    corpus: &DiscoveredCorpus,
    progress: &ProgressReporter,
) -> Result<RepairSummary> {
    repair::repair_corpus(config, corpus, progress)
        .with_context(|| format!("could not repair {}", config.corpus.path.display()))
}

fn balance_corpus(
    config: &EffectiveConfig,
    repair: &RepairSummary,
    progress: &ProgressReporter,
) -> Result<BalanceSummary> {
    balance::balance_corpus(config, repair, progress).context("could not balance corpus")
}

fn train_sentencepiece(
    config: &EffectiveConfig,
    balance: &BalanceSummary,
    progress: &ProgressReporter,
) -> Result<TrainerOutput> {
    trainer::train_sentencepiece(config, &balance.fixed_corpus, progress)
        .context("could not train SentencePiece")
}

impl TrainSummary {
    fn from_outputs(
        config: &EffectiveConfig,
        corpus: &DiscoveredCorpus,
        repair: &RepairSummary,
        balance: &BalanceSummary,
        trainer: &TrainerOutput,
    ) -> Self {
        Self {
            preset: config.preset.clone(),
            corpus_path: corpus.root.clone(),
            text_files: corpus.files.len(),
            fixed_corpus: repair.fixed_corpus.clone(),
            issue_log: repair.issue_log.clone(),
            lines_read: repair.lines_read,
            lines_written: repair.lines_written,
            lines_fixed: repair.lines_fixed,
            lines_skipped: repair.lines_skipped,
            source_issues: repair.source_issues,
            balanced_lines: balance.output_lines,
            balance_report: balance.report.clone(),
            model: trainer.model.clone(),
            vocab: trainer.vocab.clone(),
            trainer_output: config.output.work_dir.join("trainer_output.json"),
            work_dir: config.output.work_dir.clone(),
            model_prefix: config.output.model_prefix.clone(),
        }
    }
}
