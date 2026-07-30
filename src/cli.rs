use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use crate::config::{self, EffectiveConfig};
use crate::corpus::{self, DiscoveredCorpus};
use crate::progress::ProgressReporter;

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
    let summary = TrainSummary::from_config_and_corpus(&config, &corpus);

    if json_output {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "Ready: preset `{}`, {} text file(s), output `{}`",
            summary.preset,
            summary.text_files,
            summary.work_dir.display()
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
    let corpus = corpus::discover_text_files(&config.corpus.path)
        .with_context(|| format!("could not discover {}", config.corpus.path.display()))?;

    if corpus.files.is_empty() {
        bail!("no text files found under {}", config.corpus.path.display());
    }

    stage.finish(format!("found {} text file(s)", corpus.files.len()));
    Ok(corpus)
}

impl TrainSummary {
    fn from_config_and_corpus(config: &EffectiveConfig, corpus: &DiscoveredCorpus) -> Self {
        Self {
            preset: config.preset.clone(),
            corpus_path: corpus.root.clone(),
            text_files: corpus.files.len(),
            work_dir: config.output.work_dir.clone(),
            model_prefix: config.output.model_prefix.clone(),
        }
    }
}
