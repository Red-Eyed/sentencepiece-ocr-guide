//! `spm-ocr` — the command line edge.
//!
//! Everything app-specific lives here: where to read, what to print, what to exit with. The
//! library below it takes bytes and returns findings.
//!
//! Stdout carries the report and nothing else, so `--json` is always pipeable. Progress,
//! announcements and notes go to stderr.

use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};

use spm_ocr::corpus::axis::default_axes;
use spm_ocr::corpus::scan;
use spm_ocr::corpus::source::{Discovery, discover};
use spm_ocr::render;
use spm_ocr::report::{Report, Severity};

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

fn scan_corpus(paths: &[PathBuf]) -> Report {
    step("discovering files…");
    let found = discover(paths);

    if let Some(note) = found.summarize_skipped(3) {
        note_line(&note);
    }

    let axes = default_axes();
    let totals = scan_with_progress(&found, &axes);
    scan::report(&totals, &axes)
}

fn scan_with_progress(found: &Discovery, axes: &[spm_ocr::corpus::axis::Axis]) -> scan::Totals {
    let bar = byte_bar(found.total_bytes(), "scanning");

    let totals: scan::Totals = found
        .sources
        .iter()
        .map(|source| {
            let counts = scan::scan_source(source, axes).unwrap_or_default();
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
