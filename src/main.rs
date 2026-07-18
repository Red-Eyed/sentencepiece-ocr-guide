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
use rayon::prelude::*;

use spm_ocr::corpus::axis::default_axes;
use spm_ocr::corpus::canonical::Canonicalizer;
use spm_ocr::corpus::rewrite::{OnInvalidUtf8, Tally, rewrite_file};
use spm_ocr::corpus::scan;
use spm_ocr::corpus::source::{Discovery, Source, discover};
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
    let totals = scan_with_progress(&discover(&written), &axes);
    Ok(scan::report(&totals, &axes))
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

fn scan_with_progress(found: &Discovery, axes: &[spm_ocr::corpus::axis::Axis]) -> scan::Totals {
    let bar = byte_bar(found.total_bytes(), "scanning");

    let totals: scan::Totals = found
        .sources
        .par_iter()
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
