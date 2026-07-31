use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use thiserror::Error;
use unicode_script::{Script, UnicodeScript};

use crate::config::EffectiveConfig;
use crate::progress::ProgressReporter;
use crate::repair::{RepairSummary, ShardSummary};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BalanceSummary {
    pub fixed_corpus: PathBuf,
    pub report: PathBuf,
    pub input_lines: u64,
    pub output_lines: u64,
    pub buckets: Vec<BucketBalance>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BucketBalance {
    pub bucket: BucketKey,
    pub input_lines: u64,
    pub target_lines: u64,
    pub output_lines: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BucketKey {
    pub domain: Domain,
    pub label: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Text,
    Math,
}

#[derive(Debug, Error)]
pub enum BalanceError {
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

pub fn classify_line(path: &Path, text: &str) -> BucketKey {
    let domain = classify_domain(text);
    let label = language_hint(path).unwrap_or_else(|| format!("script:{}", dominant_script(text)));
    BucketKey { domain, label }
}

pub fn shard_file_name(bucket: &BucketKey) -> String {
    let serialized = serde_json::to_vec(bucket).expect("bucket key is serializable");
    blake3::hash(&serialized).to_hex().to_string()
}

pub fn balance_corpus(
    config: &EffectiveConfig,
    repair: &RepairSummary,
    progress: &ProgressReporter,
) -> Result<BalanceSummary, BalanceError> {
    let paths = BalancePaths::from_config(config);
    paths.create_dirs()?;

    let stage = progress.stage("balancing corpus");
    let summary = if config.balancing.enabled {
        assemble_balanced(config, repair, &paths)?
    } else {
        write_report(
            repair.fixed_corpus.clone(),
            paths.report.clone(),
            repair.lines_written,
            repair.lines_written,
            repair
                .shards
                .iter()
                .map(|shard| BucketBalance {
                    bucket: shard.bucket.clone(),
                    input_lines: shard.lines,
                    target_lines: shard.lines,
                    output_lines: shard.lines,
                })
                .collect(),
        )?
    };

    stage.finish(format!(
        "balanced corpus: {} input line(s), {} output line(s)",
        summary.input_lines, summary.output_lines
    ));
    Ok(summary)
}

fn assemble_balanced(
    config: &EffectiveConfig,
    repair: &RepairSummary,
    paths: &BalancePaths,
) -> Result<BalanceSummary, BalanceError> {
    let targets = compute_targets(config, repair);
    let mut selected = Vec::new();
    let mut buckets = Vec::new();

    for shard in &repair.shards {
        let target = targets.get(&shard.bucket).copied().unwrap_or(0);
        let lines = sample_shard(shard, target, config.balancing.shuffle_seed)?;
        let output_lines = lines.len() as u64;
        selected.extend(lines);
        buckets.push(BucketBalance {
            bucket: shard.bucket.clone(),
            input_lines: shard.lines,
            target_lines: target,
            output_lines,
        });
    }

    let mut rng = ChaCha20Rng::seed_from_u64(config.balancing.shuffle_seed);
    selected.shuffle(&mut rng);
    write_lines(&paths.fixed_corpus, &selected)?;

    let input_lines = repair.lines_written;
    let output_lines = selected.len() as u64;
    write_report(
        paths.fixed_corpus.clone(),
        paths.report.clone(),
        input_lines,
        output_lines,
        buckets,
    )
}

fn compute_targets(config: &EffectiveConfig, repair: &RepairSummary) -> HashMap<BucketKey, u64> {
    let total_lines = repair.lines_written;
    if total_lines == 0 {
        return HashMap::new();
    }

    let requested_total = config.balancing.total_lines.min(total_lines);
    let weights = repair
        .shards
        .iter()
        .map(|shard| {
            (
                shard.bucket.clone(),
                (shard.lines as f64).powf(config.balancing.alpha),
            )
        })
        .collect::<Vec<_>>();
    let weight_sum = weights.iter().map(|(_, weight)| weight).sum::<f64>();

    let mut targets = weights
        .into_iter()
        .map(|(bucket, weight)| {
            let input_lines = repair
                .shards
                .iter()
                .find(|shard| shard.bucket == bucket)
                .map(|shard| shard.lines)
                .unwrap_or(0);
            let target = ((weight / weight_sum) * requested_total as f64).round() as u64;
            (bucket, target.clamp(1, input_lines))
        })
        .collect::<HashMap<_, _>>();
    rebalance_target_sum(&mut targets, repair, requested_total);
    targets
}

fn rebalance_target_sum(
    targets: &mut HashMap<BucketKey, u64>,
    repair: &RepairSummary,
    requested_total: u64,
) {
    while targets.values().sum::<u64>() < requested_total {
        let Some(shard) = repair
            .shards
            .iter()
            .filter(|shard| targets.get(&shard.bucket).copied().unwrap_or(0) < shard.lines)
            .max_by_key(|shard| shard.lines - targets.get(&shard.bucket).copied().unwrap_or(0))
        else {
            return;
        };
        *targets.entry(shard.bucket.clone()).or_default() += 1;
    }

    while targets.values().sum::<u64>() > requested_total {
        let Some(bucket) = targets
            .iter()
            .filter(|(_, target)| **target > 1)
            .max_by_key(|(_, target)| **target)
            .map(|(bucket, _)| bucket.clone())
        else {
            return;
        };
        *targets.entry(bucket).or_default() -= 1;
    }
}

fn sample_shard(shard: &ShardSummary, target: u64, seed: u64) -> Result<Vec<String>, BalanceError> {
    if target == 0 {
        return Ok(Vec::new());
    }

    let file = File::open(&shard.path).map_err(|source| BalanceError::Open {
        path: shard.path.clone(),
        source,
    })?;
    let reader = BufReader::new(file);
    if shard.lines <= target {
        return reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| BalanceError::Read {
                path: shard.path.clone(),
                source,
            });
    }

    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ bucket_seed(&shard.bucket));
    let mut reservoir = Vec::with_capacity(target as usize);
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| BalanceError::Read {
            path: shard.path.clone(),
            source,
        })?;
        let seen = index as u64 + 1;
        if seen <= target {
            reservoir.push(line);
            continue;
        }

        let replace = rng.gen_range(0..seen);
        if replace < target {
            reservoir[replace as usize] = line;
        }
    }

    Ok(reservoir)
}

fn write_report(
    fixed_corpus: PathBuf,
    report: PathBuf,
    input_lines: u64,
    output_lines: u64,
    mut buckets: Vec<BucketBalance>,
) -> Result<BalanceSummary, BalanceError> {
    buckets.sort_by(|left, right| left.bucket.cmp(&right.bucket));
    let summary = BalanceSummary {
        fixed_corpus,
        report,
        input_lines,
        output_lines,
        buckets,
    };
    write_json(&summary.report, &summary)?;
    Ok(summary)
}

fn write_lines(path: &Path, lines: &[String]) -> Result<(), BalanceError> {
    let file = File::create(path).map_err(|source| BalanceError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    for line in lines {
        writer
            .write_all(line.as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
            .map_err(|source| BalanceError::Write {
                path: path.to_path_buf(),
                source,
            })?;
    }
    writer.flush().map_err(|source| BalanceError::Write {
        path: path.to_path_buf(),
        source,
    })
}

fn write_json<T>(path: &Path, value: &T) -> Result<(), BalanceError>
where
    T: Serialize,
{
    let file = File::create(path).map_err(|source| BalanceError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::to_writer_pretty(BufWriter::new(file), value).map_err(|source| {
        BalanceError::Write {
            path: path.to_path_buf(),
            source: std::io::Error::other(source),
        }
    })
}

fn classify_domain(text: &str) -> Domain {
    if text.contains('\\')
        || text.contains('∑')
        || text.contains('∫')
        || text.contains('√')
        || text.contains('≤')
        || text.contains('≥')
    {
        Domain::Math
    } else {
        Domain::Text
    }
}

fn dominant_script(text: &str) -> String {
    let mut counts = HashMap::<Script, usize>::new();
    for character in text.chars().filter(|character| character.is_alphabetic()) {
        let script = character.script();
        if matches!(script, Script::Common | Script::Inherited | Script::Unknown) {
            continue;
        }
        *counts.entry(script).or_default() += 1;
    }

    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(script, _)| format!("{script:?}").to_lowercase())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn language_hint(path: &Path) -> Option<String> {
    path.components()
        .flat_map(|component| split_path_tokens(&component.as_os_str().to_string_lossy()))
        .find(|token| is_language_token(token))
        .map(|token| format!("lang:{token}"))
}

fn split_path_tokens(value: &str) -> Vec<String> {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn is_language_token(token: &str) -> bool {
    ISO_639_1.contains(&token) || ISO_639_3_COMMON.contains(&token)
}

fn bucket_seed(bucket: &BucketKey) -> u64 {
    let bytes = serde_json::to_vec(bucket).expect("bucket key is serializable");
    let hash = blake3::hash(&bytes);
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().expect("hash has 32 bytes"))
}

struct BalancePaths {
    fixed_corpus: PathBuf,
    report: PathBuf,
}

impl BalancePaths {
    fn from_config(config: &EffectiveConfig) -> Self {
        Self {
            fixed_corpus: config.output.work_dir.join("fixed_corpus.txt"),
            report: config.output.work_dir.join("reports/balance.json"),
        }
    }

    fn create_dirs(&self) -> Result<(), BalanceError> {
        create_parent_dir(&self.fixed_corpus)?;
        create_parent_dir(&self.report)
    }
}

fn create_parent_dir(path: &Path) -> Result<(), BalanceError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };

    fs::create_dir_all(parent).map_err(|source| BalanceError::CreateDir {
        path: parent.to_path_buf(),
        source,
    })
}

const ISO_639_1: &[&str] = &[
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az", "ba", "be", "bg", "bh",
    "bi", "bm", "bn", "bo", "br", "bs", "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy", "da",
    "de", "dv", "dz", "ee", "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fj", "fo", "fr",
    "fy", "ga", "gd", "gl", "gn", "gu", "gv", "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu", "ja", "jv", "ka", "kg", "ki", "kj",
    "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky", "la", "lb", "lg", "li", "ln",
    "lo", "lt", "lu", "lv", "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "na", "nb",
    "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny", "oc", "oj", "om", "or", "os", "pa", "pi",
    "pl", "ps", "pt", "qu", "rm", "rn", "ro", "ru", "rw", "sa", "sc", "sd", "se", "sg", "si", "sk",
    "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw", "ta", "te", "tg", "th", "ti",
    "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty", "ug", "uk", "ur", "uz", "ve", "vi", "vo",
    "wa", "wo", "xh", "yi", "yo", "za", "zh", "zu",
];

const ISO_639_3_COMMON: &[&str] = &[
    "ara", "ben", "deu", "eng", "fas", "fra", "hin", "jpn", "kor", "por", "rus", "spa", "zho",
];

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use crate::config::{
        BalanceAxis, BalancingConfig, CanonicalizationConfig, CorpusConfig, EffectiveConfig,
        LinePolicy, ModelType, NormalizationRuleName, OutputConfig, PythonTrainerConfig,
        SentencePieceConfig, SoftHyphenPolicy, StripRule, TrainerKind, UnicodeForm,
        ValidationConfig, ValidationMode,
    };
    use crate::progress::ProgressReporter;

    use super::*;

    fn config(work_dir: &Path, total_lines: u64) -> EffectiveConfig {
        EffectiveConfig {
            preset: "ocr_multilingual".to_owned(),
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
                total_lines,
                alpha: 0.3,
                hierarchy: vec![BalanceAxis::Domain, BalanceAxis::Script],
                shuffle_seed: 7,
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
    fn path_language_hint_wins_when_present() {
        let bucket = classify_line(Path::new("vendor/es/books/file"), "hello");

        assert_eq!(bucket.label, "lang:es");
    }

    #[test]
    fn falls_back_to_dominant_script() {
        let bucket = classify_line(Path::new("vendor/books/file"), "Привет мир");

        assert_eq!(bucket.label, "script:cyrillic");
    }

    #[test]
    fn balances_imbalanced_shards() {
        let temp = TempDir::new().expect("temp dir");
        let config = config(temp.path(), 6);
        let latin = temp.path().join("latin");
        let cyrillic = temp.path().join("cyrillic");
        fs::write(&latin, "a\nb\nc\nd\ne\nf\n").expect("write latin");
        fs::write(&cyrillic, "ж\n").expect("write cyrillic");
        let repair = RepairSummary {
            fixed_corpus: temp.path().join("fixed_corpus.txt"),
            issue_log: temp.path().join("reports/corpus_issues.jsonl"),
            files_read: 2,
            lines_read: 7,
            lines_written: 7,
            lines_fixed: 0,
            lines_skipped: 0,
            source_issues: 0,
            shards: vec![
                ShardSummary {
                    bucket: BucketKey {
                        domain: Domain::Text,
                        label: "script:latin".to_owned(),
                    },
                    path: latin,
                    lines: 6,
                },
                ShardSummary {
                    bucket: BucketKey {
                        domain: Domain::Text,
                        label: "script:cyrillic".to_owned(),
                    },
                    path: cyrillic,
                    lines: 1,
                },
            ],
        };
        let progress = ProgressReporter::new(true);

        let summary = balance_corpus(&config, &repair, &progress).expect("balance corpus");

        assert_eq!(summary.output_lines, 6);
        assert_eq!(
            fs::read_to_string(temp.path().join("fixed_corpus.txt"))
                .expect("read balanced")
                .lines()
                .count(),
            6
        );
        assert!(temp.path().join("reports/balance.json").exists());
    }
}
