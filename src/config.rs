use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path} at {json_path}: {source}")]
    Parse {
        path: PathBuf,
        json_path: String,
        source: serde_json::Error,
    },
    #[error("preset `{0}` was not found")]
    UnknownPreset(String),
    #[error("preset inheritance cycle includes `{0}`")]
    PresetCycle(String),
    #[error("missing required preset field `{0}` after expansion")]
    MissingPresetField(&'static str),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RawConfig {
    #[serde(default)]
    pub preset: Option<String>,
    pub corpus: CorpusConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub canonicalization: PartialCanonicalizationConfig,
    #[serde(default)]
    pub balancing: PartialBalancingConfig,
    #[serde(default)]
    pub sentencepiece: PartialSentencePieceConfig,
    #[serde(default)]
    pub validation: PartialValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CorpusConfig {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    pub work_dir: PathBuf,
    pub model_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PresetCatalog {
    pub version: u32,
    pub default: String,
    pub presets: HashMap<String, PresetConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PresetConfig {
    pub description: String,
    #[serde(default)]
    pub extends: Option<String>,
    #[serde(default)]
    pub canonicalization: PartialCanonicalizationConfig,
    #[serde(default)]
    pub balancing: PartialBalancingConfig,
    #[serde(default)]
    pub sentencepiece: PartialSentencePieceConfig,
    #[serde(default)]
    pub validation: PartialValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EffectiveConfig {
    pub preset: String,
    pub corpus: CorpusConfig,
    pub output: OutputConfig,
    pub canonicalization: CanonicalizationConfig,
    pub balancing: BalancingConfig,
    pub sentencepiece: SentencePieceConfig,
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CanonicalizationConfig {
    pub unicode_form: UnicodeForm,
    pub strip: Vec<StripRule>,
    pub map_nbsp_to_space: bool,
    pub fold_arabic_presentation_forms: bool,
    pub soft_hyphen: SoftHyphenPolicy,
    pub preserve_zwj_zwnj: bool,
    pub preserve_compatibility_chars: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PartialCanonicalizationConfig {
    #[serde(default)]
    pub unicode_form: Option<UnicodeForm>,
    #[serde(default)]
    pub strip: Option<Vec<StripRule>>,
    #[serde(default)]
    pub map_nbsp_to_space: Option<bool>,
    #[serde(default)]
    pub fold_arabic_presentation_forms: Option<bool>,
    #[serde(default)]
    pub soft_hyphen: Option<SoftHyphenPolicy>,
    #[serde(default)]
    pub preserve_zwj_zwnj: Option<bool>,
    #[serde(default)]
    pub preserve_compatibility_chars: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnicodeForm {
    Nfc,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StripRule {
    Bom,
    ZeroWidthSpace,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SoftHyphenPolicy {
    LineFinalToHyphenMidlineStrip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BalancingConfig {
    pub enabled: bool,
    pub mode: BalancingMode,
    pub total_lines: u64,
    pub alpha: f64,
    pub hierarchy: Vec<BalanceAxis>,
    pub min_keep_fraction: f64,
    pub max_downsample_ratio: f64,
    pub collapse_buckets_below_lines: u64,
    pub max_part_lines: u64,
    pub shuffle_seed: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PartialBalancingConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub mode: Option<BalancingMode>,
    #[serde(default)]
    pub total_lines: Option<u64>,
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub hierarchy: Option<Vec<BalanceAxis>>,
    #[serde(default)]
    pub min_keep_fraction: Option<f64>,
    #[serde(default)]
    pub max_downsample_ratio: Option<f64>,
    #[serde(default)]
    pub collapse_buckets_below_lines: Option<u64>,
    #[serde(default)]
    pub max_part_lines: Option<u64>,
    #[serde(default)]
    pub shuffle_seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BalancingMode {
    Conservative,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BalanceAxis {
    Domain,
    Script,
    LanguageHint,
    SourceGroup,
    LengthBin,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SentencePieceConfig {
    pub trainer: TrainerKind,
    pub python: PythonTrainerConfig,
    pub model_type: ModelType,
    pub vocab_size: u32,
    pub character_coverage: f64,
    pub byte_fallback: bool,
    pub normalization_rule_name: NormalizationRuleName,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PartialSentencePieceConfig {
    #[serde(default)]
    pub trainer: Option<TrainerKind>,
    #[serde(default)]
    pub python: Option<PythonTrainerConfig>,
    #[serde(default)]
    pub model_type: Option<ModelType>,
    #[serde(default)]
    pub vocab_size: Option<u32>,
    #[serde(default)]
    pub character_coverage: Option<f64>,
    #[serde(default)]
    pub byte_fallback: Option<bool>,
    #[serde(default)]
    pub normalization_rule_name: Option<NormalizationRuleName>,
    #[serde(default)]
    pub add_dummy_prefix: Option<bool>,
    #[serde(default)]
    pub remove_extra_whitespaces: Option<bool>,
    #[serde(default)]
    pub split_by_unicode_script: Option<bool>,
    #[serde(default)]
    pub split_by_whitespace: Option<bool>,
    #[serde(default)]
    pub split_digits: Option<bool>,
    #[serde(default)]
    pub max_sentencepiece_length: Option<u32>,
    #[serde(default)]
    pub max_sentence_length: Option<u32>,
    #[serde(default)]
    pub input_sentence_size: Option<u64>,
    #[serde(default)]
    pub shuffle_input_sentence: Option<bool>,
    #[serde(default)]
    pub train_extremely_large_corpus: Option<bool>,
    #[serde(default)]
    pub user_defined_symbols: Option<Vec<String>>,
    #[serde(default)]
    pub num_threads: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainerKind {
    PythonSentencepiece,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PythonTrainerConfig {
    pub runner: String,
    pub args: Vec<String>,
    pub module: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Bpe,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationRuleName {
    Identity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ValidationConfig {
    pub mode: ValidationMode,
    pub line_policy: LinePolicy,
    pub issue_log: PathBuf,
    pub include_line_text_in_log: bool,
    pub round_trip_sample_per_bucket: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PartialValidationConfig {
    #[serde(default)]
    pub mode: Option<ValidationMode>,
    #[serde(default)]
    pub line_policy: Option<LinePolicy>,
    #[serde(default)]
    pub issue_log: Option<PathBuf>,
    #[serde(default)]
    pub include_line_text_in_log: Option<bool>,
    #[serde(default)]
    pub round_trip_sample_per_bucket: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationMode {
    Report,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LinePolicy {
    FixOrSkip,
}

#[derive(Debug, Default)]
struct ConfigBuilder {
    canonicalization: PartialCanonicalizationConfig,
    balancing: PartialBalancingConfig,
    sentencepiece: PartialSentencePieceConfig,
    validation: PartialValidationConfig,
}

pub fn load_effective_config(
    config_path: &Path,
    preset_path: &Path,
) -> Result<EffectiveConfig, ConfigError> {
    let raw = read_json::<RawConfig>(config_path)?;
    let catalog = read_json::<PresetCatalog>(preset_path)?;
    expand_config(raw, &catalog)
}

pub fn expand_config(
    raw: RawConfig,
    catalog: &PresetCatalog,
) -> Result<EffectiveConfig, ConfigError> {
    let preset_name = raw
        .preset
        .clone()
        .unwrap_or_else(|| catalog.default.clone());
    let mut builder = ConfigBuilder::default();
    let mut stack = HashSet::new();

    apply_preset(&preset_name, catalog, &mut stack, &mut builder)?;
    builder.apply_raw_overrides(&raw);

    builder.into_effective(preset_name, raw.corpus, raw.output)
}

fn read_json<T>(path: &Path) -> Result<T, ConfigError>
where
    T: for<'de> Deserialize<'de>,
{
    let contents = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut deserializer = serde_json::Deserializer::from_str(&contents);
    serde_path_to_error::deserialize(&mut deserializer).map_err(|error| ConfigError::Parse {
        path: path.to_path_buf(),
        json_path: error.path().to_string(),
        source: error.into_inner(),
    })
}

fn apply_preset(
    name: &str,
    catalog: &PresetCatalog,
    stack: &mut HashSet<String>,
    builder: &mut ConfigBuilder,
) -> Result<(), ConfigError> {
    if !stack.insert(name.to_owned()) {
        return Err(ConfigError::PresetCycle(name.to_owned()));
    }

    let preset = catalog
        .presets
        .get(name)
        .ok_or_else(|| ConfigError::UnknownPreset(name.to_owned()))?;

    if let Some(parent) = &preset.extends {
        apply_preset(parent, catalog, stack, builder)?;
    }

    builder.apply_preset(preset);
    stack.remove(name);
    Ok(())
}

impl ConfigBuilder {
    fn apply_preset(&mut self, preset: &PresetConfig) {
        self.canonicalization.merge(&preset.canonicalization);
        self.balancing.merge(&preset.balancing);
        self.sentencepiece.merge(&preset.sentencepiece);
        self.validation.merge(&preset.validation);
    }

    fn apply_raw_overrides(&mut self, raw: &RawConfig) {
        self.canonicalization.merge(&raw.canonicalization);
        self.balancing.merge(&raw.balancing);
        self.sentencepiece.merge(&raw.sentencepiece);
        self.validation.merge(&raw.validation);
    }

    fn into_effective(
        self,
        preset: String,
        corpus: CorpusConfig,
        output: OutputConfig,
    ) -> Result<EffectiveConfig, ConfigError> {
        Ok(EffectiveConfig {
            preset,
            corpus,
            output,
            canonicalization: self.canonicalization.require()?,
            balancing: self.balancing.require()?,
            sentencepiece: self.sentencepiece.require()?,
            validation: self.validation.require()?,
        })
    }
}

impl PartialCanonicalizationConfig {
    fn merge(&mut self, other: &Self) {
        replace_if_some(&mut self.unicode_form, other.unicode_form);
        replace_if_some(&mut self.strip, other.strip.clone());
        replace_if_some(&mut self.map_nbsp_to_space, other.map_nbsp_to_space);
        replace_if_some(
            &mut self.fold_arabic_presentation_forms,
            other.fold_arabic_presentation_forms,
        );
        replace_if_some(&mut self.soft_hyphen, other.soft_hyphen);
        replace_if_some(&mut self.preserve_zwj_zwnj, other.preserve_zwj_zwnj);
        replace_if_some(
            &mut self.preserve_compatibility_chars,
            other.preserve_compatibility_chars,
        );
    }

    fn require(self) -> Result<CanonicalizationConfig, ConfigError> {
        Ok(CanonicalizationConfig {
            unicode_form: require_field(self.unicode_form, "canonicalization.unicode_form")?,
            strip: require_field(self.strip, "canonicalization.strip")?,
            map_nbsp_to_space: require_field(
                self.map_nbsp_to_space,
                "canonicalization.map_nbsp_to_space",
            )?,
            fold_arabic_presentation_forms: require_field(
                self.fold_arabic_presentation_forms,
                "canonicalization.fold_arabic_presentation_forms",
            )?,
            soft_hyphen: require_field(self.soft_hyphen, "canonicalization.soft_hyphen")?,
            preserve_zwj_zwnj: require_field(
                self.preserve_zwj_zwnj,
                "canonicalization.preserve_zwj_zwnj",
            )?,
            preserve_compatibility_chars: require_field(
                self.preserve_compatibility_chars,
                "canonicalization.preserve_compatibility_chars",
            )?,
        })
    }
}

impl PartialBalancingConfig {
    fn merge(&mut self, other: &Self) {
        replace_if_some(&mut self.enabled, other.enabled);
        replace_if_some(&mut self.mode, other.mode);
        replace_if_some(&mut self.total_lines, other.total_lines);
        replace_if_some(&mut self.alpha, other.alpha);
        replace_if_some(&mut self.hierarchy, other.hierarchy.clone());
        replace_if_some(&mut self.min_keep_fraction, other.min_keep_fraction);
        replace_if_some(&mut self.max_downsample_ratio, other.max_downsample_ratio);
        replace_if_some(
            &mut self.collapse_buckets_below_lines,
            other.collapse_buckets_below_lines,
        );
        replace_if_some(&mut self.max_part_lines, other.max_part_lines);
        replace_if_some(&mut self.shuffle_seed, other.shuffle_seed);
    }

    fn require(self) -> Result<BalancingConfig, ConfigError> {
        Ok(BalancingConfig {
            enabled: require_field(self.enabled, "balancing.enabled")?,
            mode: require_field(self.mode, "balancing.mode")?,
            total_lines: require_field(self.total_lines, "balancing.total_lines")?,
            alpha: require_field(self.alpha, "balancing.alpha")?,
            hierarchy: require_field(self.hierarchy, "balancing.hierarchy")?,
            min_keep_fraction: require_field(
                self.min_keep_fraction,
                "balancing.min_keep_fraction",
            )?,
            max_downsample_ratio: require_field(
                self.max_downsample_ratio,
                "balancing.max_downsample_ratio",
            )?,
            collapse_buckets_below_lines: require_field(
                self.collapse_buckets_below_lines,
                "balancing.collapse_buckets_below_lines",
            )?,
            max_part_lines: require_field(self.max_part_lines, "balancing.max_part_lines")?,
            shuffle_seed: require_field(self.shuffle_seed, "balancing.shuffle_seed")?,
        })
    }
}

impl PartialSentencePieceConfig {
    fn merge(&mut self, other: &Self) {
        replace_if_some(&mut self.trainer, other.trainer);
        replace_if_some(&mut self.python, other.python.clone());
        replace_if_some(&mut self.model_type, other.model_type);
        replace_if_some(&mut self.vocab_size, other.vocab_size);
        replace_if_some(&mut self.character_coverage, other.character_coverage);
        replace_if_some(&mut self.byte_fallback, other.byte_fallback);
        replace_if_some(
            &mut self.normalization_rule_name,
            other.normalization_rule_name,
        );
        replace_if_some(&mut self.add_dummy_prefix, other.add_dummy_prefix);
        replace_if_some(
            &mut self.remove_extra_whitespaces,
            other.remove_extra_whitespaces,
        );
        replace_if_some(
            &mut self.split_by_unicode_script,
            other.split_by_unicode_script,
        );
        replace_if_some(&mut self.split_by_whitespace, other.split_by_whitespace);
        replace_if_some(&mut self.split_digits, other.split_digits);
        replace_if_some(
            &mut self.max_sentencepiece_length,
            other.max_sentencepiece_length,
        );
        replace_if_some(&mut self.max_sentence_length, other.max_sentence_length);
        replace_if_some(&mut self.input_sentence_size, other.input_sentence_size);
        replace_if_some(
            &mut self.shuffle_input_sentence,
            other.shuffle_input_sentence,
        );
        replace_if_some(
            &mut self.train_extremely_large_corpus,
            other.train_extremely_large_corpus,
        );
        replace_if_some(
            &mut self.user_defined_symbols,
            other.user_defined_symbols.clone(),
        );
        replace_if_some(&mut self.num_threads, other.num_threads);
    }

    fn require(self) -> Result<SentencePieceConfig, ConfigError> {
        Ok(SentencePieceConfig {
            trainer: require_field(self.trainer, "sentencepiece.trainer")?,
            python: require_field(self.python, "sentencepiece.python")?,
            model_type: require_field(self.model_type, "sentencepiece.model_type")?,
            vocab_size: require_field(self.vocab_size, "sentencepiece.vocab_size")?,
            character_coverage: require_field(
                self.character_coverage,
                "sentencepiece.character_coverage",
            )?,
            byte_fallback: require_field(self.byte_fallback, "sentencepiece.byte_fallback")?,
            normalization_rule_name: require_field(
                self.normalization_rule_name,
                "sentencepiece.normalization_rule_name",
            )?,
            add_dummy_prefix: require_field(
                self.add_dummy_prefix,
                "sentencepiece.add_dummy_prefix",
            )?,
            remove_extra_whitespaces: require_field(
                self.remove_extra_whitespaces,
                "sentencepiece.remove_extra_whitespaces",
            )?,
            split_by_unicode_script: require_field(
                self.split_by_unicode_script,
                "sentencepiece.split_by_unicode_script",
            )?,
            split_by_whitespace: require_field(
                self.split_by_whitespace,
                "sentencepiece.split_by_whitespace",
            )?,
            split_digits: require_field(self.split_digits, "sentencepiece.split_digits")?,
            max_sentencepiece_length: require_field(
                self.max_sentencepiece_length,
                "sentencepiece.max_sentencepiece_length",
            )?,
            max_sentence_length: require_field(
                self.max_sentence_length,
                "sentencepiece.max_sentence_length",
            )?,
            input_sentence_size: require_field(
                self.input_sentence_size,
                "sentencepiece.input_sentence_size",
            )?,
            shuffle_input_sentence: require_field(
                self.shuffle_input_sentence,
                "sentencepiece.shuffle_input_sentence",
            )?,
            train_extremely_large_corpus: require_field(
                self.train_extremely_large_corpus,
                "sentencepiece.train_extremely_large_corpus",
            )?,
            user_defined_symbols: require_field(
                self.user_defined_symbols,
                "sentencepiece.user_defined_symbols",
            )?,
            num_threads: require_field(self.num_threads, "sentencepiece.num_threads")?,
        })
    }
}

impl PartialValidationConfig {
    fn merge(&mut self, other: &Self) {
        replace_if_some(&mut self.mode, other.mode);
        replace_if_some(&mut self.line_policy, other.line_policy);
        replace_if_some(&mut self.issue_log, other.issue_log.clone());
        replace_if_some(
            &mut self.include_line_text_in_log,
            other.include_line_text_in_log,
        );
        replace_if_some(
            &mut self.round_trip_sample_per_bucket,
            other.round_trip_sample_per_bucket,
        );
    }

    fn require(self) -> Result<ValidationConfig, ConfigError> {
        Ok(ValidationConfig {
            mode: require_field(self.mode, "validation.mode")?,
            line_policy: require_field(self.line_policy, "validation.line_policy")?,
            issue_log: require_field(self.issue_log, "validation.issue_log")?,
            include_line_text_in_log: require_field(
                self.include_line_text_in_log,
                "validation.include_line_text_in_log",
            )?,
            round_trip_sample_per_bucket: require_field(
                self.round_trip_sample_per_bucket,
                "validation.round_trip_sample_per_bucket",
            )?,
        })
    }
}

fn replace_if_some<T>(target: &mut Option<T>, value: Option<T>) {
    if value.is_some() {
        *target = value;
    }
}

fn require_field<T>(value: Option<T>, field: &'static str) -> Result<T, ConfigError> {
    value.ok_or(ConfigError::MissingPresetField(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> PresetCatalog {
        serde_json::from_str(include_str!("../cfg.json.ocr")).expect("preset fixture is valid")
    }

    #[test]
    fn expands_default_preset() {
        let raw = RawConfig {
            preset: None,
            corpus: CorpusConfig {
                path: "data/raw-corpus".into(),
            },
            output: OutputConfig {
                work_dir: "runs/ocr-spm-v1".into(),
                model_prefix: "ocr_tokenizer".to_owned(),
            },
            canonicalization: PartialCanonicalizationConfig::default(),
            balancing: PartialBalancingConfig::default(),
            sentencepiece: PartialSentencePieceConfig::default(),
            validation: PartialValidationConfig::default(),
        };

        let effective = expand_config(raw, &catalog()).expect("default preset expands");

        assert_eq!(effective.preset, "ocr_multilingual");
        assert_eq!(effective.sentencepiece.model_type, ModelType::Bpe);
        assert!(effective.sentencepiece.byte_fallback);
        assert_eq!(effective.validation.line_policy, LinePolicy::FixOrSkip);
    }

    #[test]
    fn child_preset_replaces_list_values() {
        let raw = RawConfig {
            preset: Some("ocr_math_heavy".to_owned()),
            corpus: CorpusConfig {
                path: "data/raw-corpus".into(),
            },
            output: OutputConfig {
                work_dir: "runs/ocr-spm-v1".into(),
                model_prefix: "ocr_tokenizer".to_owned(),
            },
            canonicalization: PartialCanonicalizationConfig::default(),
            balancing: PartialBalancingConfig::default(),
            sentencepiece: PartialSentencePieceConfig::default(),
            validation: PartialValidationConfig::default(),
        };

        let effective = expand_config(raw, &catalog()).expect("child preset expands");

        assert_eq!(effective.sentencepiece.vocab_size, 48_000);
        assert!(effective
            .sentencepiece
            .user_defined_symbols
            .contains(&"\\operatorname".to_owned()));
    }

    #[test]
    fn user_overrides_preset_field() {
        let raw = RawConfig {
            preset: Some("ocr_cjk_heavy".to_owned()),
            corpus: CorpusConfig {
                path: "data/raw-corpus".into(),
            },
            output: OutputConfig {
                work_dir: "runs/ocr-spm-v1".into(),
                model_prefix: "ocr_tokenizer".to_owned(),
            },
            canonicalization: PartialCanonicalizationConfig::default(),
            balancing: PartialBalancingConfig::default(),
            sentencepiece: PartialSentencePieceConfig {
                vocab_size: Some(32_000),
                ..PartialSentencePieceConfig::default()
            },
            validation: PartialValidationConfig::default(),
        };

        let effective = expand_config(raw, &catalog()).expect("user override expands");

        assert_eq!(effective.sentencepiece.vocab_size, 32_000);
        assert_eq!(effective.sentencepiece.max_sentencepiece_length, 4);
    }

    #[test]
    fn checked_in_default_config_loads() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

        let effective = load_effective_config(&root.join("cfg.json"), &root.join("cfg.json.ocr"))
            .expect("checked-in cfg.json loads with preset catalog");

        assert_eq!(effective.preset, "ocr_multilingual");
        assert_eq!(effective.corpus.path, PathBuf::from("data/raw-corpus"));
        assert_eq!(effective.output.model_prefix, "ocr_tokenizer");
        assert!(effective.sentencepiece.byte_fallback);
    }
}
