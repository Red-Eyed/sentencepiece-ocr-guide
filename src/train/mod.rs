//! Streaming corpus preparation for SentencePiece training.
//!
//! The trainer itself is still the official `spm_train`; this module owns the part that would
//! otherwise force a second giant corpus file onto disk. It maps sources, counts bucket sizes,
//! assigns alpha-smoothed quotas, canonicalizes accepted lines, and feeds a bounded stream to a
//! writer.

use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::ops::Range;
use std::sync::mpsc::SyncSender;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::corpus::canonical::Canonicalizer;
use crate::corpus::scan::line_aligned_chunks;
use crate::corpus::source::Source;

/// A balancing bucket.
///
/// Math is deliberately above script in the hierarchy: a line full of LaTeX commands should not
/// spend the Latin text budget just because commands are ASCII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bucket {
    Math,
    Text(&'static str),
    Unclassified,
}

impl Bucket {
    pub fn label(self) -> String {
        match self {
            Bucket::Math => "math".to_string(),
            Bucket::Text(script) => format!("text:{script}"),
            Bucket::Unclassified => "unclassified".to_string(),
        }
    }
}

/// Training corpus preparation knobs.
#[derive(Debug, Clone)]
pub struct PrepareOptions {
    pub target_lines: u64,
    pub alpha: f64,
    pub seed: u64,
    pub max_line_bytes: usize,
    pub shuffle_buffer_lines: usize,
    pub memory_budget_bytes: u64,
    pub drop_invalid_utf8: bool,
    pub drop_long_lines: bool,
    pub math_policy: MathPolicy,
}

impl PrepareOptions {
    pub fn validate(&self) -> Result<()> {
        if self.target_lines == 0 {
            bail!("--lines must be greater than zero");
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            bail!("--alpha must be a finite positive number");
        }
        if self.max_line_bytes == 0 {
            bail!("--max-sentence-length must be greater than zero");
        }
        if self.shuffle_buffer_lines == 0 {
            bail!("--shuffle-buffer-lines must be greater than zero");
        }
        self.math_policy.validate()?;

        let worst_case =
            (self.shuffle_buffer_lines as u128).saturating_mul(self.max_line_bytes as u128);
        if worst_case > u128::from(self.memory_budget_bytes) {
            bail!(
                "shuffle buffer could hold up to {} bytes, above --memory-budget-gb",
                worst_case
            );
        }
        Ok(())
    }
}

/// How math-like lines affect balancing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MathPolicy {
    /// Report math presence, but balance by dominant writing system. The broad-OCR default.
    ReportOnly,
    /// Give math its own bucket, capped to this fraction of selected lines.
    Balanced { max_share: f64 },
}

impl MathPolicy {
    fn validate(self) -> Result<()> {
        match self {
            MathPolicy::ReportOnly => Ok(()),
            MathPolicy::Balanced { max_share } => {
                if !max_share.is_finite() || !(0.0..=1.0).contains(&max_share) {
                    bail!("--math-max-share must be between 0.0 and 1.0");
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CorpusPlan {
    pub chunks: Vec<ChunkPlan>,
    pub bucket_counts: BTreeMap<Bucket, u64>,
    pub bucket_quotas: BTreeMap<Bucket, u64>,
    pub eligible_lines: u64,
    pub selected_lines: u64,
    pub invalid_utf8: u64,
    pub long_lines: u64,
    pub math_lines: u64,
}

#[derive(Debug, Clone)]
pub struct ChunkPlan {
    pub source_index: usize,
    pub range: Range<usize>,
    pub counts: BTreeMap<Bucket, u64>,
    pub quotas: BTreeMap<Bucket, u64>,
}

#[derive(Debug, Clone, Default)]
struct ChunkStats {
    source_index: usize,
    range: Range<usize>,
    counts: BTreeMap<Bucket, u64>,
    invalid_utf8: u64,
    long_lines: u64,
    math_lines: u64,
}

#[cfg(test)]
impl ChunkStats {
    fn eligible_lines(&self) -> u64 {
        self.counts.values().sum()
    }
}

/// Count eligible lines and prepare exact per-chunk quotas.
pub fn plan_corpus(
    sources: &[Source],
    canonicalizer: &Canonicalizer,
    options: &PrepareOptions,
) -> Result<CorpusPlan> {
    options.validate()?;

    let mut stats = count_chunks(sources, canonicalizer, options)?;
    stats.sort_by(|a, b| {
        a.source_index
            .cmp(&b.source_index)
            .then_with(|| a.range.start.cmp(&b.range.start))
    });

    let mut bucket_counts = BTreeMap::new();
    let mut invalid_utf8 = 0;
    let mut long_lines = 0;
    let mut math_lines = 0;
    for chunk in &stats {
        invalid_utf8 += chunk.invalid_utf8;
        long_lines += chunk.long_lines;
        math_lines += chunk.math_lines;
        for (bucket, count) in &chunk.counts {
            *bucket_counts.entry(*bucket).or_default() += count;
        }
    }

    if invalid_utf8 > 0 && !options.drop_invalid_utf8 {
        bail!("{invalid_utf8} lines are not valid UTF-8; pass --drop-invalid to skip them");
    }
    if long_lines > 0 && !options.drop_long_lines {
        bail!(
            "{long_lines} lines exceed --max-sentence-length; pass --drop-long-lines to skip them"
        );
    }

    let eligible_lines = bucket_counts.values().sum();
    if eligible_lines == 0 {
        bail!("no eligible training lines found");
    }

    let bucket_quotas = bucket_quotas(&bucket_counts, options);
    let chunk_quotas = chunk_quotas(&stats, &bucket_quotas);
    let chunks = stats
        .into_iter()
        .enumerate()
        .map(|(index, stats)| ChunkPlan {
            source_index: stats.source_index,
            range: stats.range,
            counts: stats.counts,
            quotas: chunk_quotas.get(&index).cloned().unwrap_or_default(),
        })
        .collect();

    Ok(CorpusPlan {
        chunks,
        bucket_counts,
        selected_lines: bucket_quotas.values().sum(),
        bucket_quotas,
        eligible_lines,
        invalid_utf8,
        long_lines,
        math_lines,
    })
}

fn count_chunks(
    sources: &[Source],
    canonicalizer: &Canonicalizer,
    options: &PrepareOptions,
) -> Result<Vec<ChunkStats>> {
    let per_source = sources
        .par_iter()
        .enumerate()
        .map(|(source_index, source)| -> Result<Vec<ChunkStats>> {
            let mapped = source
                .map()
                .with_context(|| format!("mapping {}", source.label()))?;
            let bytes: &[u8] = &mapped;
            Ok(line_aligned_chunks(bytes)
                .into_par_iter()
                .map(|range| {
                    count_chunk(
                        source_index,
                        range.clone(),
                        bytes.get(range).unwrap_or_default(),
                        canonicalizer,
                        options,
                    )
                })
                .collect())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(per_source.into_iter().flatten().collect())
}

fn count_chunk(
    source_index: usize,
    range: Range<usize>,
    bytes: &[u8],
    canonicalizer: &Canonicalizer,
    options: &PrepareOptions,
) -> ChunkStats {
    let mut stats = ChunkStats {
        source_index,
        range,
        counts: BTreeMap::new(),
        invalid_utf8: 0,
        long_lines: 0,
        math_lines: 0,
    };

    for line in bytes.split(|&b| b == b'\n') {
        let Ok(text) = std::str::from_utf8(line) else {
            stats.invalid_utf8 += 1;
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }

        let canonical = canonicalizer.apply(text);
        if canonical.len() > options.max_line_bytes {
            stats.long_lines += 1;
            continue;
        }
        if is_math_line(&canonical) {
            stats.math_lines += 1;
        }
        *stats
            .counts
            .entry(classify_bucket(&canonical, options.math_policy))
            .or_default() += 1;
    }
    stats
}

/// Write a sampled, canonicalized, bounded-shuffled stream.
pub fn stream_prepared(
    sources: &[Source],
    plan: &CorpusPlan,
    canonicalizer: &Canonicalizer,
    options: &PrepareOptions,
    writer: &mut impl Write,
) -> Result<()> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(options.shuffle_buffer_lines);

    std::thread::scope(|scope| {
        let producer = scope.spawn(|| produce_lines(sources, plan, canonicalizer, options, sender));
        let write_result = shuffle_to_writer(receiver, options, writer);
        let produce_result = match producer.join() {
            Ok(result) => result,
            Err(_) => bail!("training producer thread panicked"),
        };
        write_result?;
        produce_result
    })
}

fn produce_lines(
    sources: &[Source],
    plan: &CorpusPlan,
    canonicalizer: &Canonicalizer,
    options: &PrepareOptions,
    sender: SyncSender<String>,
) -> Result<()> {
    sources
        .par_iter()
        .enumerate()
        .try_for_each(|(source_index, source)| -> Result<()> {
            let mapped = source
                .map()
                .with_context(|| format!("mapping {}", source.label()))?;
            let bytes: &[u8] = &mapped;
            let chunks: Vec<&ChunkPlan> = plan
                .chunks
                .iter()
                .filter(|chunk| chunk.source_index == source_index)
                .collect();

            chunks.into_par_iter().try_for_each(|chunk| {
                let data = bytes.get(chunk.range.clone()).unwrap_or_default();
                stream_chunk(data, chunk, canonicalizer, options, &sender)
            })
        })
}

fn stream_chunk(
    bytes: &[u8],
    chunk: &ChunkPlan,
    canonicalizer: &Canonicalizer,
    options: &PrepareOptions,
    sender: &SyncSender<String>,
) -> Result<()> {
    let mut remaining = chunk.counts.clone();
    let mut quotas = chunk.quotas.clone();
    let mut rng = SplitMix64::new(options.seed ^ chunk_seed(chunk));

    for line in bytes.split(|&b| b == b'\n') {
        let Ok(text) = std::str::from_utf8(line) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }

        let canonical = canonicalizer.apply(text);
        if canonical.len() > options.max_line_bytes {
            continue;
        }

        let bucket = classify_bucket(&canonical, options.math_policy);
        let Some(bucket_remaining) = remaining.get_mut(&bucket) else {
            continue;
        };
        let quota = quotas.entry(bucket).or_default();
        let take = should_take(&mut rng, *bucket_remaining, *quota);
        *bucket_remaining = bucket_remaining.saturating_sub(1);

        if take {
            *quota = quota.saturating_sub(1);
            sender
                .send(canonical.into_owned())
                .context("training writer stopped while producers were still running")?;
        }
    }
    Ok(())
}

fn should_take(rng: &mut SplitMix64, remaining: u64, quota: u64) -> bool {
    if quota == 0 || remaining == 0 {
        return false;
    }
    if quota >= remaining {
        return true;
    }
    rng.next_below(remaining) < quota
}

fn shuffle_to_writer(
    receiver: std::sync::mpsc::Receiver<String>,
    options: &PrepareOptions,
    writer: &mut impl Write,
) -> Result<()> {
    let mut rng = SplitMix64::new(options.seed ^ 0xD1CE_5EED);
    let mut buffer = Vec::with_capacity(options.shuffle_buffer_lines);

    for line in receiver {
        buffer.push(line);
        if buffer.len() >= options.shuffle_buffer_lines {
            write_random_line(&mut buffer, &mut rng, writer)?;
        }
    }

    while !buffer.is_empty() {
        write_random_line(&mut buffer, &mut rng, writer)?;
    }
    Ok(())
}

fn write_random_line(
    buffer: &mut Vec<String>,
    rng: &mut SplitMix64,
    writer: &mut impl Write,
) -> Result<()> {
    let index = rng.next_index(buffer.len());
    let line = buffer.swap_remove(index);
    writer.write_all(line.as_bytes())?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn classify_bucket(line: &str, math_policy: MathPolicy) -> Bucket {
    if matches!(math_policy, MathPolicy::Balanced { .. }) && is_math_line(line) {
        return Bucket::Math;
    }

    let mut counts: Vec<(&'static str, u64)> = Vec::new();
    for writing in line.chars().filter_map(crate::writing::writing_of) {
        let name = writing.name();
        match counts.iter_mut().find(|(seen, _)| *seen == name) {
            Some((_, count)) => *count += 1,
            None => counts.push((name, 1)),
        }
    }

    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map_or(Bucket::Unclassified, |(name, _)| Bucket::Text(name))
}

fn is_math_line(line: &str) -> bool {
    if line.contains("\\begin{") || line.contains("\\end{") {
        return true;
    }

    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek().is_some_and(|next| next.is_ascii_alphabetic()) {
            return true;
        }
    }

    line.chars()
        .any(|c| matches!(c, '^' | '_' | '∑' | '∫' | '√' | '≤' | '≥' | '≠' | '≈'))
}

fn alpha_quotas(
    counts: &BTreeMap<Bucket, u64>,
    target_lines: u64,
    alpha: f64,
) -> BTreeMap<Bucket, u64> {
    let mut quota = BTreeMap::new();
    let mut remaining_target = target_lines.min(counts.values().sum());
    let mut open: BTreeMap<Bucket, u64> = counts.clone();

    loop {
        if open.is_empty() || remaining_target == 0 {
            return quota;
        }

        let weights = alpha_weights(&open, alpha);
        let mut fixed = Vec::new();
        for (bucket, count) in &open {
            let ideal = ideal_quota(*bucket, remaining_target, &weights);
            if ideal >= *count as f64 {
                quota.insert(*bucket, *count);
                remaining_target = remaining_target.saturating_sub(*count);
                fixed.push(*bucket);
            }
        }

        if fixed.is_empty() {
            quota.extend(rounded_quotas(&open, remaining_target, &weights));
            return quota;
        }

        for bucket in fixed {
            open.remove(&bucket);
        }
    }
}

fn bucket_quotas(
    counts: &BTreeMap<Bucket, u64>,
    options: &PrepareOptions,
) -> BTreeMap<Bucket, u64> {
    let mut quotas = alpha_quotas(counts, options.target_lines, options.alpha);
    let MathPolicy::Balanced { max_share } = options.math_policy else {
        return quotas;
    };

    let selected = quotas.values().sum::<u64>();
    let max_math = (selected as f64 * max_share).floor() as u64;
    let Some(math_quota) = quotas.get_mut(&Bucket::Math) else {
        return quotas;
    };
    if *math_quota > max_math {
        *math_quota = max_math;
    }
    quotas
}

fn alpha_weights(counts: &BTreeMap<Bucket, u64>, alpha: f64) -> BTreeMap<Bucket, f64> {
    counts
        .iter()
        .map(|(bucket, count)| (*bucket, (*count as f64).powf(alpha)))
        .collect()
}

fn ideal_quota(bucket: Bucket, target: u64, weights: &BTreeMap<Bucket, f64>) -> f64 {
    let total: f64 = weights.values().sum();
    if total == 0.0 {
        return 0.0;
    }
    weights.get(&bucket).copied().unwrap_or(0.0) * target as f64 / total
}

fn rounded_quotas(
    counts: &BTreeMap<Bucket, u64>,
    target: u64,
    weights: &BTreeMap<Bucket, f64>,
) -> BTreeMap<Bucket, u64> {
    let mut quota = BTreeMap::new();
    let mut remainders = Vec::new();

    for (bucket, count) in counts {
        let ideal = ideal_quota(*bucket, target, weights);
        let floor = (ideal.floor() as u64).min(*count);
        quota.insert(*bucket, floor);
        remainders.push((*bucket, ideal - floor as f64));
    }

    let assigned: u64 = quota.values().sum();
    let mut left = target.saturating_sub(assigned);
    remainders.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut queue: VecDeque<Bucket> = remainders.into_iter().map(|(bucket, _)| bucket).collect();
    while left > 0 {
        let Some(bucket) = queue.pop_front() else {
            break;
        };
        let current = quota.get(&bucket).copied().unwrap_or(0);
        let cap = counts.get(&bucket).copied().unwrap_or(0);
        if current < cap {
            quota.insert(bucket, current + 1);
            left -= 1;
        }
        if current + 1 < cap {
            queue.push_back(bucket);
        }
    }

    quota
}

fn chunk_quotas(
    chunks: &[ChunkStats],
    bucket_quotas: &BTreeMap<Bucket, u64>,
) -> BTreeMap<usize, BTreeMap<Bucket, u64>> {
    let mut by_chunk: BTreeMap<usize, BTreeMap<Bucket, u64>> = BTreeMap::new();

    for (bucket, quota) in bucket_quotas {
        let counts: BTreeMap<usize, u64> = chunks
            .iter()
            .enumerate()
            .filter_map(|(index, chunk)| {
                chunk
                    .counts
                    .get(bucket)
                    .copied()
                    .filter(|count| *count > 0)
                    .map(|count| (index, count))
            })
            .collect();
        for (chunk_index, chunk_quota) in allocate_exact(&counts, *quota) {
            by_chunk
                .entry(chunk_index)
                .or_default()
                .insert(*bucket, chunk_quota);
        }
    }

    by_chunk
}

fn allocate_exact(counts: &BTreeMap<usize, u64>, target: u64) -> BTreeMap<usize, u64> {
    let total: u64 = counts.values().sum();
    if total == 0 || target == 0 {
        return BTreeMap::new();
    }

    let mut quota = BTreeMap::new();
    let mut remainders = Vec::new();
    for (key, count) in counts {
        let numerator = u128::from(target).saturating_mul(u128::from(*count));
        let floor = (numerator / u128::from(total)) as u64;
        let remainder = numerator % u128::from(total);
        quota.insert(*key, floor.min(*count));
        remainders.push((*key, remainder));
    }

    let assigned: u64 = quota.values().sum();
    let mut left = target.saturating_sub(assigned);
    remainders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (key, _) in remainders {
        if left == 0 {
            break;
        }
        let current = quota.get(&key).copied().unwrap_or(0);
        let cap = counts.get(&key).copied().unwrap_or(0);
        if current < cap {
            quota.insert(key, current + 1);
            left -= 1;
        }
    }

    quota
}

fn chunk_seed(chunk: &ChunkPlan) -> u64 {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    seed ^= (chunk.source_index as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    seed ^= (chunk.range.start as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    seed
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_below(&mut self, upper: u64) -> u64 {
        if upper == 0 {
            return 0;
        }
        self.next() % upper
    }

    fn next_index(&mut self, len: usize) -> usize {
        let upper = u64::try_from(len).unwrap_or(u64::MAX);
        usize::try_from(self.next_below(upper)).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::axis::default_axes;

    fn canon() -> Canonicalizer {
        Canonicalizer::new(default_axes(), &[]).unwrap()
    }

    #[test]
    fn latex_lines_are_bucketed_before_script() {
        assert_eq!(
            classify_bucket(r"\frac{x}{y}", MathPolicy::Balanced { max_share: 0.1 }),
            Bucket::Math
        );
        assert_eq!(
            classify_bucket(r"\frac{x}{y}", MathPolicy::ReportOnly),
            Bucket::Text("Latin")
        );
    }

    #[test]
    fn text_lines_use_the_dominant_writing_system() {
        assert_eq!(
            classify_bucket("hello world Пр", MathPolicy::ReportOnly),
            Bucket::Text("Latin")
        );
        assert_eq!(
            classify_bucket("Привіт hello", MathPolicy::ReportOnly),
            Bucket::Text("Cyrillic")
        );
    }

    #[test]
    fn alpha_quotas_keep_small_buckets_whole() {
        let counts = BTreeMap::from([
            (Bucket::Text("Latin"), 10_000),
            (Bucket::Text("Devanagari"), 10),
        ]);

        let quotas = alpha_quotas(&counts, 1_000, 0.3);

        assert_eq!(quotas.get(&Bucket::Text("Devanagari")), Some(&10));
        assert_eq!(quotas.values().sum::<u64>(), 1_000);
    }

    #[test]
    fn chunk_quota_allocation_is_exact() {
        let counts = BTreeMap::from([(0, 5), (1, 5), (2, 10)]);
        let quotas = allocate_exact(&counts, 7);

        assert_eq!(quotas.values().sum::<u64>(), 7);
        assert!(quotas.values().all(|quota| *quota <= 10));
    }

    #[test]
    fn streaming_sampler_takes_the_exact_chunk_quota() {
        let bytes = b"one\ntwo\nthree\nfour\n";
        let chunk = ChunkPlan {
            source_index: 0,
            range: 0..bytes.len(),
            counts: BTreeMap::from([(Bucket::Text("Latin"), 4)]),
            quotas: BTreeMap::from([(Bucket::Text("Latin"), 2)]),
        };
        let options = PrepareOptions {
            target_lines: 2,
            alpha: 0.3,
            seed: 7,
            max_line_bytes: 128,
            shuffle_buffer_lines: 4,
            memory_budget_bytes: 1024,
            drop_invalid_utf8: false,
            drop_long_lines: false,
            math_policy: MathPolicy::ReportOnly,
        };
        let (sender, receiver) = std::sync::mpsc::sync_channel(4);

        stream_chunk(bytes, &chunk, &canon(), &options, &sender).unwrap();
        drop(sender);

        assert_eq!(receiver.iter().count(), 2);
    }

    #[test]
    fn canonicalization_happens_before_counting() {
        let options = PrepareOptions {
            target_lines: 1,
            alpha: 0.3,
            seed: 7,
            max_line_bytes: 5,
            shuffle_buffer_lines: 1,
            memory_budget_bytes: 16,
            drop_invalid_utf8: false,
            drop_long_lines: false,
            math_policy: MathPolicy::ReportOnly,
        };
        let stats = count_chunk(
            0,
            0.."cafe\u{0301}\n".len(),
            "cafe\u{0301}\n".as_bytes(),
            &canon(),
            &options,
        );

        assert_eq!(
            stats.long_lines, 0,
            "NFC shortens this line before limit checks"
        );
        assert_eq!(stats.eligible_lines(), 1);
    }

    #[test]
    fn balanced_math_can_be_capped() {
        let counts = BTreeMap::from([(Bucket::Math, 100), (Bucket::Text("Latin"), 100)]);
        let options = PrepareOptions {
            target_lines: 100,
            alpha: 0.3,
            seed: 7,
            max_line_bytes: 128,
            shuffle_buffer_lines: 4,
            memory_budget_bytes: 1024,
            drop_invalid_utf8: false,
            drop_long_lines: false,
            math_policy: MathPolicy::Balanced { max_share: 0.1 },
        };

        let quotas = bucket_quotas(&counts, &options);

        assert_eq!(quotas.get(&Bucket::Math), Some(&10));
    }
}
