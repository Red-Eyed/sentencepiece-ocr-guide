//! What a run found, and how much you should care.
//!
//! Three outcomes rather than two. A check that could not run must never report success, so
//! [`Status::Skipped`] carries the reason in the same field a failure carries its evidence.
//! [`Status`] says whether something failed, [`Severity`] whether it matters, and [`Remedy`]
//! what to do next — the two questions a bare pass/fail list leaves open.
//!
//! Both halves of the tool produce these, so a corpus finding and a model finding rank, read
//! and exit identically.

use serde::Serialize;

/// The most evidence lines any single finding will carry.
///
/// Evidence exists to point at where to look, not to reproduce the defect in full: a check that
/// dumps ten thousand offending pieces is one nobody reads.
pub const MAX_EVIDENCE: usize = 5;

/// Whether a check failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Declared first so the derived ordering reads failures before anything else.
    Failed,
    Skipped,
    Passed,
}

/// How much a failure matters.
///
/// Declared worst-first, so the derived [`Ord`] *is* the ranking — `max()` over severities
/// yields the most serious one with no lookup table to keep in step with the variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// The label space is permanently broken. Do not spend training compute.
    Blocker,
    /// Measurable accuracy loss, but the system will function.
    High,
    /// Efficiency or vocabulary geometry — worth fixing, not worth blocking on.
    Medium,
    /// Reported for visibility. Never a defect: a non-zero count here is expected.
    Info,
}

/// What actually fixes a failure.
///
/// The distinction a per-artifact checklist cannot express: a *model* check can demand a
/// corpus fix, because retraining alone reproduces a defect that lives in the data.
/// Declared in the order the fixes must be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Remedy {
    /// Fix the data, *then* retrain. Retraining alone reproduces the failure.
    FixCorpus,
    /// Change a trainer flag and retrain. Cheap.
    RetrainConfig,
    /// Neither the tokenizer nor the corpus — the wiring around them.
    FixIntegration,
    /// Carried by `Info` findings, which report a measurement rather than a defect.
    NotApplicable,
}

impl Remedy {
    /// The instruction a reader should act on.
    pub fn next_step(self) -> &'static str {
        match self {
            Remedy::FixCorpus => "canonicalize the corpus, then retrain",
            Remedy::RetrainConfig => "change the trainer flags and retrain",
            Remedy::FixIntegration => "fix the wiring between tokenizer and checkpoint",
            Remedy::NotApplicable => "nothing — reported for visibility",
        }
    }
}

/// Which numbered failure mode in `docs/09-failure-modes.md` a finding is about.
///
/// A number rather than a free-text citation, so a finding cannot cite a page that does not
/// exist or drift out of step with the guide's wording. The table there carries the same numbers
/// and names the check that covers each one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FailureMode(pub u8);

impl FailureMode {
    /// The highest mode the guide defines. Anything above it is a typo rather than a reference.
    pub const HIGHEST: u8 = 21;

    pub fn citation(self) -> String {
        format!("failure mode #{} — docs/09-failure-modes.md", self.0)
    }
}

/// The outcome of one check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    pub check: String,
    pub status: Status,
    pub summary: String,
    pub evidence: Vec<String>,
    pub severity: Severity,
    pub remedy: Remedy,
    /// The guide entry this finding is about, when there is one.
    pub failure_mode: Option<FailureMode>,
}

impl Finding {
    /// A check that ran and found nothing wrong.
    ///
    /// Severity and remedy still describe what a *failure* of this check would have meant, so
    /// they are supplied by the check rather than defaulted here.
    pub fn passed(check: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            status: Status::Passed,
            summary: summary.into(),
            evidence: Vec::new(),
            severity: Severity::Info,
            remedy: Remedy::NotApplicable,
            failure_mode: None,
        }
    }

    /// A check that ran and found a defect. Evidence is truncated to [`MAX_EVIDENCE`].
    pub fn failed(check: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            status: Status::Failed,
            summary: summary.into(),
            evidence: Vec::new(),
            severity: Severity::High,
            remedy: Remedy::RetrainConfig,
            failure_mode: None,
        }
    }

    /// A check that could not run. `reason` travels with the finding.
    pub fn skipped(check: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            status: Status::Skipped,
            summary: reason.into(),
            evidence: Vec::new(),
            severity: Severity::Info,
            remedy: Remedy::NotApplicable,
            failure_mode: None,
        }
    }

    /// Attach evidence, truncated to what a reader will actually use.
    pub fn with_evidence(mut self, evidence: impl IntoIterator<Item = String>) -> Self {
        self.evidence = evidence.into_iter().take(MAX_EVIDENCE).collect();
        self
    }

    /// Stamp what this check's failure means. Applied by the runner, not by the check body,
    /// so a check states its verdict once rather than on every return path.
    pub fn graded(mut self, severity: Severity, remedy: Remedy) -> Self {
        // A skip keeps the grade as well as a failure. What a failure of this check *means* is a
        // property of the check, not of one run of it, and a skipped BLOCKER is precisely what a
        // reader must not mistake for a clean result. Only a pass carries nothing to act on.
        //
        // This changes what is displayed, never what exits non-zero: `ok`, `worst_severity` and
        // `remedies` all filter on failure, so a graded skip stays visible without blocking.
        if self.status != Status::Passed {
            self.severity = severity;
            self.remedy = remedy;
        }
        self
    }

    /// Cite the guide entry this check is about.
    ///
    /// Applies to passes as well as failures: knowing which failure mode a green line rules out
    /// is what makes the report readable as a checklist rather than a list of assertions.
    pub fn about(mut self, mode: u8) -> Self {
        debug_assert!(
            (1..=FailureMode::HIGHEST).contains(&mode),
            "failure mode {mode} is not in docs/09-failure-modes.md"
        );
        self.failure_mode = Some(FailureMode(mode));
        self
    }

    pub fn is_failure(&self) -> bool {
        self.status == Status::Failed
    }
}

/// Everything one run found.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub findings: Vec<Finding>,
}

impl Report {
    pub fn new(findings: Vec<Finding>) -> Self {
        Self { findings }
    }

    /// True when nothing failed. Skipped checks do not fail a run, but they stay visible.
    pub fn ok(&self) -> bool {
        !self.findings.iter().any(Finding::is_failure)
    }

    pub fn count(&self, status: Status) -> usize {
        self.findings.iter().filter(|f| f.status == status).count()
    }

    /// The severity of the most serious failure, or `Info` when nothing failed.
    ///
    /// `min` rather than `max` because [`Severity`] is declared worst-first.
    pub fn worst_severity(&self) -> Severity {
        self.findings
            .iter()
            .filter(|f| f.is_failure())
            .map(|f| f.severity)
            .min()
            .unwrap_or(Severity::Info)
    }

    /// Distinct remedies across failures, in the order they must be applied.
    pub fn remedies(&self) -> Vec<Remedy> {
        let mut remedies: Vec<Remedy> = self
            .findings
            .iter()
            .filter(|f| f.is_failure())
            .map(|f| f.remedy)
            .collect();
        remedies.sort_unstable();
        remedies.dedup();
        remedies
    }

    /// Failures worst-first, then skips, then passes — the order to read them in.
    pub fn ranked(&self) -> Vec<&Finding> {
        let mut ordered: Vec<&Finding> = self.findings.iter().collect();
        ordered.sort_by(|a, b| {
            (a.status, a.severity, &a.check).cmp(&(b.status, b.severity, &b.check))
        });
        ordered
    }

    /// Non-zero when any failure is at or above `fail_on`.
    ///
    /// `<=` reads backwards only until you recall the ordering is worst-first: `Blocker` is the
    /// smallest severity, so "at or above `fail_on`" is "at or before it".
    pub fn exit_code(&self, fail_on: Severity) -> i32 {
        if self.ok() {
            return 0;
        }
        i32::from(self.worst_severity() <= fail_on)
    }

    pub fn extend(&mut self, other: Report) {
        self.findings.extend(other.findings);
    }
}

impl FromIterator<Finding> for Report {
    fn from_iter<I: IntoIterator<Item = Finding>>(iter: I) -> Self {
        Self {
            findings: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure(severity: Severity, remedy: Remedy) -> Finding {
        Finding::failed("c", "s").graded(severity, remedy)
    }

    #[test]
    fn severity_orders_worst_first() {
        assert!(Severity::Blocker < Severity::High);
        assert!(Severity::High < Severity::Medium);
        assert!(Severity::Medium < Severity::Info);
    }

    #[test]
    fn worst_severity_ignores_passes() {
        let report = Report::new(vec![
            Finding::passed("a", "fine"),
            failure(Severity::Medium, Remedy::RetrainConfig),
            failure(Severity::Blocker, Remedy::FixCorpus),
        ]);
        assert_eq!(report.worst_severity(), Severity::Blocker);
    }

    #[test]
    fn a_clean_run_reports_info_and_exits_zero() {
        let report = Report::new(vec![Finding::passed("a", "fine")]);
        assert_eq!(report.worst_severity(), Severity::Info);
        assert_eq!(report.exit_code(Severity::Info), 0);
    }

    #[test]
    fn exit_code_respects_the_threshold() {
        let report = Report::new(vec![failure(Severity::Medium, Remedy::RetrainConfig)]);
        assert_eq!(report.exit_code(Severity::High), 0, "medium is below high");
        assert_eq!(report.exit_code(Severity::Medium), 1);
        assert_eq!(report.exit_code(Severity::Info), 1);
    }

    #[test]
    fn remedies_come_back_in_application_order() {
        let report = Report::new(vec![
            failure(Severity::High, Remedy::RetrainConfig),
            failure(Severity::Blocker, Remedy::FixCorpus),
            failure(Severity::High, Remedy::RetrainConfig),
        ]);
        assert_eq!(
            report.remedies(),
            vec![Remedy::FixCorpus, Remedy::RetrainConfig],
            "a corpus defect survives any number of retrains, so it is fixed first"
        );
    }

    #[test]
    fn ranking_reads_failures_then_skips_then_passes() {
        let report = Report::new(vec![
            Finding::passed("zzz", "fine"),
            Finding::skipped("mmm", "no samples"),
            failure(Severity::High, Remedy::RetrainConfig),
            failure(Severity::Blocker, Remedy::FixCorpus),
        ]);
        let order: Vec<Status> = report.ranked().iter().map(|f| f.status).collect();
        assert_eq!(
            order,
            vec![
                Status::Failed,
                Status::Failed,
                Status::Skipped,
                Status::Passed
            ]
        );
        assert_eq!(report.ranked()[0].severity, Severity::Blocker);
    }

    #[test]
    fn a_pass_never_carries_a_grade_to_act_on() {
        let passed = Finding::passed("c", "s").graded(Severity::Blocker, Remedy::FixCorpus);
        assert_eq!(passed.severity, Severity::Info);
        assert_eq!(passed.remedy, Remedy::NotApplicable);
    }

    #[test]
    fn a_skip_keeps_the_severity_of_the_check_it_stands_in_for() {
        // A skipped BLOCKER rendering as `SKIP [info]` is exactly the clean-looking report the
        // three-status split exists to prevent.
        let skipped =
            Finding::skipped("c", "no samples").graded(Severity::Blocker, Remedy::FixCorpus);
        assert_eq!(skipped.severity, Severity::Blocker);
        assert_eq!(skipped.remedy, Remedy::FixCorpus);
    }

    #[test]
    fn a_graded_skip_still_does_not_fail_the_run() {
        let report = Report::new(vec![
            Finding::skipped("c", "no samples").graded(Severity::Blocker, Remedy::FixCorpus),
        ]);
        assert!(report.ok(), "a skip is not a failure");
        assert_eq!(report.exit_code(Severity::Blocker), 0);
        assert_eq!(report.worst_severity(), Severity::Info);
    }

    #[test]
    fn evidence_is_truncated_to_what_gets_read() {
        let many = (0..50).map(|n| n.to_string());
        assert_eq!(
            Finding::failed("c", "s").with_evidence(many).evidence.len(),
            MAX_EVIDENCE
        );
    }
}
