//! Turning a [`Report`] into something to read or to pipe.
//!
//! The only module that decides how findings look. Everything upstream returns data, which is
//! what lets the corpus scan and the model checklist render, rank and exit identically.

use crate::report::{Remedy, Report, Status};

pub fn as_json(report: &Report) -> String {
    let document = serde_json::json!({
        "ok": report.ok(),
        "worst_severity": report.worst_severity(),
        "remedies": report.remedies(),
        "results": report.ranked(),
    });
    serde_json::to_string_pretty(&document).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
}

pub fn as_text(report: &Report) -> String {
    let mut lines: Vec<String> = report.ranked().iter().map(|f| format_finding(f)).collect();

    lines.push(String::new());
    lines.push(format!(
        "{} passed, {} failed, {} skipped",
        report.count(Status::Passed),
        report.count(Status::Failed),
        report.count(Status::Skipped),
    ));
    lines.extend(next_steps(report));
    lines.join("\n")
}

fn format_finding(finding: &crate::report::Finding) -> String {
    let mark = match finding.status {
        Status::Passed => "PASS",
        Status::Failed => "FAIL",
        Status::Skipped => "SKIP",
    };
    // A passing check has no severity worth showing; a failing one leads with it.
    let tag = match finding.status {
        Status::Passed => String::new(),
        _ => format!(" [{}]", severity_word(finding.severity)),
    };

    let mut out = format!("{mark}{tag}  {}: {}", finding.check, finding.summary);
    for item in &finding.evidence {
        out.push_str(&format!("\n        {item}"));
    }
    out
}

fn severity_word(severity: crate::report::Severity) -> &'static str {
    use crate::report::Severity::*;
    match severity {
        Blocker => "blocker",
        High => "high",
        Medium => "medium",
        Info => "info",
    }
}

fn next_steps(report: &Report) -> Vec<String> {
    let remedies: Vec<Remedy> = report
        .remedies()
        .into_iter()
        .filter(|r| *r != Remedy::NotApplicable)
        .collect();

    if remedies.is_empty() {
        return Vec::new();
    }

    let mut lines = vec![String::new(), "Next:".to_string()];
    for (index, remedy) in remedies.iter().enumerate() {
        lines.push(format!("  {}. {}", index + 1, remedy.next_step()));
    }
    if remedies.contains(&Remedy::FixCorpus) && remedies.contains(&Remedy::RetrainConfig) {
        lines.push("  (in that order — a corpus defect survives any number of retrains)".into());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{Finding, Severity};

    #[test]
    fn json_output_is_parseable() {
        let report = Report::new(vec![
            Finding::failed("a", "broke").graded(Severity::Blocker, Remedy::FixCorpus),
        ]);
        let parsed: serde_json::Value = serde_json::from_str(&as_json(&report)).unwrap();

        assert_eq!(parsed["ok"], false);
        assert_eq!(parsed["worst_severity"], "blocker");
        assert_eq!(parsed["results"][0]["check"], "a");
    }

    #[test]
    fn text_output_leads_with_failures_and_names_severity() {
        let report = Report::new(vec![
            Finding::passed("clean", "fine"),
            Finding::failed("broken", "bad").graded(Severity::Blocker, Remedy::FixCorpus),
        ]);
        let text = as_text(&report);

        assert!(
            text.starts_with("FAIL [blocker]  broken: bad"),
            "got:\n{text}"
        );
        assert!(text.contains("1 passed, 1 failed, 0 skipped"));
    }

    #[test]
    fn corpus_fixes_are_ordered_before_retrains() {
        let report = Report::new(vec![
            Finding::failed("cfg", "x").graded(Severity::High, Remedy::RetrainConfig),
            Finding::failed("data", "y").graded(Severity::Blocker, Remedy::FixCorpus),
        ]);
        let text = as_text(&report);
        let corpus = text.find("canonicalize the corpus").unwrap();
        let retrain = text.find("change the trainer flags").unwrap();

        assert!(corpus < retrain);
        assert!(text.contains("survives any number of retrains"));
    }

    #[test]
    fn a_clean_report_suggests_nothing() {
        let report = Report::new(vec![Finding::passed("a", "fine")]);
        assert!(!as_text(&report).contains("Next:"));
    }
}
