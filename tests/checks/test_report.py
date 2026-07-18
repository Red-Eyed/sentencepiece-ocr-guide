from sentencepiece_ocr_guide.checks.result import (
    MAX_EVIDENCE_ITEMS,
    Check,
    CheckResult,
    Remedy,
    Report,
    Severity,
    Status,
)
from sentencepiece_ocr_guide.checks.runner import run_checks
from tests.checks.conftest import FakeTokenizer


def test_report_is_ok_when_nothing_failed() -> None:
    report = Report(
        results=(
            CheckResult.passed("a", "fine"),
            CheckResult.skipped("b", "no samples supplied"),
        )
    )

    assert report.ok


def test_a_skipped_check_does_not_fail_the_run_but_stays_visible() -> None:
    report = Report(results=(CheckResult.skipped("b", "no samples supplied"),))

    assert report.ok
    assert len(report.with_status(Status.SKIPPED)) == 1


def test_report_is_not_ok_when_any_check_failed() -> None:
    report = Report(results=(CheckResult.passed("a", "fine"), CheckResult.failed("b", "broken")))

    assert not report.ok


def test_skipped_result_carries_its_reason() -> None:
    result = CheckResult.skipped("b", "no protected symbols supplied")

    assert result.status is Status.SKIPPED
    assert "no protected symbols supplied" in result.summary


def test_evidence_is_truncated_to_keep_output_readable() -> None:
    result = CheckResult.failed("a", "many", [str(index) for index in range(50)])

    assert len(result.evidence) == MAX_EVIDENCE_ITEMS


def _passing_check(name: str, severity: Severity = Severity.HIGH) -> Check:
    return Check(
        name=name,
        run=lambda _, name=name: CheckResult.passed(name, "ok"),
        severity=severity,
        remedy=Remedy.RETRAIN_CONFIG,
    )


def _failing_check(name: str, severity: Severity, remedy: Remedy) -> Check:
    return Check(
        name=name,
        run=lambda _, name=name: CheckResult.failed(name, "broken"),
        severity=severity,
        remedy=remedy,
    )


def test_runner_preserves_check_order() -> None:
    checks = [_passing_check(name) for name in ("first", "second", "third")]

    report = run_checks(checks, FakeTokenizer(pieces=("a",)))

    assert [result.check for result in report.results] == ["first", "second", "third"]


def test_runner_stamps_severity_and_remedy_from_the_check() -> None:
    check = _failing_check("a", Severity.BLOCKER, Remedy.FIX_CORPUS)

    result = run_checks([check], FakeTokenizer(pieces=("a",))).results[0]

    assert result.severity is Severity.BLOCKER
    assert result.remedy is Remedy.FIX_CORPUS
    assert result.blocks


def test_a_skipped_blocker_keeps_its_severity() -> None:
    """A skipped BLOCKER must not read as a clean result."""
    check = Check(
        name="a",
        run=lambda _: CheckResult.skipped("a", "no samples supplied"),
        severity=Severity.BLOCKER,
        remedy=Remedy.FIX_CORPUS,
    )

    result = run_checks([check], FakeTokenizer(pieces=("a",))).results[0]

    assert result.status is Status.SKIPPED
    assert result.severity is Severity.BLOCKER
    assert not result.blocks  # it did not fail, but it is not a pass either


def test_ranked_puts_worst_failures_first_and_passes_last() -> None:
    report = run_checks(
        [
            _passing_check("clean"),
            _failing_check("medium_problem", Severity.MEDIUM, Remedy.FIX_CORPUS),
            _failing_check("blocking_problem", Severity.BLOCKER, Remedy.FIX_CORPUS),
            _failing_check("high_problem", Severity.HIGH, Remedy.RETRAIN_CONFIG),
        ],
        FakeTokenizer(pieces=("a",)),
    )

    assert [result.check for result in report.ranked()] == [
        "blocking_problem",
        "high_problem",
        "medium_problem",
        "clean",
    ]


def test_worst_severity_is_info_when_nothing_failed() -> None:
    report = run_checks([_passing_check("clean")], FakeTokenizer(pieces=("a",)))

    assert report.worst_severity() is Severity.INFO


def test_remedies_put_corpus_fixes_before_retraining() -> None:
    """A corpus defect survives any number of retrains, so it must be acted on first."""
    report = run_checks(
        [
            _failing_check("config_problem", Severity.HIGH, Remedy.RETRAIN_CONFIG),
            _failing_check("data_problem", Severity.BLOCKER, Remedy.FIX_CORPUS),
        ],
        FakeTokenizer(pieces=("a",)),
    )

    assert report.remedies() == (Remedy.FIX_CORPUS, Remedy.RETRAIN_CONFIG)


def test_report_serializes_to_json() -> None:
    report = Report(results=(CheckResult.failed("a", "broken", ["evidence"]),))

    payload = report.model_dump(mode="json")

    assert payload["results"][0]["status"] == "failed"
    assert payload["results"][0]["evidence"] == ["evidence"]
