"""What each stream carries.

`--json` exists to be piped into something else, so stdout must hold the report and nothing but
the report. Everything a run has to say about itself — files passed over, rewrite tallies, phase
announcements, the progress bar — belongs on stderr. This is easy to regress by adding one
well-meaning `print`, and a single skipped file was once enough to make the JSON unparseable.
"""

import json

import pytest

from sentencepiece_ocr_guide.cli import CanonicalizeCorpus, CorpusChecklist


@pytest.fixture
def corpus(tmp_path):
    """A directory holding one scannable file and one binary file to be skipped."""
    (tmp_path / "shard.txt").write_text("hello world\ncafé\n", encoding="utf-8")
    (tmp_path / "trained.model").write_bytes(b"\x00\x01\x02 not text")
    return tmp_path


def test_json_stays_parseable_when_a_file_is_skipped(corpus, capsys) -> None:
    """The regression this file exists for: the skipped note used to land on stdout."""
    with pytest.raises(SystemExit):
        CorpusChecklist(files=[corpus], json=True).cli_cmd()

    captured = capsys.readouterr()

    report = json.loads(captured.out)
    assert report["results"], "the report should carry results"
    assert "skipped" in captured.err, "the note must still be shown, just not on stdout"


def test_canonicalize_keeps_its_tallies_off_stdout(corpus, tmp_path, capsys) -> None:
    """Canonicalize prints a per-source tally and a blank separator — both are commentary."""
    with pytest.raises(SystemExit):
        CanonicalizeCorpus(files=[corpus], out=tmp_path / "out", json=True).cli_cmd()

    captured = capsys.readouterr()

    json.loads(captured.out)
    assert "read" in captured.err, "the rewrite tally must still reach the operator"


def test_text_output_is_unaffected(corpus, capsys) -> None:
    """Moving commentary to stderr must not empty the human-readable report."""
    with pytest.raises(SystemExit):
        CorpusChecklist(files=[corpus], json=False).cli_cmd()

    captured = capsys.readouterr()
    assert captured.out.strip(), "the text report still goes to stdout"
