"""Finding text files under a corpus directory.

A corpus directory collects more than corpus — trained `.model` artifacts, editor droppings,
archives. Scanning those produces noise at best and spurious findings at worst.
"""

from pathlib import Path

import pytest

from sentencepiece_ocr_guide.corpus.discover import (
    discover_text_files,
    looks_like_text,
    summarize,
)


@pytest.fixture
def corpus_tree(tmp_path: Path) -> Path:
    """A directory shaped like a real one: shards, subdirectories, artifacts and junk."""
    (tmp_path / "shard_a.txt").write_text("café\n", encoding="utf-8")
    (tmp_path / "extensionless_shard").write_text("no suffix here\n", encoding="utf-8")

    nested = tmp_path / "vendor_b" / "2026"
    nested.mkdir(parents=True)
    (nested / "shard_a.txt").write_text("same name, different dir\n", encoding="utf-8")

    (tmp_path / "ocr.model").write_bytes(b"\x00\x01binary sentencepiece model\x00")
    (tmp_path / ".DS_Store").write_bytes(b"\x00junk")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    return tmp_path


def test_walks_a_directory_recursively(corpus_tree: Path) -> None:
    found = discover_text_files([corpus_tree])

    names = [file.relative.as_posix() for file in found.files]

    assert names == ["extensionless_shard", "shard_a.txt", "vendor_b/2026/shard_a.txt"]


def test_skips_binary_files_with_a_reason(corpus_tree: Path) -> None:
    found = discover_text_files([corpus_tree])

    reasons = {skipped.path.name: skipped.reason for skipped in found.skipped}

    assert reasons == {"ocr.model": "binary"}


def test_skips_hidden_files_and_directories(corpus_tree: Path) -> None:
    found = discover_text_files([corpus_tree])

    paths = [file.path.as_posix() for file in found.files]

    assert not any(".git" in path for path in paths)
    assert not any(".DS_Store" in path for path in paths)


def test_labels_disambiguate_repeated_shard_names(corpus_tree: Path) -> None:
    """Two shard_a.txt files must not collapse into one source in the report."""
    found = discover_text_files([corpus_tree])

    labels = {file.label for file in found.files}

    assert len(labels) == len(found.files)


def test_an_explicitly_named_file_is_accepted_even_if_binary(corpus_tree: Path) -> None:
    """An explicit path is a decision; the tool should not argue with its operator."""
    found = discover_text_files([corpus_tree / "ocr.model"])

    assert [file.path.name for file in found.files] == ["ocr.model"]
    assert not found.skipped


def test_missing_paths_are_reported_not_raised(tmp_path: Path) -> None:
    found = discover_text_files([tmp_path / "nope.txt"])

    assert found.is_empty
    assert found.skipped[0].reason == "does not exist"


def test_results_are_sorted_for_reproducible_reports(corpus_tree: Path) -> None:
    first = discover_text_files([corpus_tree])
    second = discover_text_files([corpus_tree])

    assert [f.path for f in first.files] == [f.path for f in second.files]


def test_empty_directory_discovers_nothing(tmp_path: Path) -> None:
    assert discover_text_files([tmp_path]).is_empty


@pytest.mark.parametrize(
    ("label", "head", "expected"),
    [
        ("plain ascii", b"hello world", True),
        ("utf-8 text", "café 文字".encode(), True),
        ("empty file", b"", True),
        ("nul byte", b"hello\x00world", False),
        ("leading nul", b"\x00", False),
    ],
)
def test_looks_like_text(label: str, head: bytes, expected: bool) -> None:
    assert looks_like_text(head) is expected, label


def test_summarize_is_empty_when_nothing_was_skipped() -> None:
    assert summarize([]) == ""


def test_summarize_truncates_long_lists(corpus_tree: Path) -> None:
    found = discover_text_files([corpus_tree, Path("/nope/a"), Path("/nope/b"), Path("/nope/c")])

    note = summarize(found.skipped, limit=2)

    assert note.startswith("skipped 4: ")
    assert "and 2 more" in note
