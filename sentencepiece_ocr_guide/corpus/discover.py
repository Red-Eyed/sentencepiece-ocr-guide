"""Finding the text files under a corpus directory.

A corpus directory is rarely only corpus: it collects trained `.model` and `.vocab` artifacts,
editor droppings, archives and checkpoints. Scanning those is at best noise and at worst a
report full of spurious findings from binary data that happens to decode.

Text detection uses the same heuristic as `git` and `grep` — a NUL byte in the leading chunk
means binary. It beats an extension allowlist because corpus shards are frequently
extensionless, and it beats trusting the extension because a `.txt` can still be binary.

Files named explicitly by the user are always accepted: an explicit path is a decision, and
second-guessing it would make the tool argue with its operator.
"""

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from sentencepiece_ocr_guide.concurrency import run_parallel

SNIFF_BYTES = 8192


@dataclass(frozen=True)
class TextFile:
    """A discovered file, and the root it was found under."""

    path: Path
    root: Path

    @property
    def label(self) -> str:
        """Report name. The full path, because shard names repeat across subdirectories."""
        return str(self.path)

    @property
    def relative(self) -> Path:
        """Position within its root, so an output directory can mirror the input tree."""
        try:
            return self.path.relative_to(self.root)
        except ValueError:
            return Path(self.path.name)


@dataclass(frozen=True)
class Skipped:
    """A path that was not scanned, and why. The reason travels to the caller."""

    path: Path
    reason: str


@dataclass(frozen=True)
class Discovery:
    files: tuple[TextFile, ...]
    skipped: tuple[Skipped, ...]

    @property
    def is_empty(self) -> bool:
        return not self.files


def discover_text_files(roots: Sequence[Path], workers: int = 1) -> Discovery:
    """Expand `roots` into text files, walking any directory recursively.

    Results are sorted so a report over the same tree is byte-identical between runs, whatever
    order the sniffing threads finished in.
    """
    files: list[TextFile] = []
    skipped: list[Skipped] = []
    candidates: list[TextFile] = []

    for root in roots:
        if root.is_dir():
            candidates.extend(_candidates_under(root))
        elif root.exists():
            files.append(TextFile(path=root, root=root.parent))
        else:
            skipped.append(Skipped(path=root, reason="does not exist"))

    # Sniffing opens and reads every candidate, so it is I/O-bound and worth spreading out.
    for candidate, reason in zip(
        candidates, run_parallel(_rejection, [found.path for found in candidates], workers)
    ):
        if reason is None:
            files.append(candidate)
        else:
            skipped.append(Skipped(path=candidate.path, reason=reason))

    files.sort(key=lambda found: found.path)
    skipped.sort(key=lambda entry: entry.path)
    return Discovery(files=tuple(files), skipped=tuple(skipped))


def _candidates_under(root: Path) -> Iterator[TextFile]:
    """Every non-hidden file under `root`, before any content sniffing."""
    # `followlinks=False` keeps a symlinked parent directory from causing an infinite walk.
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        subdirectories[:] = sorted(name for name in subdirectories if not name.startswith("."))

        for name in sorted(names):
            if not name.startswith("."):
                yield TextFile(path=Path(directory) / name, root=root)


def _rejection(path: Path) -> str | None:
    """Why `path` should not be scanned, or `None` when it looks like text."""
    try:
        with path.open("rb") as handle:
            head = handle.read(SNIFF_BYTES)
    except OSError as error:
        return f"unreadable ({error.strerror or error})"

    if not looks_like_text(head):
        return "binary"
    return None


def looks_like_text(head: bytes) -> bool:
    """True when a leading chunk contains no NUL byte. An empty file counts as text."""
    return b"\x00" not in head


def summarize(skipped: Iterable[Skipped], limit: int = 3) -> str:
    """One line naming what was passed over, so a surprising file count is explainable."""
    entries = list(skipped)
    if not entries:
        return ""

    shown = ", ".join(f"{entry.path.name} ({entry.reason})" for entry in entries[:limit])
    if len(entries) > limit:
        shown += f", and {len(entries) - limit:,} more"
    return f"skipped {len(entries):,}: {shown}"
