"""Applying the canonicalizer to a stream of lines.

The functional core of the write path: it consumes lines and yields canonical ones, counting as
it goes. It performs no I/O, so the caller decides where the output lands and can stream a file
of any size through it.

Undecodable bytes stop the rewrite rather than being silently carried through. Unlike every
other defect the corpus tooling reports, they cannot be repaired by normalizing — writing them
back out would either crash the encoder or launder corrupt data into a file that now *looks*
canonical. Dropping them is possible but must be asked for, because it is data loss.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from sentencepiece_ocr_guide.corpus.undecodable import has_undecodable_bytes


class UndecodableLineError(ValueError):
    """Raised when a line contains bytes that were not valid UTF-8."""

    def __init__(self, source: str, line_number: int) -> None:
        super().__init__(
            f"{source}: line {line_number} contains bytes that are not valid UTF-8. "
            "Canonicalizing cannot recover them — fix the extractor, or pass --drop-invalid "
            "to skip these lines and accept the data loss."
        )
        self.source = source
        self.line_number = line_number


@dataclass
class RewriteTally:
    """Counters for one file. Mutable by design: the caller owns it and streams past it."""

    read: int = 0
    written: int = 0
    changed: int = 0
    dropped: int = 0

    def summary(self) -> str:
        parts = [f"{self.read:,} read", f"{self.changed:,} changed"]
        if self.dropped:
            parts.append(f"{self.dropped:,} dropped (invalid UTF-8)")
        return ", ".join(parts)


@dataclass
class RewriteRun:
    """Totals across every file in one invocation."""

    per_source: dict[str, RewriteTally] = field(default_factory=dict)

    def tally_for(self, source: str) -> RewriteTally:
        return self.per_source.setdefault(source, RewriteTally())

    @property
    def changed(self) -> int:
        return sum(tally.changed for tally in self.per_source.values())

    @property
    def dropped(self) -> int:
        return sum(tally.dropped for tally in self.per_source.values())


def rewrite_lines(
    lines: Iterable[str],
    canonicalize: Callable[[str], str],
    tally: RewriteTally,
    source: str = "<input>",
    drop_undecodable: bool = False,
) -> Iterator[str]:
    """Yield the canonical form of every line, recording what happened in `tally`.

    Raises `UndecodableLineError` on the first undecodable line unless `drop_undecodable`,
    so a caller writing to a temporary file can abandon it without leaving partial output.
    """
    for line in lines:
        tally.read += 1

        if has_undecodable_bytes(line):
            if not drop_undecodable:
                raise UndecodableLineError(source, tally.read)
            tally.dropped += 1
            continue

        canonical = canonicalize(line)
        if canonical != line:
            tally.changed += 1

        tally.written += 1
        yield canonical
