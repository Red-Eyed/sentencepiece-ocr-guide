"""The single chokepoint every source passes through on the way into the corpus.

Applies the `COLLAPSE` axes and nothing else. `DECIDE` axes — currently the two soft-hyphen
cases — are opt-in, because the right answer depends on how your extractor behaves and you
should have measured it first. `PRESERVE` axes are structurally unreachable from here.

The result is idempotent, which is what makes `line == canonicalize(line)` a valid assertion at
corpus-write time and turns this from an instruction into an enforced invariant.
"""

from collections.abc import Callable, Sequence

from sentencepiece_ocr_guide.corpus.axes import DEFAULT_AXES, Action, Axis, axes_with


def canonicalizer(
    axes: tuple[Axis, ...] = DEFAULT_AXES,
    decide: Sequence[str] = (),
) -> Callable[[str], str]:
    """Build the canonicalizing function.

    `decide` names `DECIDE` axes to include, e.g. `("soft_hyphen_line_final",)`. Naming one that
    does not exist is an error rather than a silent no-op — a typo here would quietly disable a
    transform you believed was running.
    """
    applied = axes_with(Action.COLLAPSE, axes) + _selected_decide_axes(axes, decide)

    def canonicalize(text: str) -> str:
        for axis in applied:
            text = axis.transform(text)
        return text

    return canonicalize


def _selected_decide_axes(axes: tuple[Axis, ...], decide: Sequence[str]) -> tuple[Axis, ...]:
    available = {axis.name: axis for axis in axes_with(Action.DECIDE, axes)}
    unknown = sorted(set(decide) - available.keys())
    if unknown:
        raise ValueError(
            f"unknown DECIDE axes: {', '.join(unknown)}. Available: {', '.join(sorted(available))}"
        )
    return tuple(available[name] for name in decide)


def is_canonical(text: str, canonicalize: Callable[[str], str]) -> bool:
    """The write-time assertion: text that already passed through the chokepoint is unchanged."""
    return canonicalize(text) == text
