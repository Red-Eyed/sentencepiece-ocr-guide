# checks

Executable form of the [validation checklist](../../docs/08-validation.md). Every check
corresponds to a failure mode in [docs/09-failure-modes.md](../../docs/09-failure-modes.md) that
can be detected from the tokenizer artifact alone.

## Architecture

**Functional core.** Nothing in this package imports `sentencepiece`. Checks depend on the
protocols in [`protocols.py`](protocols.py) — `Encoder` (encode/decode) and `Vocabulary` (pieces
and their properties). A check takes the narrower of the two where it can: `digit_pieces` never
sees an encoder, `round_trip` never sees the vocabulary.

**Imperative shell.** [`adapters/spm.py`](../adapters/spm.py) is the only module that knows what
a SentencePiece model is. [`cli.py`](../cli.py) is the only module that knows how results are
displayed. Checks return data.

**Configuration is bound at construction.** Each check module exposes a builder that closes over
its parameters and returns a `Check`:

```python
from sentencepiece_ocr_guide.checks import round_trip, run_checks
from sentencepiece_ocr_guide.adapters import SentencePieceTokenizer

report = run_checks(
    [round_trip(samples=my_ground_truth_lines)],
    SentencePieceTokenizer.from_file(Path("ocr.model")),
)
```

The runner therefore needs to know nothing about what any individual check requires, and callers
cannot pass mismatched parameters at run time.

## Outcomes

Three, not two. `SKIPPED` exists because a check that could not run must never report success,
and it always carries the reason — a skipped check with no explanation is exactly the silent
failure this package exists to catch. `Report.ok` is false only on `FAILED`, but skips stay
visible and keep their declared severity.

`Status` answers *did it fail*. Two more fields answer the questions a pass/fail list leaves
open:

- **`Severity`** — `BLOCKER` (do not spend training compute) / `HIGH` / `MEDIUM` / `INFO`.
  Drives ranking and the CLI exit code.
- **`Remedy`** — `RETRAIN_CONFIG` (flip a flag, retrain) / `FIX_CORPUS` (fix data, *then*
  retrain) / `FIX_INTEGRATION`.

Both are declared on the `Check`, not built inside `run` — what a failure *means* is a property
of the check, not of one execution — and the runner stamps them onto each result.

The remedy split is why this is not simply "model checks" versus "corpus checks":
`nfc_vocabulary` fails on the *artifact* but carries `FIX_CORPUS`, because retraining on the
same corpus reproduces it exactly. `Report.remedies()` orders corpus fixes before retrains for
that reason.

## Adding a check

Add a module with a builder returning `Check`, then one line in [`suite.py`](suite.py). No
existing check changes. Pair it with a test in `tests/pitfalls/` that induces the real defect and
proves the check fires — a check that has only ever seen good input proves nothing.

## Thresholds

`byte_fallback_rate` and `digit_pieces` ship with defaults from the guide. `fertility` ships with
none, and the standard suite runs it only for groups you give a ceiling: acceptable fragmentation
depends on your vocab size and script mix, and a threshold nobody chose is a threshold nobody
should trust.

## What this cannot check

Failure modes #5, #7, #16 and #18–21 in [docs/09-failure-modes.md](../../docs/09-failure-modes.md)
are invisible from the artifact — they live in the corpus, the inference path, or the integration
between tokenizer and checkpoint. Handle those with process, not this suite.
