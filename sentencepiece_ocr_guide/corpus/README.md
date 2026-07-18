# corpus

Pre-training tooling: measure how your sources disagree about encoding, then collapse the
disagreements that are not real. Runs *before* the checks in [`../checks/`](../checks/README.md)
— several of those only ever report defects that originate here.

## Why measuring comes first

The exception list beyond NFC is a judgement about *your* extractors, not a universal. Soft
hyphen is the proof: U+00AD is invisible mid-line but renders as a real hyphen where the line
breaks, and OCR ground truth is usually segmented per line. Strip it blindly and you train the
model to omit a mark that is on the page.

So [`scan.py`](scan.py) reports counts per axis *and per source*, which turns that from a
guess into a decision — 9,117 line-final versus 226 mid-line is answerable. It also usually
identifies one broken extractor rather than a diffuse problem.

## Axes

An [`Axis`](axes.py) is a name, a pure `str -> str` transform, and an `Action`:

| Action | Meaning | Severity when found |
|---|---|---|
| `COLLAPSE` | Not a difference in the text | `BLOCKER` |
| `DECIDE` | Depends on your sources — opt in explicitly | `HIGH` |
| `PRESERVE` | A real character difference | `INFO` — never a failure |

A line is affected when the transform changes it, so one definition serves both the scanner and
the canonicalizer. `PRESERVE` axes carry a transform purely for *detection*: `canonicalize`
filters on `Action`, so folding one by accident is structurally impossible rather than merely
discouraged.

`PRESERVE` counts being non-zero is the expected case — fullwidth forms showing up means you
have CJK data. A report that cannot distinguish "found variation" from "found a problem" trains
you to ignore it.

## Canonicalizing

```python
from sentencepiece_ocr_guide.corpus import canonicalizer, is_canonical

canonicalize = canonicalizer(decide=("soft_hyphen_line_final",))   # after measuring

for line in incoming:
    assert is_canonical(canonicalize(line), canonicalize)   # idempotent by construction
```

`canonicalize` applies `COLLAPSE` axes plus whichever `DECIDE` axes you name. Naming one that
does not exist raises rather than silently doing nothing — a typo there would quietly disable a
transform you believed was running.

It is idempotent, which is what makes `line == canonicalize(line)` a valid assertion at
corpus-write time. That assertion is the point: it moves the guarantee from a documented step
that every pipeline is *supposed* to call into an invariant that fails loudly.

## Pointing at a directory

Both commands take files or directories; a directory is walked recursively.

```
spm-ocr corpus       corpus/
spm-ocr canonicalize corpus/ --out canonical/     # output mirrors the input tree
```

A corpus directory is rarely only corpus, so [`discover.py`](discover.py) filters it:

- **Binary files are skipped**, detected by a NUL byte in the first 8 KB — the `git`/`grep`
  heuristic. It beats an extension allowlist because corpus shards are often extensionless, and
  beats trusting the extension because a `.txt` can still be binary. Trained `.model` and
  `.vocab` artifacts living beside the corpus are the common case.
- **Hidden files and directories are skipped**, so `.git` and `.DS_Store` never appear.
- **A file named explicitly is always accepted**, binary or not. An explicit path is a decision,
  and second-guessing it would make the tool argue with its operator.
- **Skips are reported**, not silent: `skipped 2: corpus.tar.gz (binary), ocr.model (binary)`.
  An unexplained file count is how you end up scanning a quarter of your corpus.

Sources are labelled by full path rather than filename, because shard names repeat across
subdirectories and two `latin.txt` files must not collapse into one line of the report.

## Rewriting a corpus

`spm-ocr canonicalize` applies the above to files and then re-scans its own output, so the
invariant is observed rather than assumed:

```
spm-ocr canonicalize corpus/*.txt --out canonical/
spm-ocr canonicalize corpus/*.txt --in-place --decide soft_hyphen_line_final
```

```
vendor_b.txt: 50,000 read, 7,188 changed, 3 dropped (invalid UTF-8)

PASS  axis[nfc_composition]: no variation across 49,997 lines
PASS  invalid_utf8: every one of 49,997 lines decoded as valid UTF-8
PASS  axis[fullwidth_forms]: 7,176 of 49,997 lines (expected — preserve, do not fold)
```

Those last two lines are the whole point: the collapsible variation is gone, and the 7,176 lines
of fullwidth text are still there. A canonicalizer that quietly flattened them would look just
as clean.

Three behaviours worth knowing:

- **It never overwrites the input** unless `--in-place` is passed. A corpus is expensive to
  reassemble, and a run configured with the wrong `--decide` axes does not look wrong afterwards.
- **Undecodable bytes stop the run.** They cannot be repaired by normalizing, so writing them
  back would either crash the encoder or launder corrupt data into a file that now *looks*
  canonical. `--drop-invalid` skips those lines and reports the count, because that is data loss
  and should be chosen. Output is written through a temporary file, so a refusal leaves nothing
  partial behind.
- **It is idempotent**, so running it twice is a no-op and re-running over a mixed corpus is safe.

## Verifying the stage worked

Re-scan after canonicalizing. Every `COLLAPSE` axis should read zero and every `PRESERVE` axis
should be unchanged; that difference is the proof. `tests/corpus/test_scan.py` and
`tests/corpus/test_rewrite.py` assert exactly this round trip.

A one-shot rewrite fixes the corpus you have. It does not make the representation invariant
going forward — that needs `canonicalizer()` wired into whatever adds data, with
`is_canonical` asserted at write time. The command is for existing data; the library function
is for the chokepoint.

## What this does not cover

The corpus checklist in [docs/08-validation.md](../../docs/08-validation.md) is broader than
encoding: per-category share, and lines silently dropped by `max_sentence_length`. Those need
different inputs and are not implemented here — and note that neither is fixed by canonicalizing,
which is why remedy is tracked per finding rather than per checklist.
