# Math and LaTeX

## Text vs. math is the top of the balancing hierarchy

Treat the text/math split as the first level of the hierarchy described in
[corpus engineering](05-corpus-engineering.md), with script balancing nested inside the text
branch.

Both directions of imbalance hurt:

- **Too little math** — commands like `\frac` fragment. That inflates sequence length and
  multiplies the ways the decoder can produce an invalid command.
- **Too much math** — text script coverage degrades, and you've spent vocab budget on LaTeX
  that the majority of your pages don't contain.

## Protect command atomicity

Use `user_defined_symbols` to force frequent LaTeX commands to remain single unsplittable
tokens. **Curate the list from real frequency counts on your own corpus** — the list in
[configuration](03-configuration.md) is a starting point, not an answer. A command that's
frequent in your data and missing from the list will fragment; a rare command that's on the list
wastes a vocab slot.

Frequency is not a substitute for declaring them. `max_sentencepiece_length=8` caps *learned*
merges, so a command longer than the cap — `\operatorname`, `\displaystyle`, `\underbrace` —
cannot become a single piece no matter how often it appears. For those, declaring is the only
option.

## Split digits to single characters

Set **`split_digits=True`** (it is `False` by default). Merged multi-digit tokens — a single
piece for `12` or `100` — make the model memorize specific frequent numbers and generalize
poorly to unseen groupings. This is a well-documented arithmetic failure mode, and it applies to
OCR too: the model becomes better at reproducing common numbers than at reading the digits
actually on the page.

The default is easy to leave in place because nothing warns you. Ordinary corpus frequency is
enough — invoice or table data where `100` recurs will produce a `100` piece, and the rest of
the configuration in this guide does nothing to prevent it.

**Audit signal:** any digit-only piece longer than 1–2 characters in the trained vocab. Check
for this before you spend training compute — it's in the
[validation checklist](08-validation.md).
