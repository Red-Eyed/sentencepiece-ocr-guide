# Prior art

Most published guidance on multilingual SentencePiece training targets **language models**.
That guidance is a reasonable starting point but not directly transferable to OCR — see
[BPE vs Unigram](02-bpe-vs-unigram.md) for where it diverges. This page collects the
references worth reading and what each one actually gives you.

## Repositories

| Repo | What it gives you |
|---|---|
| [cisnlp/Glot500](https://github.com/cisnlp/Glot500) | Full pipeline for building a balanced multilingual corpus (500+ languages) and training a Unigram tokenizer, with scripts for merging per-language data using a configurable sampling scale. The closest end-to-end working reference. |
| [google/sentencepiece](https://github.com/google/sentencepiece) | The library. `doc/options.md` documents every training flag, including the ones that aren't obvious from the API: `train_extremely_large_corpus`, `shuffle_input_sentence`, `input_sentence_size`. |
| [facebookresearch/XLM](https://github.com/facebookresearch/XLM) | Shell scripts for building the joint SPM/BPE vocab for XLM-R over 100 languages from CC-100. |
| [fairseq `examples/multilingual`](https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual) | Clean implementation of temperature-based language balancing (`--sampling-method temperature`). |

The frequently cited `raymondhs/bert-sentencepiece/multilingual.md` — exponential smoothing of
language sampling probabilities with factor S=0.7 — describes the right idea but too little of
the surrounding pipeline to build from.

## Papers

**[NLLB — No Language Left Behind](https://arxiv.org/abs/2207.04672)** (§8.1.1)
Trains a from-scratch tokenizer for 200+ languages on 100M sampled sentences, using
non-uniform sampling specifically to avoid low-resource fragmentation. The useful data point:
100M sentences was enough for 200 languages. You do not need your full corpus.

**[Glot500](https://aclanthology.org/2023.acl-long.61/)** (ACL 2023)
Multinomial sampling with α=0.3, flooring high-resource sampling to match the lowest-resource
counts. Detailed below.

**[The Art of Breaking Words](https://arxiv.org/abs/2508.06533)**
Fertility-based iterative reweighting as an alternative to fixed temperature sampling. Worth
reading if a single α doesn't give acceptable per-script compression and you're willing to
iterate.

**[Canary/Parakeet](https://arxiv.org/abs/2509.14128)** (NVIDIA)
Practical vocab-size ablation (4K/8K/16K) with per-language compression statistics. The most
directly comparable published numbers if you're choosing a vocab size empirically.

## What Glot500 actually does

Worth spelling out, because the paper is commonly cited for a method that is partly specific to
its setup.

**Method.** SentencePiece with a Unigram language model, vocab size 250K, then **merged into
XLM-R's existing 250K vocabulary** — final size ~401K after adding ~151K genuinely new tokens.
Roughly 100K of the "new" tokens already existed in XLM-R.

**Sampling.** Multinomial distribution with **α = 0.3**. High-resource ("head") languages are
capped at the sampling amount of the lowest-resource ("tail") languages, deliberately favouring
tail languages on the reasoning that head languages are already well covered by the XLM-R base.

**The empirical finding that matters.** For head languages, 0.2%–50% of tokens changed under
the new tokenizer — but there was **no correlation** between how much tokenization changed and
downstream performance change. Tokenizer perturbation on well-resourced languages was mostly
harmless.

**The caveat.** Glot500 *extends* an existing vocabulary; it does not train from scratch. The
α=0.3 sampling idea transfers directly to a from-scratch build. The merge/reuse-old-tokens step
is specific to their continued-pretraining setup and is not something you need.
