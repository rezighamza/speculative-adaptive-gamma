# specdec-adaptive-gamma

Standalone re-implementation of **speculative decoding** with an adaptive draft-length controller.

## Original work
**Fast Inference from Transformers via Speculative Decoding** (Leviathan, Kalman, Matias; arXiv 2211.17192, ICML 2023). A small draft model proposes `γ` tokens, the large target model verifies them in a single forward pass, and a rejection-sampling rule keeps the output distribution identical to the target's. Latency drops because several tokens are committed per target call.

Re-implemented here for any pair of Hugging Face causal LMs sharing a vocabulary (default `gpt2-medium` target + `distilgpt2` draft), including the temperature > 0 acceptance / residual-resampling rule and a greedy mode.

## Issue
The draft length `γ` is a fixed hyper-parameter. The best value depends on the acceptance rate `α` (how well the draft imitates the target) and the draft/target cost ratio, and `α` changes from prompt to prompt and from token to token. A fixed `γ` over-drafts where acceptance is poor (wasted draft calls) and under-drafts where it is high (lost speed-up).

## Proposed solution (implemented)
`AdaptiveGamma` maintains a decayed running estimate of `α` from accepted / rejected counts and, every round, selects the `γ` that maximises the expected-speed-up formula `(1 − α^(γ+1)) / ((1 − α)(γc + 1))`. It plugs into the same loop as `FixedGamma`, and output correctness is unaffected since only the draft length changes. See [docs/METHOD.md](docs/METHOD.md).

Not implemented: online measurement of the cost ratio `c`, KV caching, batching, tree drafts.

## Layout
```
src/specdec/
  sampling.py     probabilities, accept rule, residual distribution
  speculative.py  draft / verify / accept loop
  baseline.py     plain autoregressive decoding
  controllers.py  FixedGamma and AdaptiveGamma
  metrics.py      acceptance rate, tokens per target call
  models.py       HF model wrappers
  cli.py          demo entry point
scripts/benchmark.py     fixed vs adaptive over several prompts
tests/                   toy bigram models, no downloads
docs/METHOD.md
```

## Run (uv)
```
uv sync
uv run specdec --controller adaptive
uv run specdec --controller fixed --gamma 4
uv run python scripts/benchmark.py
uv run pytest
```

## Status
Written but not executed: no tests were run and no results exist. Paper details are from memory; verify the citation before relying on it.
