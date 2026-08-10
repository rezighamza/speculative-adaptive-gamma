# Method notes

## Speculative sampling
A cheap draft model `q` proposes `γ` tokens; the target model `p` scores all of them in one
forward pass. Token `x ~ q` is accepted with probability `min(1, p(x)/q(x))`. On the first
rejection a replacement is drawn from `norm(max(0, p − q))` and the round ends; if all `γ` are
accepted a bonus token is drawn from `p` at the next position. The output distribution is
exactly the target's. With temperature 0, distributions are one-hot and the procedure reduces to
greedy argmax matching (output identical to plain greedy decoding).

## Expected speed-up
With per-token acceptance rate `α`, a round yields `(1 − α^(γ+1)) / (1 − α)` tokens in
expectation and costs `γ·c + 1` target-forward equivalents (`c` = draft cost / target cost).

## Adaptive γ (proposed)
`AdaptiveGamma` keeps decayed counts of accepted tokens and trials (accepted tokens plus one
for a rejection), giving a running `α̂`, and each round picks the integer `γ ∈ [γ_min, γ_max]`
maximising the ratio above. High-acceptance stretches (boilerplate, repetition) get long drafts;
low-acceptance stretches fall back to short ones, wasting fewer draft calls.

## Limits
* `c` must be supplied (`--cost_ratio`); it is not measured online.
* No KV cache, so wall-clock numbers under-represent real speed-ups; compare target calls and
  tokens per target call instead.
* `α` is treated as constant within a window; real acceptance varies per token.
* Single sequence, no batching, no tree-structured drafts.
