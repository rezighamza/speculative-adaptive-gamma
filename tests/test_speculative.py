import torch
from toy import make_gen, make_model

from specdec import AdaptiveGamma, FixedGamma, autoregressive_generate, speculative_generate

PROMPT = torch.tensor([1, 2, 3])


def test_identical_draft_accepts_everything():
    m = make_model(0)
    _, s = speculative_generate(m, m, PROMPT, 20, FixedGamma(4), temperature=1.0, gen=make_gen())
    assert s.accepted == s.drafted


def test_greedy_matches_baseline_with_bad_draft():
    target, draft = make_model(0), make_model(99)
    base, _ = autoregressive_generate(target, PROMPT, 25, 0.0, make_gen())
    spec, _ = speculative_generate(target, draft, PROMPT, 25, FixedGamma(3), 0.0, make_gen())
    assert torch.equal(base, spec)


def test_greedy_matches_baseline_with_adaptive_controller():
    target, draft = make_model(0), make_model(99)
    base, _ = autoregressive_generate(target, PROMPT, 25, 0.0, make_gen())
    spec, _ = speculative_generate(target, draft, PROMPT, 25, AdaptiveGamma(0.2), 0.0, make_gen())
    assert torch.equal(base, spec)


def test_output_length_exact():
    ids, s = speculative_generate(make_model(0), make_model(1), PROMPT, 17, FixedGamma(5), 1.0, make_gen())
    assert len(ids) == len(PROMPT) + 17 and s.new_tokens == 17


def test_sampling_matches_target_distribution():
    """First generated token should follow the target's next-token distribution."""
    target, draft = make_model(0), make_model(7)
    counts = torch.zeros(11)
    gen = make_gen(1)
    for _ in range(2000):
        ids, _ = speculative_generate(target, draft, PROMPT, 1, FixedGamma(2), 1.0, gen)
        counts[ids[-1]] += 1
    expected = torch.softmax(target(PROMPT)[-1], -1)
    assert (counts / counts.sum() - expected).abs().max() < 0.05
