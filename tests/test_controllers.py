from specdec import AdaptiveGamma, FixedGamma


def test_fixed_is_constant():
    c = FixedGamma(5)
    c.update(0, 5, True)
    assert c.gamma() == 5


def test_adaptive_grows_when_acceptance_high():
    c = AdaptiveGamma(cost_ratio=0.1, prior_alpha=0.5)
    for _ in range(30):
        c.update(accepted=c.gamma(), drafted=c.gamma(), rejected=False)
    assert c.gamma() >= 8


def test_adaptive_shrinks_when_acceptance_low():
    c = AdaptiveGamma(cost_ratio=0.1, prior_alpha=0.9)
    for _ in range(30):
        c.update(accepted=0, drafted=c.gamma(), rejected=True)
    assert c.gamma() <= 2


def test_gamma_within_bounds():
    c = AdaptiveGamma(gamma_min=2, gamma_max=6)
    assert 2 <= c.gamma() <= 6


def test_higher_draft_cost_prefers_shorter_drafts():
    cheap, costly = AdaptiveGamma(cost_ratio=0.05), AdaptiveGamma(cost_ratio=0.8)
    assert cheap.gamma() >= costly.gamma()
