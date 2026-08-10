"""Draft-length (gamma) controllers.

FixedGamma    : original method, constant gamma.
AdaptiveGamma : proposed. Tracks a decayed estimate of the per-token acceptance rate alpha and
                picks the gamma maximising expected speed-up (Leviathan et al., Thm 3.8 style):
                    tokens/iter = (1 - alpha^(g+1)) / (1 - alpha),  cost/iter = g * c + 1
                where c = draft-step cost / target-step cost.
"""


class FixedGamma:
    def __init__(self, gamma=4):
        self._gamma = gamma

    def gamma(self):
        return self._gamma

    def update(self, accepted, drafted, rejected):
        pass


class AdaptiveGamma:
    def __init__(self, cost_ratio=0.1, gamma_min=1, gamma_max=12, decay=0.9, prior_alpha=0.7):
        self.c, self.lo, self.hi, self.decay = cost_ratio, gamma_min, gamma_max, decay
        # pseudo-counts so the first iterations start from prior_alpha
        self.acc, self.trials = prior_alpha * 2.0, 2.0

    @property
    def alpha(self):
        return self.acc / self.trials

    def speedup(self, gamma, alpha=None):
        a = self.alpha if alpha is None else alpha
        tokens = gamma + 1 if a >= 0.999 else (1 - a ** (gamma + 1)) / (1 - a)
        return tokens / (gamma * self.c + 1)

    def gamma(self):
        return max(range(self.lo, self.hi + 1), key=self.speedup)

    def update(self, accepted, drafted, rejected):
        """`accepted` draft tokens were accepted; if `rejected`, one more token was tried and failed."""
        self.acc = self.decay * self.acc + accepted
        self.trials = self.decay * self.trials + accepted + (1 if rejected else 0)
