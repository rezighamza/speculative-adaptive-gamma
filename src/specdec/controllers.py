"""Draft-length (gamma) controllers.

FixedGamma    : original method, constant gamma.
AdaptiveGamma : proposed. Tracks a decayed estimate of the per-token acceptance rate alpha and
                picks the gamma maximising expected speed-up (Leviathan et al., Thm 3.8 style):
                    tokens/iter = (1 - alpha^(g+1)) / (1 - alpha),  cost/iter = g * c + 1
                where c = draft-step cost / target-step cost.
"""
from typing import Optional


class FixedGamma:
    def __init__(self, gamma: int = 4) -> None:
        self._gamma = gamma

    def gamma(self) -> int:
        return self._gamma

    def update(self, accepted: int, drafted: int, rejected: bool) -> None:
        pass


class AdaptiveGamma:
    def __init__(self, cost_ratio: float = 0.1, gamma_min: int = 1, gamma_max: int = 12, decay: float = 0.9, prior_alpha: float = 0.7) -> None:
        if gamma_min > gamma_max:
            raise ValueError(f"gamma_min ({gamma_min}) cannot exceed gamma_max ({gamma_max})")
        if cost_ratio <= 0:
            raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")
        self.c, self.lo, self.hi, self.decay = cost_ratio, gamma_min, gamma_max, decay
        self.acc, self.trials = prior_alpha * 2.0, 2.0

    @property
    def alpha(self) -> float:
        return self.acc / self.trials

    def speedup(self, gamma: int, alpha: Optional[float] = None) -> float:
        a = self.alpha if alpha is None else alpha
        tokens = gamma + 1 if a >= 0.999 else (1 - a ** (gamma + 1)) / (1 - a)
        return tokens / (gamma * self.c + 1)

    def gamma(self) -> int:
        return max(range(self.lo, self.hi + 1), key=self.speedup)

    def update(self, accepted: int, drafted: int, rejected: bool) -> None:
        """`accepted` draft tokens were accepted; if `rejected`, one more token was tried and failed."""
        self.acc = self.decay * self.acc + accepted
        self.trials = self.decay * self.trials + accepted + (1 if rejected else 0)
