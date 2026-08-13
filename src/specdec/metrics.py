from dataclasses import dataclass


@dataclass
class Stats:
    iterations: int = 0
    drafted: int = 0       # draft tokens proposed
    accepted: int = 0      # draft tokens accepted
    new_tokens: int = 0
    target_calls: int = 0
    draft_calls: int = 0

    @property
    def acceptance_rate(self):
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_target_call(self):
        return self.new_tokens / self.target_calls if self.target_calls else 0.0

    @property
    def mean_gamma(self):
        return self.drafted / self.iterations if self.iterations else 0.0
