import torch

from .metrics import Stats
from .sampling import sample, to_probs


@torch.no_grad()
def autoregressive_generate(target, prompt, max_new, temperature=0.0, gen=None):
    """Plain token-by-token sampling from the target. `target(ids[T]) -> logits[T, V]`."""
    ids, stats = prompt.clone(), Stats()
    for _ in range(max_new):
        p = to_probs(target(ids)[-1], temperature)
        ids = torch.cat([ids, torch.tensor([sample(p, gen)], device=ids.device)])
        stats.target_calls += 1
        stats.new_tokens += 1
    return ids, stats
