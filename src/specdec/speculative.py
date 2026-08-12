"""Speculative sampling loop.

`target` / `draft` are callables `ids[T] -> logits[T, V]` (logits at every position), so the
loop is model-agnostic and testable with toy tables. No KV cache is used: each call recomputes
the prefix, which is simple and correct but slower than production implementations.
"""
import torch

from .metrics import Stats
from .sampling import accept, residual, sample, to_probs


@torch.no_grad()
def speculative_generate(target, draft, prompt, max_new, controller, temperature=0.0, gen=None):
    ids, stats = prompt.clone(), Stats()
    start = len(prompt)
    dev = ids.device
    while len(ids) - start < max_new:
        gamma = controller.gamma()

        # 1) draft gamma tokens autoregressively
        cur, q_dists, toks = ids, [], []
        for _ in range(gamma):
            q = to_probs(draft(cur)[-1], temperature)
            x = sample(q, gen)
            q_dists.append(q)
            toks.append(x)
            cur = torch.cat([cur, torch.tensor([x], device=dev)])
            stats.draft_calls += 1

        # 2) one target pass scores all drafted positions plus the bonus position
        n = len(ids)
        p_dists = to_probs(target(cur)[n - 1:], temperature)  # [gamma + 1, V]
        stats.target_calls += 1

        # 3) accept / reject left to right
        accepted, rejected, new = 0, False, []
        for i, x in enumerate(toks):
            if accept(p_dists[i][x], q_dists[i][x], gen):
                new.append(x)
                accepted += 1
            else:
                new.append(sample(residual(p_dists[i], q_dists[i]), gen))
                rejected = True
                break
        if not rejected:
            new.append(sample(p_dists[gamma], gen))  # bonus token

        ids = torch.cat([ids, torch.tensor(new, device=dev)])
        controller.update(accepted, gamma, rejected)
        stats.iterations += 1
        stats.drafted += gamma
        stats.accepted += accepted

    ids = ids[:start + max_new]
    stats.new_tokens = len(ids) - start
    return ids, stats
