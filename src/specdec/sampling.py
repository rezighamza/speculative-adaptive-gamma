"""Distribution helpers. temperature == 0 is greedy (one-hot distributions), which makes the
same accept / residual rules reduce to exact argmax matching."""
import torch


def to_probs(logits, temperature):
    if temperature == 0:
        p = torch.zeros_like(logits, dtype=torch.float32)
        return p.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample(p, gen):
    return torch.multinomial(p, 1, generator=gen).item()


def accept(p_x, q_x, gen):
    """Accept draft token x ~ q with probability min(1, p(x) / q(x))."""
    u = torch.rand(1, generator=gen, device=gen.device).item()
    return u < min(1.0, float(p_x) / max(float(q_x), 1e-12))


def residual(p, q):
    """Distribution norm(max(0, p - q)) used after a rejection; falls back to p if empty."""
    r = (p - q).clamp(min=0)
    s = r.sum()
    return r / s if s > 0 else p
