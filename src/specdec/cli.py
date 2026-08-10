import argparse
import time

import torch

from .baseline import autoregressive_generate
from .controllers import AdaptiveGamma, FixedGamma
from .models import load_pair
from .speculative import speculative_generate


def main():
    p = argparse.ArgumentParser(description="Speculative decoding demo")
    p.add_argument("--target", default="gpt2-medium")
    p.add_argument("--draft", default="distilgpt2")
    p.add_argument("--prompt", default="The history of computing began")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--controller", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--gamma", type=int, default=4, help="gamma for the fixed controller")
    p.add_argument("--cost_ratio", type=float, default=0.3, help="draft step cost / target step cost")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, target, draft = load_pair(a.target, a.draft, device)
    prompt = tok(a.prompt, return_tensors="pt").input_ids[0].to(device)
    gen = torch.Generator(device=device).manual_seed(a.seed)

    t = time.perf_counter()
    _, base = autoregressive_generate(target, prompt, a.max_new, a.temperature, gen)
    t_base = time.perf_counter() - t

    ctrl = FixedGamma(a.gamma) if a.controller == "fixed" else AdaptiveGamma(cost_ratio=a.cost_ratio)
    t = time.perf_counter()
    ids, s = speculative_generate(target, draft, prompt, a.max_new, ctrl, a.temperature, gen)
    t_spec = time.perf_counter() - t

    print(tok.decode(ids))
    print(f"\nbaseline    : {base.target_calls} target calls, {t_base:.2f}s")
    print(f"speculative : {s.target_calls} target calls, {t_spec:.2f}s, "
          f"acceptance {s.acceptance_rate:.2f}, mean gamma {s.mean_gamma:.1f}, "
          f"{s.tokens_per_target_call:.2f} tokens/target call")


if __name__ == "__main__":
    main()
