"""Compare fixed-gamma and adaptive-gamma speculative decoding over several prompts.

    uv run python scripts/benchmark.py --target gpt2-medium --draft distilgpt2
"""
import argparse

import torch

from specdec import AdaptiveGamma, FixedGamma, speculative_generate
from specdec.models import load_pair

PROMPTS = [
    "The history of computing began",
    "def fibonacci(n):\n    ",
    "In 1969, the first humans",
    "Q: What is the capital of France?\nA:",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="gpt2-medium")
    p.add_argument("--draft", default="distilgpt2")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cost_ratio", type=float, default=0.3)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, target, draft = load_pair(a.target, a.draft, device)
    configs = {f"fixed-{g}": (lambda g=g: FixedGamma(g)) for g in (2, 4, 8)}
    configs["adaptive"] = lambda: AdaptiveGamma(cost_ratio=a.cost_ratio)

    print(f"{'controller':12s} {'tok/target-call':>16s} {'accept':>8s} {'mean gamma':>11s}")
    for name, make in configs.items():
        tpc, acc, gam = [], [], []
        for text in PROMPTS:
            prompt = tok(text, return_tensors="pt").input_ids[0].to(device)
            gen = torch.Generator(device=device).manual_seed(0)
            _, s = speculative_generate(target, draft, prompt, a.max_new, make(), a.temperature, gen)
            tpc.append(s.tokens_per_target_call); acc.append(s.acceptance_rate); gam.append(s.mean_gamma)
        n = len(PROMPTS)
        print(f"{name:12s} {sum(tpc) / n:16.2f} {sum(acc) / n:8.2f} {sum(gam) / n:11.1f}")


if __name__ == "__main__":
    main()
