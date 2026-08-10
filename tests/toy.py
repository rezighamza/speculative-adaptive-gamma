"""Toy 'language models': next-token logits depend only on the current token (a bigram table)."""
import torch


def make_model(seed, vocab=11, scale=2.0):
    table = torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(seed)) * scale

    def fn(ids):
        return table[ids]  # [T, V]
    return fn


def make_gen(seed=0):
    return torch.Generator().manual_seed(seed)
