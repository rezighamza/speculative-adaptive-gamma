"""Wrap Hugging Face causal LMs as `ids[T] -> logits[T, V]` callables."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def as_logits_fn(model):
    @torch.no_grad()
    def fn(ids):
        return model(ids.unsqueeze(0)).logits[0]
    return fn


def load_pair(target_name, draft_name, device):
    """Target and draft must share a tokenizer / vocabulary (e.g. gpt2-medium + distilgpt2)."""
    tok = AutoTokenizer.from_pretrained(target_name)
    target = AutoModelForCausalLM.from_pretrained(target_name).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_name).to(device).eval()
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("target and draft vocabularies differ")
    return tok, as_logits_fn(target), as_logits_fn(draft)
