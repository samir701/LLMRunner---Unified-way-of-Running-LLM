"""Next-token sampling from logits using repetition penalty, top-k, temperature, and top-p."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.5,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    generated_ids: Sequence[int] | None = None,
) -> int:
    """Select the next token ID from `[1, vocab_size]` or `[vocab_size]` logits on GPU or CPU."""
    if logits.dim() == 2:
        logits = logits[0]

    has_rep = repetition_penalty > 1.0 and bool(generated_ids)
    work = logits.to(dtype=torch.float32)
    if has_rep:
        work = work.clone()
        seen = torch.tensor(list(set(generated_ids)), dtype=torch.long, device=work.device)
        scores = work[seen]
        work[seen] = torch.where(
            scores < 0,
            scores * repetition_penalty,
            scores / repetition_penalty,
        )

    if temperature <= 0.0:
        return int(torch.argmax(work).item())

    scaled = work / temperature

    if top_k > 0 and top_k < scaled.numel():
        # torch.topk returns descending-sorted values, avoiding a full-vocab sort for top-p.
        top_vals, top_idx = torch.topk(scaled, top_k)
        if 0.0 < top_p < 1.0:
            probs = torch.softmax(top_vals, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            to_remove = cumulative_probs > top_p
            to_remove[1:] = to_remove[:-1].clone()
            to_remove[0] = False
            top_vals = top_vals.masked_fill(to_remove, float("-inf"))
        probs = torch.softmax(top_vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(top_idx[choice].item())

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        to_remove = cumulative_probs > top_p
        to_remove[1:] = to_remove[:-1].clone()
        to_remove[0] = False
        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(sorted_indices[choice].item())

    probs = torch.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
