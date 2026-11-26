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
    """Select the next token ID from `[1, vocab_size]` or `[vocab_size]` logits."""
    if logits.dim() == 2:
        logits = logits[0]
    logits = logits.to(dtype=torch.float32, device="cpu").clone()

    if repetition_penalty > 1.0 and generated_ids:
        seen = torch.tensor(list(set(generated_ids)), dtype=torch.long)
        scores = logits[seen]
        logits[seen] = torch.where(
            scores < 0,
            scores * repetition_penalty,
            scores / repetition_penalty,
        )

    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature

    if top_k > 0 and top_k < scaled.numel():
        top_vals, _ = torch.topk(scaled, top_k)
        cutoff = top_vals[-1]
        scaled[scaled < cutoff] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Remove tokens with cumulative probability above top_p (always keep top-1).
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled[indices_to_remove] = float("-inf")

    probs = torch.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
