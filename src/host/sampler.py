"""Next-token sampling from logits using temperature and top-p (nucleus) filtering."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> int:
    """Select the next token ID from `[1, vocab_size]` or `[vocab_size]` logits."""
    if logits.dim() == 2:
        logits = logits[0]
    logits = logits.to(dtype=torch.float32, device="cpu")

    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature

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
