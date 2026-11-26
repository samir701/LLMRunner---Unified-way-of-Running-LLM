"""Tests for universal sampler controls (repetition penalty, top-k, greedy)."""

from __future__ import annotations

import torch

from src.host.sampler import sample_next_token


def test_repetition_penalty_reduces_repeated_token_preference() -> None:
    logits = torch.tensor([[10.0, 9.0, 1.0]])
    # Without penalty, token 0 wins greedily.
    assert sample_next_token(logits, temperature=0.0, repetition_penalty=1.0, generated_ids=[0]) == 0
    # With penalty 1.5, token 0 drops to 6.67 while token 1 stays 9.0.
    assert sample_next_token(logits, temperature=0.0, repetition_penalty=1.5, generated_ids=[0]) == 1


def test_top_k_restricts_sampling_pool() -> None:
    logits = torch.tensor([[10.0, 5.0, 1.0, -5.0]])
    for _ in range(10):
        tok = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=1)
        assert tok == 0
