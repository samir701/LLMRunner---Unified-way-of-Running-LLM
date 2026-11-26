"""End-to-end smoke test: load a tiny Qwen2 safetensors model split across 3 shards."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.common.tensor_codec import decode_tensor, encode_tensor
from src.model.engine import ShardEngine
from src.model.loader import load_model_shard
from src.model.shard import LayerRange


def test_qwen2_three_shard_pipeline_matches_full_model(tmp_path: Path) -> None:
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    full_model = Qwen2ForCausalLM(cfg).eval()
    full_model.save_pretrained(tmp_path, safe_serialization=True)

    shard0 = load_model_shard(tmp_path, LayerRange(0, 0), is_first=True, is_last=False, device_pref="cpu")
    shard1 = load_model_shard(tmp_path, LayerRange(1, 2), is_first=False, is_last=False, device_pref="cpu")
    shard2 = load_model_shard(tmp_path, LayerRange(3, 3), is_first=False, is_last=True, device_pref="cpu")

    eng0 = ShardEngine(shard0)
    eng1 = ShardEngine(shard1)
    eng2 = ShardEngine(shard2)

    input_ids = torch.tensor([[1, 5, 9]], dtype=torch.long)

    # Full-model reference logits (prefill + 1 decode step).
    with torch.inference_mode():
        ref_out1 = full_model(input_ids, use_cache=True)
        ref_logits1 = ref_out1.logits[:, -1, :]
        next_tok = torch.tensor([[7]], dtype=torch.long)
        ref_out2 = full_model(next_tok, past_key_values=ref_out1.past_key_values, use_cache=True)
        ref_logits2 = ref_out2.logits[:, -1, :]

    # Sharded pipeline step 1 (prefill).
    h0 = eng0.embed_tokens(input_ids)
    a0 = eng0.forward_shard(h0, session_id="s1", token_pos=0)
    b0, s0, d0 = encode_tensor(a0)
    a1 = eng1.forward_shard(decode_tensor(b0, s0, d0), session_id="s1", token_pos=0)
    b1, s1, d1 = encode_tensor(a1)
    logits1 = eng2.forward_shard(decode_tensor(b1, s1, d1), session_id="s1", token_pos=0)

    assert logits1.shape == (1, cfg.vocab_size)
    assert torch.allclose(logits1, ref_logits1, atol=5e-2, rtol=5e-2)

    # Sharded pipeline step 2 (decode 1 token reusing KV cache).
    h0_step2 = eng0.embed_tokens(next_tok)
    a0_step2 = eng0.forward_shard(h0_step2, session_id="s1", token_pos=3)
    b0_s2, s0_s2, d0_s2 = encode_tensor(a0_step2)
    a1_step2 = eng1.forward_shard(decode_tensor(b0_s2, s0_s2, d0_s2), session_id="s1", token_pos=3)
    b1_s2, s1_s2, d1_s2 = encode_tensor(a1_step2)
    logits2 = eng2.forward_shard(decode_tensor(b1_s2, s1_s2, d1_s2), session_id="s1", token_pos=3)

    assert logits2.shape == (1, cfg.vocab_size)
    assert torch.allclose(logits2, ref_logits2, atol=5e-2, rtol=5e-2)
