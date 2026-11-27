"""Tests for peer-side TOKEN sampling and bounded KV cache eviction."""

from __future__ import annotations

from pathlib import Path
import threading

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.common.messages import MessageType, activate_msg
from src.common.tensor_codec import encode_tensor
from src.common.transport import connect, listen
from src.model.engine import KVCacheStore, ShardEngine
from src.model.loader import load_model_shard
from src.model.shard import LayerRange
from src.peer.worker import ShardWorker


def test_kv_cache_store_evicts_oldest_session() -> None:
    store = KVCacheStore(max_sessions=2)
    c1 = store.get_or_create("s1")
    c2 = store.get_or_create("s2")
    assert store.get_or_create("s1") is c1
    c3 = store.get_or_create("s3")
    assert "s1" not in store._caches
    assert store.get_or_create("s2") is c2
    assert store.get_or_create("s3") is c3


def test_worker_peer_side_sampling_returns_compact_token_message(tmp_path: Path) -> None:
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    full_model = Qwen2ForCausalLM(cfg).eval()
    full_model.save_pretrained(tmp_path, safe_serialization=True)

    shard0 = load_model_shard(tmp_path, LayerRange(0, 0), is_first=True, is_last=False, device_pref="cpu")
    shard1 = load_model_shard(tmp_path, LayerRange(1, 1), is_first=False, is_last=True, device_pref="cpu")
    eng0 = ShardEngine(shard0)
    eng1 = ShardEngine(shard1)

    listener = listen("127.0.0.1", 0)
    port = listener._sock.getsockname()[1]

    def peer_task() -> None:
        conn, _ = listener.accept(timeout=5.0)
        try:
            worker = ShardWorker(
                peer_name="p1",
                engine=eng1,
                is_last=True,
                host_conn=conn,
                relay_via_host=True,
            )
            for _ in range(2):
                msg = conn.recv_message()
                worker.process_activate(msg)
        finally:
            conn.close()

    t = threading.Thread(target=peer_task, daemon=True)
    t.start()

    try:
        host_conn = connect("127.0.0.1", port, timeout=5.0, transport="ws")
        try:
            # Step 1: prefill with greedy sampling config -> peer returns TOKEN
            input_ids = torch.tensor([[1, 5, 9]], dtype=torch.long)
            act0 = eng0.forward_shard(eng0.embed_tokens(input_ids), session_id="s1", token_pos=0)
            b0, s0, d0 = encode_tensor(act0)
            host_conn.send_message(
                activate_msg(
                    "s1",
                    token_pos=0,
                    shape=s0,
                    dtype=d0,
                    payload=b0,
                    sample={"t": 0.0, "p": 1.0, "k": 0, "r": 1.0},
                )
            )
            rep1 = host_conn.recv_message()
            assert rep1.msg_type == MessageType.TOKEN
            assert rep1.payload == b""
            tok1 = int(rep1.header["token_id"])

            # Verify greedy token matches full model reference logits
            with torch.inference_mode():
                ref1 = full_model(input_ids, use_cache=True)
                ref_tok1 = int(torch.argmax(ref1.logits[:, -1, :]).item())
            assert tok1 == ref_tok1

            # Step 2: decode step without resending sample header -> peer reuses cached sample config
            next_in = torch.tensor([[tok1]], dtype=torch.long)
            act1 = eng0.forward_shard(eng0.embed_tokens(next_in), session_id="s1", token_pos=3)
            b1, s1, d1 = encode_tensor(act1)
            host_conn.send_message(
                activate_msg("s1", token_pos=3, shape=s1, dtype=d1, payload=b1)
            )
            rep2 = host_conn.recv_message()
            assert rep2.msg_type == MessageType.TOKEN
            tok2 = int(rep2.header["token_id"])

            with torch.inference_mode():
                ref2 = full_model(next_in, past_key_values=ref1.past_key_values, use_cache=True)
                ref_tok2 = int(torch.argmax(ref2.logits[:, -1, :]).item())
            assert tok2 == ref_tok2
        finally:
            host_conn.close()
        t.join(timeout=5.0)
    finally:
        listener.close()
