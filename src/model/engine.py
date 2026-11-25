"""Forward pass execution for a layer shard and per-layer KV-cache management."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from src.model.loader import LoadedShard


class KVCacheStore:
    """Stores per-session DynamicCache instances so earlier tokens are not recomputed."""

    def __init__(self) -> None:
        self._caches: dict[str, DynamicCache] = {}

    def get_or_create(self, session_id: str) -> DynamicCache:
        cache = self._caches.get(session_id)
        if cache is None:
            cache = DynamicCache()
            self._caches[session_id] = cache
        return cache

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._caches.clear()
        else:
            self._caches.pop(session_id, None)


class ShardEngine:
    """Runs the forward pass for a single machine's LoadedShard."""

    def __init__(self, shard: LoadedShard) -> None:
        self.shard = shard
        self.kv_store = KVCacheStore()

    @torch.inference_mode()
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs on the host (`is_first=True`)."""
        if not self.shard.is_first or self.shard.embed_tokens is None:
            raise RuntimeError("embed_tokens() can only be called on the first shard (host).")
        ids = input_ids.to(device=self.shard.device)
        return self.shard.embed_tokens(ids)

    @torch.inference_mode()
    def forward_shard(
        self,
        hidden_states: torch.Tensor,
        session_id: str,
        token_pos: int,
    ) -> torch.Tensor:
        """Run all layers in this shard on `hidden_states`.

        If `is_last` is True, applies final norm and `lm_head` and returns logits
        for the last sequence position `[batch, vocab_size]`. Otherwise returns
        hidden activations `[batch, seq_len, hidden_size]`.
        """
        hidden = hidden_states.to(device=self.shard.device, dtype=self.shard.dtype)
        batch_size, seq_len, _ = hidden.shape
        cache = self.kv_store.get_or_create(session_id)

        position_ids = torch.arange(
            token_pos,
            token_pos + seq_len,
            device=self.shard.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = None
        if self.shard.rotary_emb is not None:
            position_embeddings = self.shard.rotary_emb(hidden, position_ids)

        # Causal mask for prefill when seq_len > 1.
        attention_mask = None
        if seq_len > 1:
            total_len = token_pos + seq_len
            mask = torch.full(
                (seq_len, total_len),
                fill_value=torch.finfo(self.shard.dtype).min,
                dtype=self.shard.dtype,
                device=self.shard.device,
            )
            mask = torch.triu(mask, diagonal=token_pos + 1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        for idx in range(self.shard.layer_range.start, self.shard.layer_range.end + 1):
            layer = self.shard.layers[str(idx)]
            kwargs: dict = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_value": cache,
                "use_cache": True,
            }
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings

            layer_out = layer(hidden, **kwargs)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        if self.shard.is_last:
            if self.shard.norm is not None:
                hidden = self.shard.norm(hidden)
            if self.shard.lm_head is None:
                raise RuntimeError("Last shard is missing lm_head.")
            logits = self.shard.lm_head(hidden[:, -1, :])
            return logits

        return hidden

    def clear_session(self, session_id: str | None = None) -> None:
        """Clear stored KV caches for a session (or all sessions)."""
        self.kv_store.clear(session_id)
