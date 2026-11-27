"""Forward pass execution for a layer shard and per-layer KV-cache management."""

from __future__ import annotations

import inspect

import torch
from transformers.cache_utils import DynamicCache

from src.model.loader import LoadedShard


class KVCacheStore:
    """Stores bounded per-session DynamicCache instances so earlier tokens are not recomputed."""

    def __init__(self, max_sessions: int = 4) -> None:
        self._max_sessions = max(1, max_sessions)
        self._caches: dict[str, DynamicCache] = {}

    def get_or_create(self, session_id: str, reset: bool = False) -> DynamicCache:
        if reset:
            self._caches.pop(session_id, None)
        cache = self._caches.get(session_id)
        if cache is None:
            while len(self._caches) >= self._max_sessions:
                oldest_key = next(iter(self._caches))
                self._caches.pop(oldest_key, None)
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
        self._neg_inf = torch.finfo(shard.dtype).min
        # Inspect layer forward signatures once instead of on every token/layer.
        self._layer_specs: dict[str, tuple[str, bool, bool]] = {}
        for key, layer in shard.layers.items():
            fwd_params = inspect.signature(layer.forward).parameters
            cache_arg = "past_key_values" if "past_key_values" in fwd_params else "past_key_value"
            self._layer_specs[key] = (
                cache_arg,
                "cache_position" in fwd_params,
                "position_embeddings" in fwd_params,
            )

    @torch.inference_mode()
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs on the host (`is_first=True`)."""
        if not self.shard.is_first or self.shard.embed_tokens is None:
            raise RuntimeError("embed_tokens() can only be called on the first shard (host).")
        ids = input_ids if input_ids.device == self.shard.device else input_ids.to(device=self.shard.device)
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
        if hidden_states.device != self.shard.device or hidden_states.dtype != self.shard.dtype:
            hidden = hidden_states.to(device=self.shard.device, dtype=self.shard.dtype)
        else:
            hidden = hidden_states

        batch_size, seq_len, _ = hidden.shape
        cache = self.kv_store.get_or_create(session_id, reset=(token_pos == 0))

        if seq_len == 1:
            cache_position = torch.tensor(
                [token_pos],
                device=self.shard.device,
                dtype=torch.long,
            )
            position_ids = cache_position.unsqueeze(0)
            if batch_size != 1:
                position_ids = position_ids.expand(batch_size, 1)
            attention_mask = None
        else:
            cache_position = torch.arange(
                token_pos,
                token_pos + seq_len,
                device=self.shard.device,
                dtype=torch.long,
            )
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
            total_len = token_pos + seq_len
            mask = torch.full(
                (seq_len, total_len),
                fill_value=self._neg_inf,
                dtype=self.shard.dtype,
                device=self.shard.device,
            )
            mask = torch.triu(mask, diagonal=token_pos + 1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        position_embeddings = None
        if self.shard.rotary_emb is not None:
            position_embeddings = self.shard.rotary_emb(hidden, position_ids)

        for idx in range(self.shard.layer_range.start, self.shard.layer_range.end + 1):
            key = str(idx)
            layer = self.shard.layers[key]
            cache_arg, has_cache_pos, has_pos_emb = self._layer_specs[key]
            kwargs: dict = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                cache_arg: cache,
                "use_cache": True,
            }
            if has_cache_pos:
                kwargs["cache_position"] = cache_position
            if has_pos_emb and position_embeddings is not None:
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
