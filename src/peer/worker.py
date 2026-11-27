"""Peer worker: runs this peer's layer shard and forwards activations, sampled tokens, or logits."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.common.messages import (
    Message,
    activate_msg,
    error_msg,
    logits_msg,
    token_msg,
)
from src.common.sampler import sample_next_token
from src.common.tensor_codec import decode_tensor, encode_tensor
from src.common.transport import TCPConnection
from src.model.engine import ShardEngine

logger = logging.getLogger(__name__)


class ShardWorker:
    """Executes forward passes on incoming activations for a peer."""

    def __init__(
        self,
        peer_name: str,
        engine: ShardEngine,
        is_last: bool,
        host_conn: TCPConnection,
        next_conn: TCPConnection | None = None,
        relay_via_host: bool = False,
        max_sessions: int = 8,
    ) -> None:
        self.peer_name = peer_name
        self.engine = engine
        self.is_last = is_last
        self.host_conn = host_conn
        self.next_conn = next_conn
        self.relay_via_host = relay_via_host
        self._max_sessions = max(1, max_sessions)
        self._session_sampling: dict[str, dict[str, Any]] = {}
        self._session_generated_ids: dict[str, list[int]] = {}

    def _record_session_sampling(self, session_id: str, sample_hdr: dict[str, Any]) -> None:
        while len(self._session_sampling) >= self._max_sessions and session_id not in self._session_sampling:
            oldest = next(iter(self._session_sampling))
            self._session_sampling.pop(oldest, None)
            self._session_generated_ids.pop(oldest, None)
        self._session_sampling[session_id] = sample_hdr
        self._session_generated_ids[session_id] = []

    def process_activate(self, msg: Message) -> None:
        """Decode activations, run local shard, and forward to next peer or host."""
        session_id = str(msg.header["session_id"])
        token_pos = int(msg.header["token_pos"])
        shape = msg.header["shape"]
        dtype = str(msg.header["dtype"])
        sample_hdr = msg.header.get("sample")

        try:
            t_start = time.perf_counter()
            hidden = decode_tensor(
                msg.payload,
                shape=shape,
                dtype=dtype,
                device=self.engine.shard.device,
            )
            out = self.engine.forward_shard(
                hidden,
                session_id=session_id,
                token_pos=token_pos,
            )

            if self.is_last:
                if sample_hdr is not None:
                    self._record_session_sampling(session_id, sample_hdr)
                active_sample = self._session_sampling.get(session_id)
                if active_sample is not None:
                    gen_ids = self._session_generated_ids.setdefault(session_id, [])
                    next_id = sample_next_token(
                        out,
                        temperature=float(active_sample.get("t", 0.5)),
                        top_p=float(active_sample.get("p", 0.9)),
                        top_k=int(active_sample.get("k", 40)),
                        repetition_penalty=float(active_sample.get("r", 1.1)),
                        generated_ids=gen_ids,
                    )
                    gen_ids.append(next_id)
                    peer_ms = (time.perf_counter() - t_start) * 1000.0
                    self.host_conn.send_message(
                        token_msg(
                            session_id=session_id,
                            token_pos=token_pos,
                            token_id=next_id,
                            peer_ms=peer_ms,
                        )
                    )
                    return

                out_bytes, out_shape, out_dtype = encode_tensor(out)
                peer_ms = (time.perf_counter() - t_start) * 1000.0
                self.host_conn.send_message(
                    logits_msg(
                        session_id=session_id,
                        token_pos=token_pos,
                        shape=out_shape,
                        dtype=out_dtype,
                        payload=out_bytes,
                        peer_ms=peer_ms,
                    )
                )
            elif self.relay_via_host:
                out_bytes, out_shape, out_dtype = encode_tensor(out)
                # Return intermediate activations to host so host relays to next NAT/WAN peer.
                self.host_conn.send_message(
                    activate_msg(
                        session_id=session_id,
                        token_pos=token_pos,
                        shape=out_shape,
                        dtype=out_dtype,
                        payload=out_bytes,
                    )
                )
            else:
                if self.next_conn is None:
                    raise RuntimeError("Non-last peer is missing connection to next peer.")
                out_bytes, out_shape, out_dtype = encode_tensor(out)
                self.next_conn.send_message(
                    activate_msg(
                        session_id=session_id,
                        token_pos=token_pos,
                        shape=out_shape,
                        dtype=out_dtype,
                        payload=out_bytes,
                        sample=sample_hdr,
                    )
                )
        except Exception as exc:
            logger.exception("Peer '%s' failed processing ACTIVATE", self.peer_name)
            try:
                self.host_conn.send_message(error_msg(self.peer_name, str(exc)))
            except Exception:
                pass
            raise

    def clear_caches(self) -> None:
        """Clear all stored KV caches and sampling state on this peer."""
        self.engine.clear_session()
        self._session_sampling.clear()
        self._session_generated_ids.clear()
