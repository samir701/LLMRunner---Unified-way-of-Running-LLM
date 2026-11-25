"""Peer worker: runs this peer's layer shard and forwards activations or logits."""

from __future__ import annotations

import logging

from src.common.messages import (
    Message,
    activate_msg,
    error_msg,
    logits_msg,
)
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
    ) -> None:
        self.peer_name = peer_name
        self.engine = engine
        self.is_last = is_last
        self.host_conn = host_conn
        self.next_conn = next_conn
        self.relay_via_host = relay_via_host

    def process_activate(self, msg: Message) -> None:
        """Decode activations, run local shard, and forward to next peer or host."""
        session_id = str(msg.header["session_id"])
        token_pos = int(msg.header["token_pos"])
        shape = msg.header["shape"]
        dtype = str(msg.header["dtype"])

        try:
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
            out_bytes, out_shape, out_dtype = encode_tensor(out)

            if self.is_last:
                self.host_conn.send_message(
                    logits_msg(
                        session_id=session_id,
                        token_pos=token_pos,
                        shape=out_shape,
                        dtype=out_dtype,
                        payload=out_bytes,
                    )
                )
            elif self.relay_via_host:
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
                self.next_conn.send_message(
                    activate_msg(
                        session_id=session_id,
                        token_pos=token_pos,
                        shape=out_shape,
                        dtype=out_dtype,
                        payload=out_bytes,
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
        """Clear all stored KV caches on this peer."""
        self.engine.clear_session()
