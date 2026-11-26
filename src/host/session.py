"""Single chat session lifecycle: prompt -> token IDs -> pipeline -> streamed answer."""

from __future__ import annotations

import logging
import uuid
from typing import Iterator

import torch

from src.common.config import GenerationConfig
from src.common.messages import MessageType, activate_msg
from src.common.tensor_codec import decode_tensor, encode_tensor
from src.host.coordinator import Coordinator
from src.host.sampler import sample_next_token
from src.model.engine import ShardEngine
from src.model.loader import load_tokenizer

logger = logging.getLogger(__name__)


class ChatSession:
    """Runs one prompt-response generation loop across the host + peer pipeline."""

    def __init__(self, coordinator: Coordinator, gen_config: GenerationConfig) -> None:
        if coordinator.host_shard is None:
            raise RuntimeError("Coordinator must be started before creating a ChatSession.")
        self.coordinator = coordinator
        self.gen_config = gen_config
        self.engine = ShardEngine(coordinator.host_shard)
        self.tokenizer = load_tokenizer(coordinator.config.model)

    def _run_pipeline_step(
        self,
        token_ids: torch.Tensor,
        session_id: str,
        token_pos: int,
    ) -> torch.Tensor:
        """Embed `token_ids`, run host shard, route through peers, and decode LOGITS."""
        hidden = self.engine.embed_tokens(token_ids)
        activations = self.engine.forward_shard(hidden, session_id=session_id, token_pos=token_pos)
        payload, shape, dtype = encode_tensor(activations)

        with self.coordinator.pipeline_io():
            if self.coordinator.use_relay:
                reply = self._step_host_relay(session_id, token_pos, shape, dtype, payload)
            else:
                reply = self._step_direct(session_id, token_pos, shape, dtype, payload)

        return decode_tensor(
            reply.payload,
            shape=reply.header["shape"],
            dtype=reply.header["dtype"],
            device="cpu",
        )

    def _step_direct(
        self,
        session_id: str,
        token_pos: int,
        shape: list[int],
        dtype: str,
        payload: bytes,
    ):
        """Direct V1 LAN pipeline: Host -> Peer 1 -> ... -> Last Peer -> Host."""
        first_peer = self.coordinator.first_peer
        try:
            first_peer.conn.send_message(
                activate_msg(
                    session_id=session_id,
                    token_pos=token_pos,
                    shape=shape,
                    dtype=dtype,
                    payload=payload,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed sending activations to '{first_peer.target.name}': {exc}"
            ) from exc

        last_peer = self.coordinator.last_peer
        try:
            reply = last_peer.conn.recv_message()
        except Exception as exc:
            raise RuntimeError(
                f"Failed receiving logits from '{last_peer.target.name}': {exc}"
            ) from exc

        if reply.msg_type == MessageType.ERROR:
            src = reply.header.get("source", last_peer.target.name)
            detail = reply.header.get("detail", "unknown error")
            raise RuntimeError(f"Pipeline error from '{src}': {detail}")
        if reply.msg_type != MessageType.LOGITS:
            raise RuntimeError(
                f"Expected LOGITS from '{last_peer.target.name}', got {reply.msg_type.value}."
            )
        return reply

    def _step_host_relay(
        self,
        session_id: str,
        token_pos: int,
        shape: list[int],
        dtype: str,
        payload: bytes,
    ):
        """Host-relayed pipeline for WAN/NAT peers: Host <-> Peer 1, Host <-> Peer 2, ..."""
        cur_shape = shape
        cur_dtype = dtype
        cur_payload = payload

        for peer in self.coordinator.peers:
            try:
                peer.conn.send_message(
                    activate_msg(
                        session_id=session_id,
                        token_pos=token_pos,
                        shape=cur_shape,
                        dtype=cur_dtype,
                        payload=cur_payload,
                    )
                )
                reply = peer.conn.recv_message()
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline communication failed with '{peer.target.name}': {exc}"
                ) from exc

            if reply.msg_type == MessageType.ERROR:
                src = reply.header.get("source", peer.target.name)
                detail = reply.header.get("detail", "unknown error")
                raise RuntimeError(f"Pipeline error from '{src}': {detail}")

            if peer.is_last:
                if reply.msg_type != MessageType.LOGITS:
                    raise RuntimeError(
                        f"Expected LOGITS from '{peer.target.name}', got {reply.msg_type.value}."
                    )
                return reply

            if reply.msg_type != MessageType.ACTIVATE:
                raise RuntimeError(
                    f"Expected ACTIVATE from '{peer.target.name}', got {reply.msg_type.value}."
                )
            cur_shape = reply.header["shape"]
            cur_dtype = reply.header["dtype"]
            cur_payload = reply.payload

        raise RuntimeError("No peers configured in pipeline.")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Apply chat template when available (for Instruct models) and tokenize."""
        if getattr(self.tokenizer, "chat_template", None):
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer.encode(
                formatted,
                add_special_tokens=False,
                return_tensors="pt",
            )
        return self.tokenizer.encode(prompt, return_tensors="pt")

    def _stop_token_ids(self) -> set[int]:
        """Collect all EOS / end-of-turn token IDs for clean stopping."""
        stop_ids: set[int] = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            stop_ids.add(eos)
        elif isinstance(eos, (list, tuple, set)):
            stop_ids.update(int(x) for x in eos)

        for special in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                stop_ids.add(tid)
        return stop_ids

    def generate(self, prompt: str) -> Iterator[str]:
        """Flowchart B: tokenize prompt and stream generated text token by token."""
        session_id = uuid.uuid4().hex[:12]
        self.engine.clear_session(session_id)

        input_ids = self._encode_prompt(prompt)
        if input_ids.numel() == 0:
            return

        token_pos = 0
        generated_ids: list[int] = []
        prev_text = ""

        # Prefill step with full prompt tokens.
        logits = self._run_pipeline_step(input_ids, session_id=session_id, token_pos=token_pos)
        token_pos += input_ids.shape[1]

        stop_ids = self._stop_token_ids()
        for _ in range(self.gen_config.max_tokens):
            next_id = sample_next_token(
                logits,
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
            )
            if next_id in stop_ids:
                break

            generated_ids.append(next_id)
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta = full_text[len(prev_text) :]
            prev_text = full_text
            if delta:
                yield delta

            # Next token only; KV caches on host and all peers are reused.
            next_tensor = torch.tensor([[next_id]], dtype=torch.long)
            logits = self._run_pipeline_step(
                next_tensor,
                session_id=session_id,
                token_pos=token_pos,
            )
            token_pos += 1

    def clear_caches(self) -> None:
        self.engine.clear_session()
