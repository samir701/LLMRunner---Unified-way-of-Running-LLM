"""Single chat session lifecycle: prompt -> token IDs -> pipeline -> streamed answer."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
import uuid
from typing import Any, Iterator

import torch

from src.common.config import GenerationConfig
from src.common.messages import Message, MessageType, activate_msg
from src.common.sampler import sample_next_token
from src.common.tensor_codec import decode_tensor, encode_tensor
from src.host.coordinator import Coordinator
from src.model.engine import ShardEngine
from src.model.loader import load_generation_eos_ids, load_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Latency and payload breakdown for the most recent `generate()` call."""

    steps: int = 0
    host_ms: float = 0.0
    peer_ms: float | None = None
    tunnel_ms: float = 0.0
    rx_mode: str = "TOKEN"
    rx_payload_bytes: int = 0


class ChatSession:
    """Runs one multi-turn chat session across the host + peer pipeline."""

    def __init__(self, coordinator: Coordinator, gen_config: GenerationConfig) -> None:
        if coordinator.host_shard is None:
            raise RuntimeError("Coordinator must be started before creating a ChatSession.")
        self.coordinator = coordinator
        self.gen_config = gen_config
        self.engine = ShardEngine(coordinator.host_shard)
        self.tokenizer = load_tokenizer(coordinator.config.model)
        self._model_eos_ids = load_generation_eos_ids(coordinator.config.model)
        self.history: list[dict[str, str]] = []
        self.last_stats: GenerationStats = GenerationStats()

    def _run_pipeline_step(
        self,
        token_ids: torch.Tensor,
        session_id: str,
        token_pos: int,
        sample: dict[str, Any] | None = None,
    ) -> tuple[Message, float, float]:
        """Embed `token_ids`, run host shard, route through peers, and return `(reply, host_ms, net_plus_peer_ms)`."""
        t0 = time.perf_counter()
        hidden = self.engine.embed_tokens(token_ids)
        activations = self.engine.forward_shard(hidden, session_id=session_id, token_pos=token_pos)
        payload, shape, dtype = encode_tensor(activations)
        t1 = time.perf_counter()

        if self.coordinator.use_relay:
            reply = self._step_host_relay(session_id, token_pos, shape, dtype, payload, sample=sample)
        else:
            reply = self._step_direct(session_id, token_pos, shape, dtype, payload, sample=sample)
        t2 = time.perf_counter()
        return reply, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    def _step_direct(
        self,
        session_id: str,
        token_pos: int,
        shape: list[int],
        dtype: str,
        payload: bytes,
        sample: dict[str, Any] | None = None,
    ) -> Message:
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
                    sample=sample,
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
                f"Failed receiving response from '{last_peer.target.name}': {exc}"
            ) from exc

        if reply.msg_type == MessageType.ERROR:
            src = reply.header.get("source", last_peer.target.name)
            detail = reply.header.get("detail", "unknown error")
            raise RuntimeError(f"Pipeline error from '{src}': {detail}")
        if reply.msg_type not in (MessageType.TOKEN, MessageType.LOGITS):
            raise RuntimeError(
                f"Expected TOKEN or LOGITS from '{last_peer.target.name}', got {reply.msg_type.value}."
            )
        return reply

    def _step_host_relay(
        self,
        session_id: str,
        token_pos: int,
        shape: list[int],
        dtype: str,
        payload: bytes,
        sample: dict[str, Any] | None = None,
    ) -> Message:
        """Host-relayed pipeline for WAN/NAT peers: Host <-> Peer 1, Host <-> Peer 2, ..."""
        cur_shape = shape
        cur_dtype = dtype
        cur_payload: bytes | bytearray = payload

        for peer in self.coordinator.peers:
            try:
                peer.conn.send_message(
                    activate_msg(
                        session_id=session_id,
                        token_pos=token_pos,
                        shape=cur_shape,
                        dtype=cur_dtype,
                        payload=cur_payload,
                        sample=sample if peer.is_last else None,
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
                if reply.msg_type not in (MessageType.TOKEN, MessageType.LOGITS):
                    raise RuntimeError(
                        f"Expected TOKEN or LOGITS from '{peer.target.name}', got {reply.msg_type.value}."
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

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Assemble recent conversation turns plus the current prompt."""
        max_msgs = self.gen_config.history_turns * 2
        recent = self.history[-max_msgs:] if max_msgs > 0 else []
        return [*recent, {"role": "user", "content": prompt}]

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Apply chat template (universally supporting models with/without system role)."""
        turns = self._build_messages(prompt)
        sys_prompt = self.gen_config.system_prompt.strip()

        if getattr(self.tokenizer, "chat_template", None):
            if sys_prompt:
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": sys_prompt}, *turns],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    return self.tokenizer.encode(
                        formatted,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                except Exception:
                    # Fallback for templates without a system role (e.g. Gemma / older Mistral).
                    turns = list(turns)
                    turns[0] = {
                        "role": "user",
                        "content": f"{sys_prompt}\n\n{turns[0]['content']}",
                    }

            formatted = self.tokenizer.apply_chat_template(
                turns,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer.encode(
                formatted,
                add_special_tokens=False,
                return_tensors="pt",
            )

        # Fallback for base models without a chat template.
        parts = [f"System: {sys_prompt}"] if sys_prompt else []
        for msg in turns:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role_label}: {msg['content']}")
        parts.append("Assistant:")
        return self.tokenizer.encode("\n\n".join(parts), return_tensors="pt")

    def _stop_token_ids(self) -> set[int]:
        """Collect universal EOS / end-of-turn token IDs across model families."""
        stop_ids: set[int] = set(self._model_eos_ids)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            stop_ids.add(eos)
        elif isinstance(eos, (list, tuple, set)):
            stop_ids.update(int(x) for x in eos)

        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        for special in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "</s>"):
            tid = self.tokenizer.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                stop_ids.add(tid)
        return stop_ids

    def _extract_next_token(self, reply: Message, generated_ids: list[int]) -> int:
        if reply.msg_type == MessageType.TOKEN:
            return int(reply.header["token_id"])
        logits = decode_tensor(
            reply.payload,
            shape=reply.header["shape"],
            dtype=reply.header["dtype"],
            device="cpu",
        )
        return sample_next_token(
            logits,
            temperature=self.gen_config.temperature,
            top_p=self.gen_config.top_p,
            top_k=self.gen_config.top_k,
            repetition_penalty=self.gen_config.repetition_penalty,
            generated_ids=generated_ids,
        )

    def generate(self, prompt: str) -> Iterator[str]:
        """Tokenize prompt, stream tokens across the pipeline, and record conversation turn."""
        session_id = uuid.uuid4().hex[:12]
        self.engine.clear_session(session_id)

        input_ids = self._encode_prompt(prompt)
        if input_ids.numel() == 0:
            return

        token_pos = 0
        generated_ids: list[int] = []
        prev_text = ""
        stop_ids = self._stop_token_ids()
        sample_cfg = {
            "t": self.gen_config.temperature,
            "p": self.gen_config.top_p,
            "k": self.gen_config.top_k,
            "r": self.gen_config.repetition_penalty,
        }

        # Hold pipeline_io once for the entire generation so background heartbeats
        # never stall streaming between tokens with extra PING/PONG round-trips.
        steps = 0
        total_host_ms = 0.0
        total_peer_ms = 0.0
        peer_ms_count = 0
        total_tunnel_ms = 0.0
        last_rx_mode = "TOKEN"
        total_rx_bytes = 0

        with self.coordinator.pipeline_io():
            reply, h_ms, np_ms = self._run_pipeline_step(
                input_ids,
                session_id=session_id,
                token_pos=token_pos,
                sample=sample_cfg,
            )
            token_pos += input_ids.shape[1]

            for _ in range(self.gen_config.max_tokens):
                steps += 1
                total_host_ms += h_ms
                p_ms = reply.header.get("peer_ms")
                if p_ms is not None:
                    total_peer_ms += float(p_ms)
                    peer_ms_count += 1
                    total_tunnel_ms += max(0.0, np_ms - float(p_ms))
                else:
                    total_tunnel_ms += np_ms
                last_rx_mode = reply.msg_type.value
                total_rx_bytes += len(reply.payload)

                next_id = self._extract_next_token(reply, generated_ids)
                if next_id in stop_ids:
                    break

                generated_ids.append(next_id)
                full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                delta = full_text[len(prev_text) :]
                prev_text = full_text
                if delta:
                    yield delta

                # Next token only; KV caches and sampling config are cached on host and peers.
                next_tensor = torch.tensor([[next_id]], dtype=torch.long)
                reply, h_ms, np_ms = self._run_pipeline_step(
                    next_tensor,
                    session_id=session_id,
                    token_pos=token_pos,
                )
                token_pos += 1

        self.last_stats = GenerationStats(
            steps=steps,
            host_ms=total_host_ms,
            peer_ms=(total_peer_ms if peer_ms_count > 0 else None),
            tunnel_ms=total_tunnel_ms,
            rx_mode=last_rx_mode,
            rx_payload_bytes=total_rx_bytes,
        )

        if prev_text.strip() and self.gen_config.history_turns > 0:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": prev_text.strip()})

    def clear_history(self) -> None:
        self.history.clear()

    def clear_caches(self) -> None:
        self.engine.clear_session()
