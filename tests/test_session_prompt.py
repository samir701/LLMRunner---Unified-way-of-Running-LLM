"""Tests for universal chat formatting and multi-turn history in ChatSession."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.common.config import GenerationConfig
from src.host.session import ChatSession


class _FakeTokenizer:
    def __init__(self, reject_system: bool = False) -> None:
        self.chat_template = "fake"
        self.eos_token_id = 2
        self.unk_token_id = 0
        self.reject_system = reject_system
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if self.reject_system and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        self.last_messages = messages
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def encode(self, text: str, add_special_tokens: bool = True, return_tensors: str = "pt"):
        return torch.tensor([[len(text)]], dtype=torch.long)

    def convert_tokens_to_ids(self, tok: str):
        return -1


def _make_bare_session(tokenizer, gen_cfg: GenerationConfig) -> ChatSession:
    sess = object.__new__(ChatSession)
    sess.gen_config = gen_cfg
    sess.tokenizer = tokenizer
    sess._model_eos_ids = set()
    sess.history = []
    return sess


def test_encode_prompt_uses_system_prompt_and_history() -> None:
    tok = _FakeTokenizer(reject_system=False)
    cfg = GenerationConfig(system_prompt="Be concise.", history_turns=2)
    sess = _make_bare_session(tok, cfg)
    sess.history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    sess._encode_prompt("Full form of API?")
    assert tok.last_messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Full form of API?"},
    ]


def test_encode_prompt_falls_back_when_template_has_no_system_role() -> None:
    tok = _FakeTokenizer(reject_system=True)
    cfg = GenerationConfig(system_prompt="Be concise.", history_turns=2)
    sess = _make_bare_session(tok, cfg)

    sess._encode_prompt("Full form of API?")
    assert tok.last_messages == [
        {"role": "user", "content": "Be concise.\n\nFull form of API?"},
    ]
