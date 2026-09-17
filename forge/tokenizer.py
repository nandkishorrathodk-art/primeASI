"""Byte-level tokenizer.

Deliberately dependency-free: a byte tokenizer always works, needs no training
corpus, and keeps the vocabulary at 256 + specials so a CPU can afford the
output head.  A BPE trainer can be dropped in behind the same interface.
"""
from __future__ import annotations

from typing import Iterable, Sequence

SPECIALS = ["<pad>", "<bos>", "<eos>", "<image>"]
PAD, BOS, EOS, IMAGE = 0, 1, 2, 3


class ByteTokenizer:
    """Maps text <-> ids.  ids 0..3 are specials, 4..259 are raw bytes."""

    def __init__(self) -> None:
        self.vocab_size = 256 + len(SPECIALS)
        self._byte_offset = len(SPECIALS)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = [self._byte_offset + b for b in text.encode("utf-8")]
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        out = bytearray()
        for i in ids:
            if i < self._byte_offset:
                if skip_specials:
                    continue
                out.extend(b" ")
                continue
            out.append(i - self._byte_offset)
        return out.decode("utf-8", errors="replace")

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]


TOKENIZER = ByteTokenizer()