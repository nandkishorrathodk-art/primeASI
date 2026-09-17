"""Tokenizers: a byte baseline and a trained byte-level BPE.

Both expose the same interface (``encode`` / ``decode`` / ``vocab_size`` /
``encode_batch``) so the model and trainer can swap between them without
knowing which one they hold.

The byte baseline always works and needs no training corpus.  BPE is what you
want when the model actually has to *produce* text: measured, the byte
tokenizer costs 1.00 tokens per character, which both wastes the context
window and gives a fixed parameter budget far less effective capacity.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Iterable, Optional, Sequence

SPECIALS = ["<pad>", "<bos>", "<eos>", "<image>"]
PAD, BOS, EOS, IMAGE = 0, 1, 2, 3

# Merged tokens are numbered from here upward, by merge rank.  Deriving ids
# from rank keeps save/load round-trips stable.
_MERGED_BASE = 256 + len(SPECIALS)
_BYTE_OFFSET = len(SPECIALS)          # raw byte b is token id b + _BYTE_OFFSET


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


class BPETokenizer:
    """Byte-level BPE: iteratively merge the most frequent adjacent pair.

    Why this exists, concretely.  The byte tokenizer costs **1.00 tokens per
    character** -- measured, not assumed.  So a 418-token system prompt does
    not fit a 128-token context, and the model is asked a question it cannot
    even see.  Worse, every character of training text is a separate token,
    which means a fixed parameter budget buys far less effective capacity.

    BPE attacks both problems: it compresses text (fewer tokens for the same
    content) and it gives common subwords their own token, so `mkdir` costs
    one token instead of five.

    Trained on the *same* corpus the model learns from, so the vocabulary and
    the task share one distribution rather than two.
    """

    def __init__(self, merges: Optional[list[tuple[int, int]]] = None) -> None:
        # merges[i] is the i-th merge performed; its rank is the merge order.
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._rank: dict[tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self._byte_offset = len(SPECIALS)
        self.vocab_size = 256 + len(SPECIALS) + len(self.merges)

    # -- training ------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 1024) -> "BPETokenizer":
        """Learn merges until ``vocab_size`` is reached or nothing repeats."""
        target_merges = max(0, vocab_size - 256 - len(SPECIALS))
        # Symbols are *token ids*, not raw byte values: raw bytes are shifted
        # up by the specials so that byte b and special 0..3 cannot collide.
        lines = [[b + _BYTE_OFFSET for b in line.encode("utf-8")] + [10 + _BYTE_OFFSET]
                 for line in text.splitlines()]
        if not any(lines):
            return cls()

        merges: list[tuple[int, int]] = []
        for _ in range(target_merges):
            counts = Counter()
            for seq in lines:
                for pair in zip(seq, seq[1:]):
                    counts[pair] += 1
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            rank = len(merges)
            merges.append(pair)
            lines = [_merge_pair(seq, pair, rank) for seq in lines]
        return cls(merges)

    # -- encoding ------------------------------------------------------
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        for line in text.splitlines(keepends=True):
            seq = [b + _BYTE_OFFSET for b in line.encode("utf-8")]
            ids.extend(self._apply(seq))
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def _apply(self, seq: list[int]) -> list[int]:
        """Apply learned merges, lowest rank first.  O(n·m) but n is small."""
        while len(seq) >= 2:
            best_rank = None
            best_pos = -1
            for i in range(len(seq) - 1):
                rank = self._rank.get((seq[i], seq[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_rank is None:
                break
            seq = _merge_pair(seq, (seq[best_pos], seq[best_pos + 1]), best_rank)
        return seq

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        raw = bytearray()
        for i in ids:
            if i < self._byte_offset:
                if skip_specials:
                    continue
                raw.extend(b" ")
            elif i < 256 + self._byte_offset:
                raw.append(i - self._byte_offset)
            else:
                raw.extend(_expand(i, self.merges))
        return raw.decode("utf-8", errors="replace")

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    # -- persistence ---------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for a, b in self.merges:
                fh.write(f"{a} {b}\n")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        merges: list[tuple[int, int]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    a, b = line.split()
                    merges.append((int(a), int(b)))
        return cls(merges)

    def compression(self, text: str) -> float:
        """Tokens per character.  Lower is better; 1.00 is the byte baseline.

        Measured on whatever text you pass.  Measuring on the training corpus
        flatters the number, so pass held-out text to judge generalisation.
        """
        n = len(self.encode(text, add_bos=False))
        return n / max(len(text), 1)


def _merge_pair(seq: list[int], pair: tuple[int, int], rank: int) -> list[int]:
    """Replace every occurrence of ``pair`` with the token id for ``rank``.

    The new id is derived from the merge's rank, not from a mutable registry,
    so a tokenizer reloaded from disk assigns identical ids to identical
    merges.  A registry would silently produce different ids after a reload.
    """
    a, b = pair
    new_id = _MERGED_BASE + rank
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _expand(token_id: int, merges: list[tuple[int, int]]) -> bytes:
    """Recursively expand a token id back to raw bytes.

    Raw byte ids live in ``_BYTE_OFFSET .. _MERGED_BASE`` and must be handled
    before the merge branch.  Skipping that check silently drops every byte
    that is not part of a merge (a round-trip of "mkdir docs" came back as
    "mkdidocs"), which is exactly the class of bug that only shows up when
    you actually decode and compare.
    """
    if token_id < _BYTE_OFFSET:
        return b""                                    # special token
    if token_id < _MERGED_BASE:
        return bytes([token_id - _BYTE_OFFSET])       # raw byte
    idx = token_id - _MERGED_BASE
    if not (0 <= idx < len(merges)):
        return b""
    a, b = merges[idx]
    return _expand(a, merges) + _expand(b, merges)


TOKENIZER = ByteTokenizer()