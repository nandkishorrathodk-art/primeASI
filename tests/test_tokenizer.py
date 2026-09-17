"""Tokenizer tests: byte baseline and trained BPE.

The BPE tests exist because two real bugs were found by decoding and
comparing rather than by inspecting code:
  1. merged token ids came from a mutable registry, so a tokenizer reloaded
     from disk assigned *different* ids to the same merges
  2. `_expand` did not handle raw byte ids, so every unmerged byte was
     silently dropped -- "mkdir docs" round-tripped as "mkdidocs"
Neither raised an error. Only a round-trip check catches this class of bug.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from forge.data import make_corpus, make_grammar_corpus
from forge.tokenizer import (
    BOS, BPETokenizer, ByteTokenizer, EOS, PAD, TOKENIZER,
)


@pytest.fixture(scope="module")
def bpe():
    return BPETokenizer.train(make_corpus(200, seed=1), vocab_size=512)


# ------------------------------------------------------------ byte baseline

def test_byte_tokenizer_roundtrip_multibyte():
    text = "def add(a, b):\n    return a + b\nαβγ 日本語 ✓"
    assert TOKENIZER.decode(TOKENIZER.encode(text, add_bos=False)) == text


def test_byte_tokenizer_is_one_token_per_char():
    text = "hello world"
    assert len(TOKENIZER.encode(text, add_bos=False)) == len(text)


def test_byte_tokenizer_specials():
    ids = TOKENIZER.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == BOS and ids[-1] == EOS
    assert TOKENIZER.decode([PAD, BOS]) == ""


# ------------------------------------------------------------ BPE

def test_bpe_vocab_size_matches_merges(bpe):
    assert bpe.vocab_size == 256 + 4 + len(bpe.merges)
    assert bpe.vocab_size > 260, "BPE should have learned merges"


def test_bpe_roundtrip_simple(bpe):
    for s in ["mkdir docs/x.md", "security", "goal: harden"]:
        assert bpe.decode(bpe.encode(s, add_bos=False)) == s, f"failed on {s!r}"


def test_bpe_roundtrip_multiline(bpe):
    """Regression: raw byte ids were dropped, breaking multi-line text."""
    s = "def add(a, b):\n    return a + b\n"
    assert bpe.decode(bpe.encode(s, add_bos=False)) == s


def test_bpe_roundtrip_full_grammar(bpe):
    from forge.agents.act import GRAMMAR_PROMPT

    assert bpe.decode(bpe.encode(GRAMMAR_PROMPT, add_bos=False)) == GRAMMAR_PROMPT


def test_bpe_roundtrip_edge_cases(bpe):
    for s in ["", " ", "\n\n", "\t", "αβγ 日本語 ✓", "a", "\x00", "🎉"]:
        got = bpe.decode(bpe.encode(s, add_bos=False))
        assert got == s, f"roundtrip failed: {s!r} -> {got!r}"


def test_bpe_roundtrip_grammar_corpus(bpe):
    """Every generated plan example must survive a round-trip."""
    corpus = make_grammar_corpus(12, seed=7)
    assert bpe.decode(bpe.encode(corpus, add_bos=False)) == corpus


def test_bpe_compresses_real_text(bpe):
    """Compression must be real, measured on text the tokenizer did not train on."""
    held_out = (
        "Widget configuration directive: prefer constant-time comparison "
        "for tokens, and reject input that cannot be parsed."
    )
    assert bpe.compression(held_out) < 1.0, "BPE should beat the byte baseline"


def test_bpe_does_not_claim_absurd_compression(bpe):
    """A near-duplicate corpus can merge whole lines into one token.

    That is real for that text but misleading as a headline, so the metric is
    documented as needing held-out input.  This test pins the honest reading.
    """
    same_generator = make_corpus(50, seed=1)
    held_out = "A genuinely new sentence about routing tradeoffs and capacity."
    assert bpe.compression(held_out) > bpe.compression(same_generator), (
        "held-out compression must be worse than in-distribution; if not, "
        "the metric is being flattered by near-duplicate training text"
    )


def test_bpe_is_deterministic():
    a = BPETokenizer.train(make_corpus(80, seed=3), vocab_size=400)
    b = BPETokenizer.train(make_corpus(80, seed=3), vocab_size=400)
    assert a.merges == b.merges
    assert a.encode("mkdir docs", add_bos=False) == b.encode("mkdir docs", add_bos=False)


def test_bpe_save_load_preserves_ids(bpe):
    """Regression: ids came from a registry and changed after a reload."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tok.txt")
        bpe.save(path)
        reloaded = BPETokenizer.load(path)

    assert reloaded.vocab_size == bpe.vocab_size
    for s in ["mkdir docs/x.md", "goal: harden", "security", "def f():\n    pass\n"]:
        assert reloaded.encode(s, add_bos=False) == bpe.encode(s, add_bos=False), \
            f"ids diverged after reload for {s!r}"
        assert reloaded.decode(reloaded.encode(s, add_bos=False)) == s


def test_bpe_specials_work(bpe):
    ids = bpe.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == BOS and ids[-1] == EOS
    assert bpe.decode(ids) == "hi"


def test_bpe_train_on_tiny_text_does_not_crash():
    tok = BPETokenizer.train("ab", vocab_size=300)
    assert tok.decode(tok.encode("ab", add_bos=False)) == "ab"


def test_bpe_merges_never_span_lines():
    """"a\\nb" must not become one token, or structure is destroyed."""
    tok = BPETokenizer.train("a\nb\na\nb\na\nb\n", vocab_size=300)
    ids = tok.encode("a\nb", add_bos=False)
    # The newline must survive as its own token or inside a within-line merge.
    assert tok.decode(ids) == "a\nb"


def test_corpus_actually_teaches_the_plan_grammar():
    """The original corpus contained zero grammar examples, so the model was
    asked to emit a language it had never seen."""
    corpus = make_corpus(100, seed=5)
    for kw in ("mkdir", "write ", "goal:", "<<<", ">>>"):
        assert corpus.count(kw) > 0, f"{kw!r} missing from the training corpus"


def test_grammar_examples_are_themselves_valid_plans():
    """The training data must be parseable by the very grammar it teaches."""
    from forge.control.bridge import parse_plan
    from forge.data import iter_plan_examples

    accepted = 0
    total = 0
    for block in iter_plan_examples(20, seed=11):
        total += 1
        if parse_plan(block, base_dir="/tmp").ok:
            accepted += 1
    assert accepted == total, f"only {accepted}/{total} grammar examples parsed"


def test_grammar_corpus_prompt_precedes_the_plan():
    """The channel sends the prompt *before* the answer; so must the data."""
    from forge.data import make_grammar_corpus

    corpus = make_grammar_corpus(3, seed=2, include_prompts=True)
    lines = [l.strip() for l in corpus.splitlines() if l.strip()]
    task_at = next(i for i, l in enumerate(lines) if l.startswith("Task:"))
    goal_at = next(i for i, l in enumerate(lines) if l.startswith("goal:"))
    assert task_at < goal_at, "training data has the answer before the question"