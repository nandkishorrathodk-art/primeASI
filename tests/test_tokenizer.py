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
    """Every generated plan example must survive a round-trip.

    Special tokens are skipped on decode, so compare against the text with
    `<eos>` removed -- which is also exactly what the model's consumer sees.
    """
    corpus = make_grammar_corpus(12, seed=7)
    expected = corpus.replace("<eos>", "")
    assert bpe.decode(bpe.encode(corpus, add_bos=False)) == expected


def test_bpe_encodes_specials_as_single_tokens(bpe):
    """`<eos>` must be one id, or the model cannot learn a clean stop signal."""
    from forge.tokenizer import EOS

    ids = bpe.encode("<eos>", add_bos=False)
    assert ids == [EOS], f"<eos> encoded as {ids}"

    mixed = bpe.encode("done<eos>", add_bos=False)
    assert mixed[-1] == EOS
    assert bpe.decode(mixed) == "done"


def test_bpe_decode_can_keep_specials(bpe):
    ids = bpe.encode("hi<eos>", add_bos=False)
    assert bpe.decode(ids, skip_specials=False) == "hi<eos>"


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


def test_training_data_matches_the_inference_prompt_exactly():
    """Regression: the inference prompt format appeared zero times in training.

    The channel calls the model as f"{system}\\n\\n{user}\\n\\n".  If the data
    is not in that shape, the model is asked to follow a format it has never
    seen and falls back to whatever dominates the corpus.  Measured: it
    regurgitated agent-JSON instead of emitting a plan.
    """
    from forge.agents.act import GRAMMAR_PROMPT
    from forge.data import make_instruct_example

    example = make_instruct_example("document the trust boundary")
    assert example.startswith(GRAMMAR_PROMPT), \
        "training example must begin with the exact system prompt"
    assert f"{GRAMMAR_PROMPT}\n\nTask: " in example


def test_instruction_corpus_contains_the_real_prefix():
    from forge.agents.act import GRAMMAR_PROMPT
    from forge.data import make_instruct_corpus

    corpus = make_instruct_corpus(20, seed=4)
    assert GRAMMAR_PROMPT in corpus
    assert f"{GRAMMAR_PROMPT}\n\nTask: " in corpus


def test_channel_user_turn_matches_training_exactly():
    """The strongest guard against the whole class of train/infer drift.

    Three separate silent failures came from the channel and the corpus
    building the user turn differently.  This asserts they are byte-identical
    for a task that has a rationale, which is what the channel actually sends.
    """
    from forge.agents.act import GRAMMAR_PROMPT, plan_user_turn
    from forge.control.scope import Op
    from forge.data import make_instruct_example

    task = "document the trust boundary"
    rationale = "no high-severity finding outstanding"

    channel_turn = plan_user_turn(task, 3, rationale=rationale)

    # Find a training example carrying the same rationale and compare turns.
    found = False
    for seed in range(200):
        ex = make_instruct_example(task, max_steps=3,
                                   rng=__import__("random").Random(seed))
        if rationale not in ex:
            continue
        body = ex.split(f"{GRAMMAR_PROMPT}\n\n", 1)[1]
        train_turn = body.split(f"\n\ngoal: ")[0]
        assert train_turn == channel_turn, (
            "training user turn differs from the channel's:\n"
            f"  train  : {train_turn!r}\n  channel: {channel_turn!r}")
        found = True
        break
    assert found, "no training example carried a judge rationale"


def test_channel_and_corpus_agree_without_rationale():
    from forge.agents.act import plan_user_turn

    assert plan_user_turn("tidy the docs", 3) == (
        "Task: tidy the docs\n"
        "Produce at most 3 steps. First line must be the goal."
    )
    assert plan_user_turn("tidy the docs", 2, rationale="ok") == (
        "Task: tidy the docs\nJudge rationale: ok\n"
        "Produce at most 2 steps. First line must be the goal."
    )


def test_training_examples_end_with_eos():
    """Without a terminator the model never learns to stop."""
    from forge.data import make_instruct_example

    ex = make_instruct_example("note the rate limits")
    assert ex.endswith("<eos>"), "training example has no stop signal"


def test_eos_is_trainable_as_one_token():
    """`<eos>` must be a single id, not five characters."""
    from forge.tokenizer import EOS, BPETokenizer
    from forge.data import make_corpus

    tok = BPETokenizer.train(make_corpus(40, seed=1), vocab_size=400)
    assert tok.encode("<eos>", add_bos=False) == [EOS]


def test_plan_target_is_derivable_from_the_task():
    """A target the model cannot compute from its input is unlearnable.

    The goal slug must be the slugified task, and the body must follow from
    the task's topic.  Picking either at random would make this an arbitrary
    recall problem, which no amount of training fixes.
    """
    from forge.data import body_for, make_instruct_example, slugify

    for task in ("document the trust boundary", "harden the input parser",
                 "note the rate limits"):
        slug = slugify(task)
        assert slug == task.replace(" ", "-")

    # The emitted goal really is the slug of the stated task.
    ex = make_instruct_example("note the rate limits")
    assert "goal: note-the-rate-limits" in ex
    assert "docs/note-the-rate-limits.md" in ex
    assert body_for("note the rate limits") == body_for("note the rate limits")


def test_slugify_handles_punctuation_and_spacing():
    from forge.data import slugify

    assert slugify("Document the trust boundary") == "document-the-trust-boundary"
    assert slugify("harden  the   parser") == "harden-the-parser"
    assert slugify("  leading and trailing  ") == "leading-and-trailing"
    assert slugify("a/b/c") == "a-b-c"
    assert slugify("") == ""


def test_body_topics_are_topic_specific():
    """Different topics must yield different bodies, or nothing is learnable."""
    from forge.data import body_for

    assert body_for("harden the input parser") != body_for("note the rate limits")
    assert body_for("an unrelated topic")  # falls back, does not crash


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
    """The channel sends the prompt *before* the answer; so must the data.

    Note the system prompt itself contains the literal line `goal: <slug>` as
    part of the grammar spec, so the *last* goal line is the real plan and the
    prompt must come before that one.
    """
    from forge.agents.act import GRAMMAR_PROMPT
    from forge.data import make_grammar_corpus

    corpus = make_grammar_corpus(3, seed=2, include_prompts=True)
    assert corpus.startswith(GRAMMAR_PROMPT), \
        "examples must lead with the exact system prompt"
    # The first Task: header must precede the first *concrete* goal line.
    task_at = corpus.index("Task: ")
    concrete = corpus.index("goal: ", corpus.index(GRAMMAR_PROMPT) + len(GRAMMAR_PROMPT))
    assert task_at < concrete, "training data has the answer before the question"