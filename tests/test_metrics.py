"""Tests for planning-quality metrics and for the prompt-echo measurement bug.

Two things are guarded here.

1. The metrics module itself: each of P/S/V/H must respond to the thing it
   claims to measure.  A metric that cannot distinguish a task-blind plan
   from a task-faithful one is worse than no metric, so the tests use
   hand-written plans with known ground truth rather than model output.

2. The measurement bug that produced a wrong number during development:
   decoding the full sequence and stripping the prompt by string match.  The
   model echoes the prompt, so the strip silently returned the echo and every
   metric read as a hard zero.  ``generate_plan`` decodes only the new ids;
   ``test_generate_plan_returns_only_new_text`` pins that behaviour.
"""
from __future__ import annotations

import torch

from forge.agents.act import GRAMMAR_PROMPT, plan_user_turn
from forge.data import PLAN_TASKS, slugify
from forge.tokenizer import BPETokenizer
from forge.training.metrics import (
    PlanMetrics,
    _machine,
    evaluate_planner,
    generate_plan,
    score_plan,
)

TASK = "record the threat model"
SLUG = slugify(TASK)


def _corpus():
    return "\n\n".join(
        f"{GRAMMAR_PROMPT}\n\n{plan_user_turn(t, 3)}\n\n"
        f"goal: {slugify(t)}\nmkdir docs\n"
        f"write docs/{slugify(t)}.md <<<\n- a\n>>>\n<eos>"
        for t in PLAN_TASKS
    )


def test_score_plan_counts_a_faithful_plan_as_semantic():
    base, kernel = _machine()
    text = (f"goal: {SLUG}\nmkdir docs\n"
            f"write docs/{SLUG}.md <<<\n- a\n>>>")
    parsed, semantic, verified = score_plan(text, TASK, base, kernel)
    assert parsed and semantic and verified


def test_score_plan_flags_a_task_blind_plan_as_hack():
    """Parses cleanly, names a different task -> semantic False, so P and S
    disagree and the hack counter is what catches it."""
    base, kernel = _machine()
    text = ("goal: document-the-vision-pipeline\nmkdir docs\n"
            "write docs/document-the-vision-pipeline.md <<<\n- a\n>>>")
    parsed, semantic, verified = score_plan(text, TASK, base, kernel)
    assert parsed, "this plan is grammatically valid on purpose"
    assert not semantic, "goal does not match the task"
    assert verified, "it still executes: V alone cannot see this failure"


def test_score_plan_rejects_unparseable_text():
    base, kernel = _machine()
    parsed, semantic, verified = score_plan("I cannot help with that.", TASK,
                                            base, kernel)
    assert not (parsed or semantic or verified)


def test_plan_metrics_rates_are_fractions():
    m = PlanMetrics(n=4, parse=1.0, semantic=0.5, verify=0.25, hack=0.5)
    assert m.row().startswith("P 100.0%")
    assert m.as_dict()["semantic"] == 0.5


def test_generate_plan_returns_only_new_text():
    """The regression guard for the prompt-echo measurement bug."""
    tok = BPETokenizer.train(_corpus(), vocab_size=512)

    class Echo:
        """Emits the prompt verbatim, then a plan.  Mimics the real model's
        prompt-echo behaviour, which is what broke the string-strip
        approach."""

        def generate(self, idx, max_new_tokens=90, temperature=0.7, top_k=40,
                     eos_id=None):
            prompt_ids = idx[0].tolist()
            plan_ids = tok.encode(f"goal: {SLUG}\nmkdir docs", add_bos=False)
            return torch.tensor([prompt_ids + plan_ids])

    out = generate_plan(Echo(), tok, TASK)
    assert GRAMMAR_PROMPT not in out, "prompt echo leaked into the measurement"
    assert out.startswith("goal:")


def test_evaluate_planner_reports_all_four_metrics():
    tok = BPETokenizer.train(_corpus(), vocab_size=512)

    class Perfect:
        def generate(self, idx, max_new_tokens=90, temperature=0.7, top_k=40,
                     eos_id=None):
            # Recover the task from the prompt so the plan is faithful.
            prompt = tok.decode(idx[0].tolist(), skip_specials=True)
            task = PLAN_TASKS[0]
            for t in PLAN_TASKS:
                if t in prompt:
                    task = t
                    break
            s = slugify(task)
            ids = tok.encode(f"goal: {s}\nmkdir docs\n"
                             f"write docs/{s}.md <<<\n- a\n>>>", add_bos=False)
            return torch.tensor([idx[0].tolist() + ids])

    m = evaluate_planner(Perfect(), tok, tasks=[PLAN_TASKS[0]], n=3)
    assert m.parse == 1.0 and m.semantic == 1.0 and m.verify == 1.0
    assert m.hack == 0.0


def test_hack_rate_is_parse_minus_semantic():
    """H is defined as the parsed-but-task-blind slice, so it can never
    exceed P and P = S + H must hold on any single evaluation."""
    tok = BPETokenizer.train(_corpus(), vocab_size=512)

    class Blind:
        def generate(self, idx, max_new_tokens=90, temperature=0.7, top_k=40,
                     eos_id=None):
            ids = tok.encode("goal: document-the-vision-pipeline\nmkdir docs\n"
                             "write docs/document-the-vision-pipeline.md <<<\n- a\n>>>",
                             add_bos=False)
            return torch.tensor([idx[0].tolist() + ids])

    m = evaluate_planner(Blind(), tok, tasks=[TASK], n=4)
    assert m.parse == 1.0
    assert m.semantic == 0.0
    assert m.hack == 1.0
    assert abs((m.semantic + m.hack) - m.parse) < 1e-9