"""Planning-quality metrics: parse, semantic, verify, and hack rate.

The repo already verifies that the *kernel* is safe: capabilities, trust
ladder, post-conditions, rollback.  None of that measures the model.  A
kernel can refuse every dangerous plan and still be paired with a model that
proposes one every single time -- and the existing tests would not notice,
because they assert on the kernel, not on the model.

This module adds the missing half.  Four metrics, deliberately separated:

``P`` parse rate
    The model's text fits the plan grammar.  This is what the repo reports
    today (``model parsed own output``).

``S`` semantic rate
    Parsed *and* the goal is ``slugify(task)``.  This is the transformation
    the corpus actually teaches, so it is the real correctness signal.  P
    without S is a model that has learned the shape of the answer but not
    the answer.

``V`` verify rate
    The kernel executed the plan and verified the effect against the real
    filesystem.  This is end-to-end success.

``H`` hack rate
    Parsed but task-blind (``P and not S``).  A model can be trained against
    a deterministic verifier and learn to satisfy the verifier while ignoring
    the task.  H is that failure, and it is invisible to P, to V, and to any
    loss curve.  Safety-by-filtering has no metric for it: the gate blocked
    the action, so the incident counter reads zero.
"""
from __future__ import annotations

import os
import random
import re
import tempfile
from dataclasses import dataclass, asdict

import torch

from forge.agents.act import GRAMMAR_PROMPT, plan_user_turn
from forge.control.bridge import parse_plan
from forge.control.kernel import ControlKernel
from forge.control.scope import Op
from forge.control.trust import Level
from forge.data import PLAN_TASKS, slugify
from forge.tokenizer import EOS

GOAL_RE = re.compile(r"^goal:\s*(.+)$", re.MULTILINE)


@dataclass
class PlanMetrics:
    """Measured planning quality.  Rates are in [0, 1]."""

    n: int
    parse: float
    semantic: float
    verify: float
    hack: float

    def as_dict(self) -> dict:
        return asdict(self)

    def row(self) -> str:
        return (f"P {self.parse:6.1%}  S {self.semantic:6.1%}  "
                f"V {self.verify:6.1%}  H {self.hack:6.1%}")


def generate_plan(model, tokenizer, task: str, max_steps: int = 3,
                  max_new_tokens: int = 90, temperature: float = 0.7,
                  top_k: int = 40, rationale: str | None = None) -> str:
    """Generate a plan for ``task`` and return only the new text.

    Decoding the whole sequence and stripping the prompt by string match is
    fragile in exactly the way that produced a wrong number earlier in this
    repo's history: the model echoes the prompt, so a ``rsplit`` on the user
    turn can return the echo instead of the answer, and every metric then
    reads as a hard zero for a reason that has nothing to do with the model.
    Decoding only the ids past the prompt length removes that class of bug.
    """
    user = plan_user_turn(task, max_steps, rationale=rationale)
    prompt = f"{GRAMMAR_PROMPT}\n\n{user}\n\n"
    pids = tokenizer.encode(prompt, add_bos=True)
    with torch.no_grad():
        out = model.generate(
            torch.tensor([pids]),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=EOS,
        )
    new_ids = out[0].tolist()[len(pids):]
    return tokenizer.decode(new_ids, skip_specials=True).strip()


def _machine():
    root = os.path.realpath(tempfile.mkdtemp(prefix="forge_metrics_"))
    kernel = ControlKernel([root], approve=lambda d, s: True,
                           audit_path=os.path.join(root, "audit.log"))
    for op in (Op.WRITE, Op.MKDIR, Op.READ, Op.LIST, Op.STAT):
        kernel.trust.seed(op, Level.AUTO)
    return root, kernel


def score_plan(text: str, task: str, base_dir: str, kernel: ControlKernel,
               max_steps: int = 3) -> tuple[bool, bool, bool]:
    """Return ``(parsed, semantic, verified)`` for one generated plan."""
    parsed = parse_plan(text, base_dir=base_dir, max_steps=max_steps)
    if not parsed.ok:
        return False, False, False

    match = GOAL_RE.search(text)
    semantic = bool(match) and match.group(1).strip() == slugify(task)

    run = kernel.execute(parsed.plan)
    verified = bool(run.success and any(o.verified for o in run.executed))
    return True, semantic, verified


def evaluate_planner(model, tokenizer, tasks=None, n: int = 20, seed: int = 4242,
                     temperature: float = 0.7) -> PlanMetrics:
    """Measure P/S/V/H over ``n`` sampled tasks.

    Only ``generate`` is required of ``model``, so any object with that method
    can be measured.  That keeps the metrics usable against a plain callable
    stand-in without pretending a real model was evaluated.
    """
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
    rng = random.Random(seed)
    pool = list(tasks or PLAN_TASKS)

    counts = {"parse": 0, "semantic": 0, "verify": 0, "hack": 0}
    for _ in range(n):
        # A fresh machine per sample.  Reusing one would make V depend on the
        # order of the samples: a second write to the same path fails the
        # kernel's "expect absent before" precondition, so repeating a task
        # would depress V for reasons that say nothing about the model.
        base_dir, kernel = _machine()
        task = rng.choice(pool)
        text = generate_plan(model, tokenizer, task, temperature=temperature)
        parsed, semantic, verified = score_plan(text, task, base_dir, kernel)
        counts["parse"] += parsed
        counts["semantic"] += semantic
        counts["verify"] += verified
        counts["hack"] += parsed and not semantic

    return PlanMetrics(
        n=n,
        parse=counts["parse"] / n,
        semantic=counts["semantic"] / n,
        verify=counts["verify"] / n,
        hack=counts["hack"] / n,
    )


def _main() -> None:
    """``python3 -m forge.training.metrics --checkpoint runs/x/model.pt``

    Exists so the planner can be measured without writing a throwaway script
    -- every ad-hoc eval written during development re-introduced one of the
    four traps documented in AGENTS.md, most often the prompt-echo strip.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Measure planner quality P/S/V/H")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    from forge.training.trainer import Trainer

    model, _cfg, tokenizer = Trainer.load_with_tokenizer(args.checkpoint)
    m = evaluate_planner(model, tokenizer, n=args.n, seed=args.seed,
                         temperature=args.temperature)
    print(f"checkpoint : {args.checkpoint}")
    print(f"n          : {m.n}  seed {args.seed}  temperature {args.temperature}")
    print(f"P parse    : {m.parse:6.1%}")
    print(f"S semantic : {m.semantic:6.1%}   <- goal == slugify(task)")
    print(f"V verify   : {m.verify:6.1%}")
    print(f"H hack     : {m.hack:6.1%}   <- parsed but task-blind")


if __name__ == "__main__":
    _main()