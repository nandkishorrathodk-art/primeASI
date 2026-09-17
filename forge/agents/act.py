"""The action channel: how a debate turns into something that happens.

Before this module, the framework had three islands that had never been
connected: a model that trains, agents that debate, and a kernel that can
operate a filesystem.  This is the wire between the second and the third.

The contract, and the reason this is not a thin wrapper:

1. **Nothing acts until the debate concludes.**  The orchestrator must have
   reached an ACCEPT ruling, every proposal must have survived the policy
   gate, and the plan must parse.  A debate that is still arguing has no
   authority over the machine.

2. **The model is asked for a plan, not for an opinion.**  Proposals are
   prose or JSON; the machine needs typed steps.  So the channel issues a
   second, narrower request to the same backend, in the plan grammar.  If
   that output does not parse, the channel degrades to a rules-derived plan
   and **records the fallback in the audit chain** -- visible, never silent.

3. **One run per plan, one outcome, one audit record.**  The channel does not
   retry, does not loop, and does not decide for itself what to do next.  It
   performs one judged, simulated, capability-checked act and reports exactly
   what the machine did.

The narrow second request matters more than it looks.  A 2.6M-parameter
model cannot be asked "write a full plan" and deliver; it *can* deliver if it
is asked a small, strongly-constrained question.  The channel is where that
constraint lives.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from forge.agents.backends import Backend, RuleBackend
from forge.agents.blackboard import Blackboard
from forge.agents.protocol import Evidence, Kind, Message, Ruling, Verdict
from forge.control.bridge import BridgeOutcome, model_drives_machine, parse_plan
from forge.control.kernel import ControlKernel
from forge.control.scope import Op

# The plan grammar, shown to the model verbatim.  Kept short on purpose: a
# small model conditions better on a few exact lines than on prose.
GRAMMAR_PROMPT = """You output only lines in this exact grammar. No prose, no code.

goal: <one short line>
mkdir <path>
write <path> <<<content>>>
append <path> <<<content>>>
read <path>
list <path>
stat <path>
delete <path>
run_tests <command-name>

Rules:
- One step per line.
- Valid <path> values: docs/notes.md, src/report.txt, notes/plan.md
- Never use an absolute path. Never use ; | & $ ` or > in a path.
- Write at most 3 steps."""

# Targets a small model can plausibly emit and that are always in scope.
SAFE_DEFAULT_PATHS = ["docs/notes.md", "notes/plan.md", "src/notes.md"]


def default_fallback_plan(task: str) -> str:
    """Each agent's stated plan, turned into grammar by local code.

    No model call happens here on purpose.  The fallback exists so an
    unparseable proposal still produces a *deterministic, auditable* action
    rather than either nothing or an unpredictable one.
    """
    slug = "".join(c if c.isalnum() else "-" for c in task.lower())[:40].strip("-")
    slug = slug or "notes"
    return (
        f"goal: {slug}\n"
        f"mkdir docs\n"
        f"write docs/{slug}.md <<<\n"
        f"# {task}\n\n"
        f"- review the task\n"
        f"- record the outcome\n"
        f">>>"
    )


@dataclass
class ChannelResult:
    rationale: str
    acted: bool = False
    model_text: str = ""
    model_parsed: bool = False      # did the *model's own* output parse?
    used_fallback: bool = False
    bridge: Optional[BridgeOutcome] = None
    run: Optional[object] = None
    steps_executed: int = 0
    verified: int = 0
    rolled_back: int = 0

    def summary(self) -> dict:
        return {
            "acted": self.acted,
            "model_parsed": self.model_parsed,
            "fallback": self.used_fallback,
            "executed": self.steps_executed,
            "verified": self.verified,
            "rolled_back": self.rolled_back,
            "rationale": self.rationale,
        }


class ActionChannel:
    """Turns an accepted debate into at most one verified machine action."""

    def __init__(
        self,
        kernel: ControlKernel,
        base_dir: str,
        backend: Optional[Backend] = None,
        fallback_plan: Optional[Callable[[str], str]] = None,
        max_steps: int = 3,
    ) -> None:
        self.kernel = kernel
        self.base_dir = base_dir
        self.backend = backend or RuleBackend()
        # The fallback is deterministic *local* rules, not a model.  Calling a
        # backend and then ignoring its reply would be theatre, so there is no
        # backend here at all -- just a function from task to plan text.
        self.fallback_plan = fallback_plan or default_fallback_plan
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    def _request_plan(self, task: str, ruling: Ruling) -> str:
        user = (
            f"Task: {task}\n"
            f"Judge rationale: {ruling.rationale}\n"
            f"Produce at most {self.max_steps} steps. First line must be the goal."
        )
        return self.backend.complete(GRAMMAR_PROMPT, user, max_tokens=160)

    # ------------------------------------------------------------------
    def act(self, task: str, ruling: Ruling, blackboard: Optional[Blackboard] = None,
            allow_act: bool = True) -> ChannelResult:
        """Perform at most one judged action for ``ruling``.

        Refuses outright unless the ruling is ACCEPT.  A REVISE or REJECT has
        no authority over the machine.
        """
        if ruling.verdict is not Verdict.ACCEPT:
            self.kernel.audit.append("channel", "refused", {
                "task": task, "verdict": ruling.verdict.value,
                "reason": "only an accepted ruling may act",
            })
            return ChannelResult(
                acted=False,
                rationale=f"no action: ruling was {ruling.verdict.value}",
            )
        if not allow_act:
            return ChannelResult(acted=False,
                                 rationale="action disabled by caller")

        model_text = self._request_plan(task, ruling)
        parsed = parse_plan(model_text, base_dir=self.base_dir,
                            max_steps=self.max_steps)
        model_parsed = parsed.ok
        used_fallback = False

        if not parsed.ok:
            # Degrade visibly.  The fallback is recorded, not hidden, because
            # a pipeline that silently swaps in rules is lying about what ran.
            errors = list(parsed.errors)
            fallback_text = self.fallback_plan(task)
            parsed = parse_plan(fallback_text, base_dir=self.base_dir,
                                max_steps=self.max_steps)
            model_text = fallback_text
            used_fallback = True
            self.kernel.audit.append("channel", "fallback", {
                "task": task,
                "reason": "model output did not parse as a plan",
                "errors": errors,
            })

        if not parsed.ok:
            self.kernel.audit.append("channel", "abandoned", {
                "task": task, "reason": "neither model nor fallback produced a plan",
            })
            return ChannelResult(acted=False, model_text=model_text,
                                 rationale="no parseable plan; nothing attempted")

        # Bound at parse time so the plan and the text agree; the bridge must
        # re-parse the same text to stay consistent with what was checked.
        bridge = model_drives_machine(model_text, self.kernel,
                                      base_dir=self.base_dir, execute=True,
                                      max_steps=self.max_steps)
        run = bridge.run
        executed = len(run.executed) if run else 0
        verified = sum(1 for o in run.executed if o.verified) if run else 0
        rolled = run.rolled_back if run else 0

        self.kernel.audit.append("channel", "channel", {
            "task": task,
            "model_parsed": model_parsed,
            "fallback": used_fallback,
            "executed": executed,
            "verified": verified,
            "rolled_back": rolled,
            "goal": parsed.plan.goal if parsed.plan else None,
        })

        if blackboard is not None and run is not None:
            blackboard.write(
                "action_result", run.summary(), author="action-channel",
                evidence=[Evidence("trace", run.audit_head or "no-head",
                                   f"executed={executed} verified={verified}")],
            )

        return ChannelResult(
            acted=bool(executed),
            rationale=(
                f"executed {executed}, verified {verified}"
                + (f", rolled back {rolled}" if rolled else "")
            ),
            model_text=model_text,
            model_parsed=model_parsed,
            used_fallback=used_fallback,
            bridge=bridge,
            run=run,
            steps_executed=executed,
            verified=verified,
            rolled_back=rolled,
        )


def plan_request_prompt() -> str:
    """Exposed so the model's training corpus can include this exact grammar."""
    return GRAMMAR_PROMPT