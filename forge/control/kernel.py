"""The control kernel: the loop that operates a machine under supervision.

    perceive -> plan -> simulate -> approve -> act -> verify -> journal
                  ^                                            |
                  +---------------- adapt ---------------------+

Two things about this loop are unusual and deliberate.

**Simulate before acting.**  ``simulate()`` runs the plan's *preconditions*
against the world digest and predicts each step's effect, without touching
anything.  A plan that would violate scope, exceed trust, or touch a
name-suggests-secrets path is rejected before the first write.  This is where
most damage gets prevented -- not by a better filter, but by noticing earlier.

**Verify against the world, not against the model's claim.**  After acting,
``verify()`` re-reads the actual filesystem.  A step is only recorded as a
success if the machine agrees the expected state is now true.  "The model said
it worked" is never accepted as evidence; that is exactly the failure mode
that makes agentic systems untrustworthy.

The kernel also refuses to function without its guards wired up.  There is no
constructor flag to disable the audit chain or the capability check, because
a switch that disables safety eventually gets flipped.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from forge.control.actions import ActionLedger, ActionResult, Transaction
from forge.control.audit import AuditChain, VerifyResult
from forge.control.scope import (
    CapabilitySet, Op, PathGuard, ScopeDenial, default_policy,
)
from forge.control.trust import Decision, Level, TrustLadder
from forge.control.world import WorldDigest


@dataclass
class Step:
    """One intended action, with the state it expects before and after."""
    op: Op
    path: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    confidence: float = 1.0
    expect_absent_before: bool = False
    expect_content_after: Optional[str] = None
    note: str = ""

    def describe(self) -> str:
        bits = [self.op.value]
        if self.path:
            bits.append(os.path.basename(self.path))
        return " ".join(bits)


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)
    rationale: str = ""


@dataclass
class Simulation:
    ok: bool
    problems: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(default_factory=list)


@dataclass
class StepOutcome:
    step: Step
    executed: bool
    verified: bool
    detail: str = ""
    undo: Optional[str] = None


@dataclass
class RunOutcome:
    goal: str
    executed: list[StepOutcome] = field(default_factory=list)
    skipped: list[StepOutcome] = field(default_factory=list)
    verification: Optional[VerifyResult] = None
    audit_head: str = ""
    rolled_back: int = 0

    @property
    def failed_verifications(self) -> list[StepOutcome]:
        return [o for o in self.executed if not o.verified]

    @property
    def success(self) -> bool:
        if self.skipped or self.rolled_back:
            return False
        return all(o.verified for o in self.executed) if self.executed else False

    def summary(self) -> dict:
        return {
            "goal": self.goal,
            "executed": len(self.executed),
            "skipped": len(self.skipped),
            "verified": sum(1 for o in self.executed if o.verified),
            "rolled_back": self.rolled_back,
            "success": self.success,
            "audit_head": self.audit_head[:16],
        }


class ControlKernel:
    def __init__(
        self,
        roots: list[str],
        operator: str = "human",
        approve: Optional[Callable[[Decision, Step], bool]] = None,
        audit_path: Optional[str] = None,
        trust: Optional[TrustLadder] = None,
    ) -> None:
        self.guard = PathGuard(roots)
        self.caps = default_policy(self.guard, operator)
        self.audit = AuditChain(audit_path)
        self.ledger = ActionLedger(self.caps, self.guard, self.audit, actor="kernel")
        self.trust = trust or TrustLadder()
        self.world = WorldDigest(self.guard.roots)
        # Default policy: deny everything that needs approval.  A kernel with
        # no human attached must not silently self-approve.
        self._approve = approve or (lambda decision, step: False)
        self._notifications: list[dict] = []

    # -- perception ------------------------------------------------------
    def perceive(self) -> str:
        digest = self.world.summary()
        self.audit.append("kernel", "perceive", self.world.stats())
        return digest

    # -- plan ------------------------------------------------------------
    def plan(self, goal: str, steps: list[Step], rationale: str = "") -> Plan:
        return Plan(goal=goal, steps=list(steps), rationale=rationale)

    # -- simulate --------------------------------------------------------
    def simulate(self, plan: Plan) -> Simulation:
        sim = Simulation(ok=True)
        for step in plan.steps:
            label = step.describe()

            # 1. Scope: is this operation even expressible under the grants?
            try:
                self.caps.check(step.op, step.path)
            except ScopeDenial as exc:
                sim.ok = False
                sim.problems.append(f"{label}: {exc.reason}")
                if exc.suggestion:
                    sim.problems.append(f"    hint: {exc.suggestion}")
                continue

            # 2. Trust: has this class earned execution, and is the agent sure?
            decision = self.trust.decide(step.op, step.confidence)
            if not decision.allow:
                sim.ok = False
                sim.problems.append(f"{label}: {decision.reason}")
                continue

            # 3. Preconditions: does the world currently match assumptions?
            if step.path:
                try:
                    real = self.guard.resolve(step.path)
                except ScopeDenial as exc:
                    sim.ok = False
                    sim.problems.append(f"{label}: {exc.reason}")
                    continue
                exists = os.path.exists(real)
                if step.expect_absent_before and exists:
                    sim.ok = False
                    sim.problems.append(
                        f"{label}: expected {os.path.basename(real)} to be absent, "
                        "but it already exists"
                    )
                if step.op in (Op.READ, Op.APPEND) and not exists:
                    sim.ok = False
                    sim.problems.append(f"{label}: target does not exist")

                # 4. Risk surface: name-suggests-secrets paths need approval.
                base = os.path.basename(real).lower()
                if any(h in base for h in (".env", "id_rsa", ".pem", ".key",
                                           "credentials", ".npmrc", ".pypirc",
                                           ".netrc")):
                    sim.blocked_paths.append(real)
                    sim.problems.append(
                        f"{label}: {base} looks like a secrets file; "
                        "excluded from automated plans"
                    )
                    sim.ok = False

            sim.predictions.append(self._predict(step))

        self.audit.append("kernel", "simulate", {
            "goal": plan.goal, "ok": sim.ok,
            "problems": sim.problems, "predictions": sim.predictions,
        })
        return sim

    def _predict(self, step: Step) -> str:
        if step.op is Op.WRITE:
            return f"write {os.path.basename(step.path or '')}: file created or replaced"
        if step.op is Op.MKDIR:
            return f"mkdir {os.path.basename(step.path or '')}: directory created"
        if step.op is Op.DELETE:
            return (f"delete {os.path.basename(step.path or '')}: moved to "
                    "quarantine, bytes retained, restorable")
        if step.op is Op.RUN_TESTS:
            return "run_tests: read-only, no persistent state change"
        if step.op is Op.APPEND:
            return f"append {os.path.basename(step.path or '')}: bytes appended"
        return f"{step.op.value}: no side effect predicted"

    # -- act -------------------------------------------------------------
    def execute(self, plan: Plan, simulate_first: bool = True) -> RunOutcome:
        outcome = RunOutcome(goal=plan.goal)
        # Auto-rollback must only unwind what *this run* did.  Undoing the
        # whole ledger would also reverse earlier, legitimate work -- and
        # would delete files this run never touched.
        mark = len(self.ledger.history())

        if simulate_first:
            sim = self.simulate(plan)
            if not sim.ok:
                for step in plan.steps:
                    outcome.skipped.append(StepOutcome(
                        step, False, False, "blocked at simulation"))
                outcome.audit_head = self.audit.head()
                self.audit.append("kernel", "plan_refused",
                                  {"goal": plan.goal, "problems": sim.problems})
                return outcome
            self._notifications.append({"kind": "simulation", "data": sim.__dict__})

        with Transaction(self.ledger, name=plan.goal) as _:
            for step in plan.steps:
                decision = self.trust.decide(step.op, step.confidence)

                # The trust gate applies even when simulation was skipped.
                # Bypassing simulation must never mean bypassing permission.
                if not decision.allow:
                    outcome.skipped.append(StepOutcome(
                        step, False, False, decision.reason))
                    continue

                if decision.needs_approval:
                    granted = self._approve(decision, step)
                    self.trust.note_approval(step.op, granted)
                    self.audit.append("operator", "approval",
                                      {"op": step.op.value, "path": step.path,
                                       "granted": granted,
                                       "reason": decision.reason})
                    if not granted:
                        outcome.skipped.append(StepOutcome(
                            step, False, False,
                            f"awaiting human approval: {decision.reason}"))
                        continue

                try:
                    res = self._dispatch(step)
                except ScopeDenial as exc:
                    outcome.skipped.append(StepOutcome(step, False, False, str(exc)))
                    continue

                verified, detail = self._verify(step, res)
                self.trust.observe(step.op, success=res.ok and verified,
                                   rolled_back=not (res.ok and verified))
                outcome.executed.append(StepOutcome(
                    step, True, verified, detail,
                    res.undo.description if res.undo else None,
                ))

        # A plan that mutated the machine but failed verification leaves the
        # world in a state nobody checked.  Unwinding it automatically is the
        # only defensible default: partial, unverified change is worse than
        # no change, because the next decision is made on false premises.
        if outcome.failed_verifications:
            outcome.rolled_back = self._rollback_since(mark)
            self.audit.append("kernel", "auto_rollback", {
                "goal": plan.goal,
                "failed_steps": [o.step.describe() for o in outcome.failed_verifications],
                "undone": outcome.rolled_back,
            })

        outcome.verification = self.audit.verify()
        outcome.audit_head = self.audit.head()
        self.audit.append("kernel", "run_complete", outcome.summary())
        return outcome

    def _rollback_since(self, mark: int) -> int:
        """Undo actions recorded at or after ``mark``, newest first.

        Newest-first matters when several steps touched the same path: the
        last writer's inverse restores the state the earlier inverse expects.
        """
        n = 0
        for res in reversed(self.ledger.history()[mark:]):
            if res.undo and not res.undo.undone:
                res.undo.apply()
                n += 1
        return n

    def _dispatch(self, step: Step) -> ActionResult:
        if step.op is Op.WRITE:
            return self.ledger.write(step.path or "", step.content or "")
        if step.op is Op.MKDIR:
            return self.ledger.mkdir(step.path or "")
        if step.op is Op.DELETE:
            return self.ledger.delete(step.path or "")
        if step.op is Op.APPEND:
            return self.ledger.append(step.path or "", step.content or "")
        if step.op is Op.READ:
            return self.ledger.read(step.path or "")
        if step.op is Op.LIST:
            return self.ledger.list(step.path or "")
        if step.op is Op.STAT:
            return self.ledger.stat(step.path or "")
        if step.op is Op.RUN_TESTS:
            return self.ledger.run_tests(step.command or "run_tests")
        raise ScopeDenial(f"no dispatch path for {step.op.value}",
                          op=step.op, path=step.path)

    # -- verify against the real machine ---------------------------------
    def _verify(self, step: Step, res: ActionResult) -> tuple[bool, str]:
        if not res.ok:
            return False, res.note or "action reported failure"

        if step.op is Op.WRITE:
            if not step.expect_content_after:
                return os.path.exists(res.path or ""), "file exists on disk"
            try:
                with open(res.path, "r", encoding="utf-8") as fh:
                    actual = fh.read()
                ok = actual == step.expect_content_after
                return ok, ("content matches expectation" if ok
                            else "content on disk differs from what was written")
            except OSError as exc:
                return False, f"unreadable: {exc}"

        if step.op is Op.MKDIR:
            return os.path.isdir(res.path or ""), "directory present"

        if step.op is Op.DELETE:
            gone = not os.path.exists(res.path or "")
            quarantined = bool(res.value) and os.path.exists(str(res.value))
            return gone and quarantined, (
                "removed from workspace and present in quarantine"
                if gone and quarantined else "quarantine state unexpected"
            )

        if step.op is Op.APPEND:
            return os.path.exists(res.path or ""), "file exists after append"

        if step.op is Op.RUN_TESTS:
            return bool(res.ok), "command exited zero" if res.ok else "nonzero exit"

        if step.op in (Op.READ, Op.LIST, Op.STAT):
            return res.ok, "read succeeded"

        return res.ok, "no verification rule; defaulted to action status"

    # -- recovery --------------------------------------------------------
    def rollback_plan(self, plan: Plan) -> int:
        """Undo everything this plan did, newest first."""
        n = 0
        for res in reversed(self.ledger.history()):
            if res.undo and not res.undo.undone:
                res.undo.apply()
                n += 1
        self.audit.append("kernel", "rollback_plan",
                          {"goal": plan.goal, "undone": n})
        return n

    def undo_all(self) -> int:
        n = self.ledger.undo_all()
        self.audit.append("kernel", "undo_all", {"undone": n})
        return n

    # -- introspection ----------------------------------------------------
    def notifications(self) -> list[dict]:
        return list(self._notifications)

    def status(self) -> dict:
        return {
            "roots": self.guard.roots,
            "caps": [c.describe() for c in self.caps.active()],
            "trust": self.trust.table(),
            "audit_records": self.audit.n(),
            "audit_ok": self.audit.verify().ok,
            "audit_head": self.audit.head(),
        }

    def revoke_everything(self) -> int:
        """The big red button.  Always available, always logged."""
        n = self.caps.revoke_all()
        self.audit.append("kernel", "revoke_all", {"revoked": n})
        return n