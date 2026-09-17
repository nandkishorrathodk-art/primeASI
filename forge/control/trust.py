"""Earned autonomy: permissions expand from a measured track record.

The common proposal is a fixed autonomy level chosen by a human.  That wastes
the calibration information the system generates every time it acts.  This
module instead makes autonomy a *measurement*:

    dry_run  ->  approval  ->  auto

An operation class climbs one rung only after ``promote_after`` consecutive
successes with zero rollbacks, and drops straight back to ``dry_run`` on a
failure.  Trust is asymmetric on purpose: it is slow to earn and fast to lose.

Two more properties that matter:

* **Per-class, not global.**  Writing a new file is not the same risk as
  deleting one.  Each operation earns its own level.
* **Uncertainty escalation.**  A self-reported confidence below
  ``confidence_floor`` forces the action down to at least ``approval`` no
  matter how good the track record is.  A confident system that is sometimes
  wrong is more dangerous than an uncertain one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from forge.control.scope import Op


class Level(int, Enum):
    DRY_RUN = 0      # plan only, never touch anything
    APPROVAL = 1     # execute, but a human must confirm first
    AUTO = 2         # execute unattended


@dataclass
class ClassRecord:
    op: Op
    level: Level = Level.DRY_RUN
    successes: int = 0
    failures: int = 0
    approvals: int = 0
    rejections: int = 0
    streak: int = 0

    @property
    def accuracy(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.0


@dataclass
class Decision:
    op: Op
    level: Level
    allow: bool
    needs_approval: bool
    reason: str


class TrustLadder:
    def __init__(
        self,
        promote_after: int = 3,
        confidence_floor: float = 0.6,
        demote_on_failure: bool = True,
        start_level: Level = Level.DRY_RUN,
    ) -> None:
        self.promote_after = promote_after
        self.confidence_floor = confidence_floor
        self.demote_on_failure = demote_on_failure
        self.start_level = start_level
        self._records: dict[Op, ClassRecord] = {}

    def record_for(self, op: Op) -> ClassRecord:
        if op not in self._records:
            self._records[op] = ClassRecord(op=op, level=self.start_level)
        return self._records[op]

    # -- the decision ----------------------------------------------------
    def decide(self, op: Op, confidence: float = 1.0) -> Decision:
        rec = self.record_for(op)
        level = rec.level

        # Uncertainty always wins over history.
        if confidence < self.confidence_floor and level is Level.AUTO:
            return Decision(
                op, Level.APPROVAL, allow=True, needs_approval=True,
                reason=f"confidence {confidence:.2f} below floor "
                       f"{self.confidence_floor:.2f}: escalating to approval",
            )

        if level is Level.DRY_RUN:
            return Decision(op, level, allow=False, needs_approval=False,
                            reason="dry_run: plan only, not yet earned execution")
        if level is Level.APPROVAL:
            return Decision(op, level, allow=True, needs_approval=True,
                            reason=f"approval rung ({rec.streak}/{self.promote_after} "
                                   "toward auto)")
        return Decision(op, level, allow=True, needs_approval=False,
                        reason=f"auto rung, accuracy {rec.accuracy:.0%} "
                               f"over {rec.successes + rec.failures} runs")

    # -- updating --------------------------------------------------------
    def observe(self, op: Op, success: bool, rolled_back: bool = False) -> Level:
        rec = self.record_for(op)
        if success and not rolled_back:
            rec.successes += 1
            rec.streak += 1
            if rec.level is not Level.AUTO and rec.streak >= self.promote_after:
                rec.level = Level(rec.level + 1)
                rec.streak = 0
        else:
            rec.failures += 1
            rec.streak = 0
            if self.demote_on_failure:
                rec.level = Level.DRY_RUN
        return rec.level

    def note_approval(self, op: Op, approved: bool) -> None:
        rec = self.record_for(op)
        if approved:
            rec.approvals += 1
        else:
            rec.rejections += 1
            rec.streak = 0

    def grant_level(self, op: Op, level: Level, reason: str = "operator override") -> None:
        """Seed a class's level directly.  Human-only, and deliberately so.

        This exists because of a bootstrap problem the ladder cannot solve on
        its own: a class at DRY_RUN never executes, so it never accumulates the
        successes needed to leave DRY_RUN.  Autonomy therefore has to be
        *seeded* by a human.  That is the correct place for the decision to
        live -- an agent should not be able to bootstrap its own permissions.
        """
        rec = self.record_for(op)
        rec.level = level
        rec.streak = 0

    def seed(self, op: Op, level: Level = Level.APPROVAL,
             reason: str = "operator seeds initial autonomy") -> None:
        """Alias for grant_level, named for its actual purpose."""
        self.grant_level(op, level, reason)

    # -- introspection ---------------------------------------------------
    def table(self) -> list[dict]:
        return [
            {
                "op": r.op.value, "level": r.level.name.lower(),
                "successes": r.successes, "failures": r.failures,
                "accuracy": round(r.accuracy, 2), "streak": r.streak,
            }
            for r in sorted(self._records.values(), key=lambda x: x.op.value)
        ]

    def describe(self) -> str:
        rows = self.table()
        if not rows:
            return "(no history yet: every class starts at dry_run)"
        return "\n".join(
            f"  {r['op']:<10} {r['level']:<9} acc={r['accuracy']:.2f} "
            f"ok={r['successes']} fail={r['failures']} streak={r['streak']}"
            for r in rows
        )

    def badge(self, op: Op) -> str:
        """Human-readable HTML badge for the report."""
        rec = self.record_for(op)
        colours = {Level.DRY_RUN: "#fbbf24", Level.APPROVAL: "#60a5fa",
                   Level.AUTO: "#4ade80"}
        return (f'<span style="color:{colours[rec.level]}">'
                f'{op.value}: {rec.level.name.lower()}</span>')