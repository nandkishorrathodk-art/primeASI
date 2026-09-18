"""Capability scoping: what an agent is physically allowed to do.

The central claim of this module is that *safety is a capability problem, not
a filtering problem*.  A model that is merely told "do not delete files" can
be persuaded.  A model holding no delete capability simply cannot issue one:
there is nothing to persuade.

Three mechanisms:

1. **Action space** -- a fixed enumeration of typed operations.  Args are
   validated into these types; nothing else can be expressed.
2. **Capability tokens** -- a grant from a human for a specific operation on a
   specific scope, with an expiry and a call budget.  Absence is denial.
3. **PathGuard** -- filesystem containment by resolved real path, which
   rejects ``..`` traversal and symlink escapes at the kernel of the design
   rather than at the edge.

Denial returns a ``ScopeDenial`` carrying a constructive alternative, so an
agent that oversteps gets redirected rather than merely stopped.  A hard wall
teaches nothing; a wall with a door in it teaches the boundary.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

# ---------------------------------------------------------------- action space


class Op(str, Enum):
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    LIST = "list"
    STAT = "stat"
    SCAN = "scan"            # static analysis
    COPY = "copy"
    MOVE = "move"
    MKDIR = "mkdir"
    DELETE = "delete"        # quarantines; never unlinks
    RUN_TESTS = "run_tests"  # hardcoded allowlist of commands only


# Which ops need an explicit grant before they can be called at all.
MUTATING_OPS = {
    Op.WRITE, Op.APPEND, Op.COPY, Op.MOVE, Op.MKDIR, Op.DELETE, Op.RUN_TESTS,
}

# Ops that observe the machine without changing it.  They still require a
# grant -- absence is denial -- but they are not "unknown operations".
OBSERVING_OPS = {Op.READ, Op.LIST, Op.STAT, Op.SCAN}

# Every operation that can be granted.  `check` accepted only MUTATING_OPS and
# READ, which made `list`, `stat`, and `scan` permanently unreachable even
# though `default_policy` grants all three and `_dispatch` implements all
# three.  Keeping the accepted set explicit and derived from the enum is what
# stops a new op from being added in two places and forgotten in a third.
GRANTABLE_OPS = MUTATING_OPS | OBSERVING_OPS


class ScopeDenial(Exception):
    """Raised when an action is outside the granted capability set.

    Carries a ``suggestion``: the constructive move that would be permitted
    instead of the one that was refused.
    """

    def __init__(self, reason: str, op: Optional[Op] = None,
                 path: Optional[str] = None, suggestion: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.op = op
        self.path = path
        self.suggestion = suggestion

    def __str__(self) -> str:
        base = f"DENIED: {self.reason}"
        if self.suggestion:
            base += f" | try instead: {self.suggestion}"
        return base


# ---------------------------------------------------------------- path guard


class PathGuard:
    """Filesystem containment that survives traversal and symlink attacks."""

    def __init__(self, roots: Iterable[str]) -> None:
        self.roots = [os.path.realpath(r) for r in roots]

    def resolve(self, path: str) -> str:
        """Return the real path, or deny."""
        if "\x00" in path:
            raise ScopeDenial("null byte in path", suggestion="use a plain path")
        try:
            real = os.path.realpath(path)
        except (OSError, ValueError) as exc:
            raise ScopeDenial(f"unresolvable path: {exc}") from exc

        for root in self.roots:
            if real == root or real.startswith(root + os.sep):
                return real
        raise ScopeDenial(
            f"path escapes the sandbox: {real}",
            path=path,
            suggestion=f"use a path under one of {self.roots}",
        )

    def contains(self, path: str) -> bool:
        try:
            self.resolve(path)
            return True
        except ScopeDenial:
            return False


# ---------------------------------------------------------------- capabilities


@dataclass
class Capability:
    """A human grant: this op, on these paths, until this time, this often."""

    op: Op
    paths: list[str]                      # real paths or prefixes
    granted_by: str
    expires_at: float
    budget: int = 100                     # max calls; -1 means unlimited
    used: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    note: str = ""

    def covers(self, path: str) -> bool:
        real = os.path.realpath(path)
        return any(real == p or real.startswith(p + os.sep)
                   for p in (os.path.realpath(x) for x in self.paths))

    def live(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        if now > self.expires_at:
            return False
        return self.budget < 0 or self.used < self.budget

    def spend(self) -> None:
        self.used += 1

    def describe(self) -> str:
        ttl = max(0, int(self.expires_at - time.time()))
        budget = "unlimited" if self.budget < 0 else f"{self.used}/{self.budget}"
        return f"{self.id} {self.op.value} on {self.paths} ttl={ttl}s used={budget}"


class CapabilitySet:
    """The set of grants.  Deny by default: an empty set permits nothing."""

    def __init__(self, guard: PathGuard) -> None:
        self.guard = guard
        self._caps: list[Capability] = []
        self._audit: list[dict] = []

    # -- granting -------------------------------------------------------
    def grant(
        self,
        op: Op,
        paths: Iterable[str],
        granted_by: str,
        ttl_seconds: float = 300,
        budget: int = 100,
        note: str = "",
    ) -> Capability:
        resolved = [self.guard.resolve(p) for p in paths]
        cap = Capability(
            op=op,
            paths=resolved,
            granted_by=granted_by,
            expires_at=time.time() + ttl_seconds,
            budget=budget,
            note=note,
        )
        self._caps.append(cap)
        self._audit.append({"event": "grant", "id": cap.id, "op": op.value,
                            "paths": resolved, "by": granted_by,
                            "ttl": ttl_seconds, "budget": budget})
        return cap

    def revoke(self, cap_id: str) -> bool:
        before = len(self._caps)
        self._caps = [c for c in self._caps if c.id != cap_id]
        removed = len(self._caps) < before
        self._audit.append({"event": "revoke", "id": cap_id, "ok": removed})
        return removed

    def revoke_all(self) -> int:
        n = len(self._caps)
        for c in list(self._caps):
            self._audit.append({"event": "revoke", "id": c.id, "ok": True})
        self._caps.clear()
        return n

    # -- checking -------------------------------------------------------
    def check(self, op: Op, path: Optional[str] = None) -> Capability:
        """Raise ScopeDenial unless a live capability covers this call.

        Only genuinely unknown operations are rejected here.  `list`, `stat`,
        and `scan` used to fall into this branch and were unreachable no
        matter what was granted, which silently contradicted both
        ``default_policy`` (which grants them) and ``_dispatch`` (which
        implements them).  A capability check that refuses an operation the
        policy grants is not a security control; it is a bug that hides one.
        """
        if op not in GRANTABLE_OPS:
            raise ScopeDenial(f"unknown operation {op!r}")

        resolved = self.guard.resolve(path) if path else None

        for cap in self._caps:
            if cap.op is not op or not cap.live():
                continue
            if resolved is None or cap.covers(resolved):
                return cap

        # Build a helpful, constructive denial.
        expired = [c for c in self._caps if c.op is op and not c.live()]
        if expired:
            raise ScopeDenial(
                f"capability for {op.value} expired or exhausted",
                op=op, path=path,
                suggestion="ask the operator to re-grant with a fresh TTL",
            )
        if op in (Op.WRITE, Op.MKDIR, Op.APPEND):
            raise ScopeDenial(
                f"no write capability for {path}",
                op=op, path=path,
                suggestion="write into the workspace root, or request a grant "
                           f"scoped to {resolved}",
            )
        raise ScopeDenial(
            f"no capability for {op.value}" + (f" on {path}" if path else ""),
            op=op, path=path,
            suggestion="request a capability from the operator",
        )

    def spend(self, cap: Capability) -> None:
        cap.spend()
        self._audit.append({"event": "spend", "id": cap.id, "op": cap.op.value,
                            "used": cap.used})

    # -- introspection ---------------------------------------------------
    def active(self) -> list[Capability]:
        return [c for c in self._caps if c.live()]

    def audit_log(self) -> list[dict]:
        return list(self._audit)

    def describe(self) -> str:
        live = self.active()
        if not live:
            return "(no active capabilities)"
        return "\n".join(f"  {c.describe()}" for c in live)


def default_policy(guard: PathGuard, operator: str = "human") -> CapabilitySet:
    """A sane starting grant set: read everywhere in-sandbox, write only new
    files for a short window, delete only via quarantine."""
    caps = CapabilitySet(guard)
    electron = next(iter(guard.roots), None)
    if electron:
        caps.grant(Op.READ, [electron], operator, ttl_seconds=3600, budget=-1,
                   note="reading is safe and reversible")
        caps.grant(Op.LIST, [electron], operator, ttl_seconds=3600, budget=-1)
        caps.grant(Op.STAT, [electron], operator, ttl_seconds=3600, budget=-1)
        caps.grant(Op.SCAN, [electron], operator, ttl_seconds=3600, budget=-1)
        caps.grant(Op.WRITE, [electron], operator, ttl_seconds=600, budget=50,
                   note="time-boxed writes")
        caps.grant(Op.APPEND, [electron], operator, ttl_seconds=600, budget=50,
                   note="append is byte-reversible")
        caps.grant(Op.MKDIR, [electron], operator, ttl_seconds=600, budget=20)
        caps.grant(Op.DELETE, [electron], operator, ttl_seconds=600, budget=10,
                   note="quarantine only")
        caps.grant(Op.RUN_TESTS, [electron], operator, ttl_seconds=600, budget=10)
    return caps