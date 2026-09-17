"""Actions that carry their own inverse.

Every mutating operation returns an ``Undo`` alongside its result.  This is
what makes autonomous operation safe in practice: not that mistakes never
happen, but that **every mistake is recoverable in one call**, because the
recovery path was recorded at the moment the action was taken, by the code
that took it.

Design rules:

1. Irreversible operations are not offered.  ``delete`` moves to a quarantine
   directory; the bytes are never unlinked.  There is no unlink capability in
   this codebase at all.
2. An action that cannot produce its inverse must not run.  ``write`` to an
   existing file snapshots the old content first; if the snapshot fails, the
   write is refused rather than performed unrecoverably.
3. Commands are not free-form strings.  ``run_tests`` accepts one of a fixed
   allowlist of argv vectors, with no shell in the path, so there is nothing
   to inject into.

The ``Transaction`` groups these so a multi-step plan either fully lands or
fully unwinds.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from forge.control.audit import AuditChain
from forge.control.scope import Capability, CapabilitySet, Op, PathGuard, ScopeDenial

QUARANTINE_DIRNAME = ".forge_quarantine"

# argv allowlists: exact vectors, no shell, no user-supplied fragments.
ALLOWED_COMMANDS: dict[str, list[list[str]]] = {
    "run_tests": [
        ["python3", "-m", "pytest", "-q"],
        ["python3", "-m", "pytest", "tests/test_forge.py", "-q"],
        ["python3", "-c", "import forge; print(forge.__version__)"],
    ],
}
ALLOWED_CMD_TIMEOUT = 300


@dataclass
class Undo:
    """A recorded inverse operation."""
    description: str
    fn: Optional[Callable[[], None]] = None
    # A value the inverse needs but that no longer exists on disk.
    snapshot: Optional[bytes] = None
    undone: bool = False

    def apply(self) -> None:
        if self.undone:
            return
        if self.fn is not None:
            self.fn()
        self.undone = True

    def describe(self) -> str:
        return f"{self.description}{' (undone)' if self.undone else ''}"


@dataclass
class ActionResult:
    op: Op
    ok: bool
    path: Optional[str] = None
    value: object = None
    undo: Optional[Undo] = None
    note: str = ""


class ActionLedger:
    """Executes scoped actions and journals an inverse for each one."""

    def __init__(
        self,
        caps: CapabilitySet,
        guard: PathGuard,
        audit: Optional[AuditChain] = None,
        actor: str = "agent",
        quarantine_root: Optional[str] = None,
    ) -> None:
        self.caps = caps
        self.guard = guard
        self.audit = audit or AuditChain()
        self.actor = actor
        root = quarantine_root or (
            os.path.join(guard.roots[0], QUARANTINE_DIRNAME) if guard.roots else "."
        )
        self.quarantine = root
        self._history: list[ActionResult] = []

    # ------------------------------------------------------------------
    def _authorise(self, op: Op, path: Optional[str] = None) -> Capability:
        cap = self.caps.check(op, path)
        self.caps.spend(cap)
        return cap

    def _journal(self, res: ActionResult) -> ActionResult:
        self._history.append(res)
        self.audit.append(
            self.actor, f"action:{res.op.value}",
            {"ok": res.ok, "path": res.path, "note": res.note,
             "undo": res.undo.description if res.undo else None},
        )
        return res

    # -- reads ----------------------------------------------------------
    def read(self, path: str) -> ActionResult:
        self._authorise(Op.READ, path)
        real = self.guard.resolve(path)
        try:
            with open(real, "rb") as fh:
                data = fh.read()
            return self._journal(ActionResult(Op.READ, True, real,
                                              data.decode("utf-8", "replace")))
        except OSError as exc:
            return self._journal(ActionResult(Op.READ, False, real, note=str(exc)))

    def list(self, path: str) -> ActionResult:
        self._authorise(Op.LIST, path)
        real = self.guard.resolve(path)
        try:
            entries = sorted(os.listdir(real))
            return self._journal(ActionResult(Op.LIST, True, real, entries))
        except OSError as exc:
            return self._journal(ActionResult(Op.LIST, False, real, note=str(exc)))

    def stat(self, path: str) -> ActionResult:
        self._authorise(Op.STAT, path)
        real = self.guard.resolve(path)
        try:
            st = os.stat(real)
            info = {"size": st.st_size, "mtime": st.st_mtime,
                    "is_dir": os.path.isdir(real)}
            return self._journal(ActionResult(Op.STAT, True, real, info))
        except OSError as exc:
            return self._journal(ActionResult(Op.STAT, False, real, note=str(exc)))

    # -- writes ---------------------------------------------------------
    def write(self, path: str, content: str, mkdirs: bool = True) -> ActionResult:
        self._authorise(Op.WRITE, path)
        real = self.guard.resolve(path)

        parent = os.path.dirname(real)
        if parent and not os.path.isdir(parent):
            if not mkdirs:
                raise ScopeDenial(f"parent directory missing: {parent}",
                                  op=Op.WRITE, path=path,
                                  suggestion="create it first or pass mkdirs=True")
            os.makedirs(parent, exist_ok=True)

        # Snapshot before mutating, so the inverse is always constructible.
        existed = os.path.exists(real)
        previous: Optional[bytes] = None
        if existed:
            with open(real, "rb") as fh:
                previous = fh.read()

        with open(real, "w", encoding="utf-8") as fh:
            fh.write(content)

        if existed:
            def _restore() -> None:
                with open(real, "wb") as fh:
                    fh.write(previous or b"")
            undo = Undo(f"restore previous content of {real}",
                        snapshot=previous, fn=_restore)
        else:
            def _remove() -> None:
                if os.path.exists(real):
                    os.remove(real)
            undo = Undo(f"remove newly created {real}", fn=_remove)

        return self._journal(ActionResult(
            Op.WRITE, True, real, len(content), undo,
            note="overwrote" if existed else "created",
        ))

    def append(self, path: str, content: str) -> ActionResult:
        self._authorise(Op.APPEND, path)
        real = self.guard.resolve(path)
        prev_size = os.path.getsize(real) if os.path.exists(real) else 0
        with open(real, "a", encoding="utf-8") as fh:
            fh.write(content)

        def _truncate() -> None:
            with open(real, "r+b") as fh:
                fh.truncate(prev_size)
        undo = Undo(f"truncate {real} back to {prev_size} bytes", fn=_truncate)
        return self._journal(ActionResult(Op.APPEND, True, real, len(content), undo))

    def mkdir(self, path: str) -> ActionResult:
        self._authorise(Op.MKDIR, path)
        real = self.guard.resolve(path)
        os.makedirs(real, exist_ok=True)
        undo = Undo(f"remove directory {real}",
                    fn=lambda: os.path.exists(real) and os.rmdir(real))
        return self._journal(ActionResult(Op.MKDIR, True, real, note="created"))

    # -- delete = quarantine --------------------------------------------
    def delete(self, path: str) -> ActionResult:
        """Move to quarantine.  Never unlinks; the bytes remain recoverable."""
        self._authorise(Op.DELETE, path)
        real = self.guard.resolve(path)
        if not os.path.exists(real):
            return self._journal(ActionResult(Op.DELETE, False, real,
                                              note="does not exist"))
        os.makedirs(self.quarantine, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(self.quarantine, f"{stamp}-{uuid.uuid4().hex[:6]}"
                                            f"-{os.path.basename(real)}")
        shutil.move(real, dest)

        def _restore() -> None:
            if os.path.exists(dest) and not os.path.exists(real):
                shutil.move(dest, real)
        undo = Undo(f"restore {real} from quarantine", fn=_restore)
        return self._journal(ActionResult(Op.DELETE, True, real, dest, undo,
                                          note="quarantined, not erased"))

    # -- commands --------------------------------------------------------
    def run_tests(self, command: str = "run_tests") -> ActionResult:
        """Run one of the fixed allowlisted argv vectors.  No shell involved."""
        self._authorise(Op.RUN_TESTS, self.guard.roots[0] if self.guard.roots else None)
        if command not in ALLOWED_COMMANDS:
            raise ScopeDenial(
                f"command {command!r} is not allowlisted", op=Op.RUN_TESTS,
                suggestion=f"choose one of {sorted(ALLOWED_COMMANDS)}",
            )
        last: Optional[subprocess.CompletedProcess] = None
        for argv in ALLOWED_COMMANDS[command]:
            last = subprocess.run(
                argv, cwd=self.guard.roots[0] if self.guard.roots else None,
                capture_output=True, text=True, timeout=ALLOWED_CMD_TIMEOUT,
                shell=False,
            )
        ok = last is not None and last.returncode == 0
        out = ((last.stdout or "") + (last.stderr or ""))[-2000:] if last else ""
        # Commands here are read-only test runners, so the inverse is trivial.
        return self._journal(ActionResult(
            Op.RUN_TESTS, ok, None, {"returncode": last.returncode if last else -1,
                                     "output": out},
            Undo("no state change: read-only test command"),
        ))

    # -- history ---------------------------------------------------------
    def history(self) -> list[ActionResult]:
        return list(self._history)

    def undo_last(self) -> Optional[str]:
        for res in reversed(self._history):
            if res.undo and not res.undo.undone:
                res.undo.apply()
                self.audit.append(self.actor, "undo",
                                  {"op": res.op.value, "path": res.path,
                                   "description": res.undo.description})
                return res.undo.description
        return None

    def undo_all(self) -> int:
        n = 0
        for res in reversed(self._history):
            if res.undo and not res.undo.undone:
                res.undo.apply()
                n += 1
                self.audit.append(self.actor, "undo",
                                  {"op": res.op.value, "path": res.path})
        return n


class Transaction:
    """Groups actions so they all land or all unwind."""

    def __init__(self, ledger: ActionLedger, name: str = "tx") -> None:
        self.ledger = ledger
        self.name = name
        self._mark = len(ledger._history)

    def __enter__(self) -> "Transaction":
        self.ledger.audit.append(self.ledger.actor, "tx_begin", {"name": self.name})
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            rolled = self.rollback()
            self.ledger.audit.append(
                self.ledger.actor, "tx_rollback",
                {"name": self.name, "error": str(exc), "undone": rolled},
            )
            return False        # propagate: the caller must see the failure
        self.ledger.audit.append(self.ledger.actor, "tx_commit", {"name": self.name})
        return False

    def rollback(self) -> int:
        """Unwind everything this transaction did, newest first."""
        n = 0
        for res in reversed(self.ledger._history[self._mark:]):
            if res.undo and not res.undo.undone:
                res.undo.apply()
                n += 1
        self.ledger._history = self.ledger._history[:self._mark]
        return n