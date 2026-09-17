"""Bridging the model to the machine.

This is the piece that makes the loop real: the ForgeLM (or any backend) emits
a *plan* as text, the text is parsed into a typed ``Plan``, and the kernel
decides whether the machine is allowed to carry it out.

The critical design rule: **the model's output is data, never instructions.**

Nothing the model writes is executed, evaluated, or interpreted as code.  The
parser matches the text against a fixed grammar of typed steps; anything that
does not fit is a parse failure, and a parse failure means nothing happens.
There is no path from model text to ``eval``, ``exec``, a shell, or an
unlisted operation -- the grammar simply has no way to express them.

The second rule: the model cannot lie its way to an effect.  A plan that
parses still has to survive capability checks, the trust ladder, precondition
checks, and post-hoc verification against the real filesystem.

Grammar (one step per line, JSON-ish, deliberately narrow)::

    goal: <free text>
    mkdir <path>
    write <path> <<< <content> >>>
    append <path> <<< <content> >>>
    delete <path>
    read <path>
    list <path>
    stat <path>
    run_tests <command-name>
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from forge.control.kernel import ControlKernel, Plan, RunOutcome, Step
from forge.control.scope import Op

VALID_OPS = {op.value for op in Op}

PATH_RE = r'[^\s<>"]+'
# DOTALL: the `rest` group must be able to span a multi-line content block.
STEP_RE = re.compile(
    rf'^(?P<op>{"|".join(sorted(VALID_OPS))})\s+(?P<rest>.+)$', re.DOTALL
)
BLOCK_RE = re.compile(r'^(?P<path>\S+)\s*<<<\s*(?P<content>.*?)\s*>>>$', re.DOTALL)


@dataclass
class ParseResult:
    plan: Optional[Plan]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.errors


def parse_plan(text: str, base_dir: str = "", max_steps: int = 0) -> ParseResult:
    """Turn model text into a typed Plan.  Strict: unknown syntax is an error.

    ``base_dir`` is prepended to relative paths so the model can work in
    relative terms without being able to name an absolute location itself.
    The kernel's PathGuard still applies afterwards.

    ``max_steps`` stops collection after that many steps.  Bounding here rather
    than trimming the resulting plan keeps the parsed plan and the text that
    produced it in agreement; otherwise a caller limbos the plan but the bridge
    re-parses the full text and runs everything anyway.
    """
    import os

    errors: list[str] = []
    warnings: list[str] = []
    goal = "unspecified goal"
    steps: list[Step] = []

    for raw_line in _statements(text):
        if max_steps and len(steps) >= max_steps:
            warnings.append(f"step limit {max_steps} reached; later steps ignored")
            break
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.lower().startswith("goal:"):
            goal = line.split(":", 1)[1].strip()
            continue

        match = STEP_RE.match(line)
        if not match:
            errors.append(f"unrecognised step syntax: {line[:80]!r}")
            continue

        op = Op(match.group("op"))
        rest = match.group("rest").strip()

        if op in (Op.WRITE, Op.APPEND):
            block = BLOCK_RE.match(rest)
            if not block:
                errors.append(f"{op.value} requires '<<< ... >>>' content block")
                continue
            path = block.group("path")
            content = block.group("content")
            if len(content) > 20_000:
                warnings.append("content truncated to 20000 characters")
                content = content[:20_000]
            steps.append(Step(
                op=op, path=_join(base_dir, path), content=content,
                expect_absent_before=(op is Op.WRITE),
                expect_content_after=(content if op is Op.WRITE else None),
            ))
            continue

        if op is Op.RUN_TESTS:
            # Defence in depth: the ledger would refuse a non-allowlisted
            # command anyway, but the grammar layer should never let one
            # through in the first place.  Two independent layers is the whole
            # point; letting the outer one be permissive wastes it.
            from forge.control.actions import ALLOWED_COMMANDS

            if rest not in ALLOWED_COMMANDS:
                errors.append(
                    f"run_tests command {rest[:40]!r} is not allowlisted"
                )
                continue
            steps.append(Step(op=op, command=rest))
            continue

        if op is Op.DELETE:
            steps.append(Step(
                op=op, path=_join(base_dir, rest),
                # A model deleting something must not be certain about it.
                confidence=0.7,
            ))
            continue

        # read / list / stat / scan / copy / move: single path argument
        if any(c in rest for c in "<>|;&$`"):
            errors.append(f"illegal characters in path for {op.value}: {rest[:60]!r}")
            continue
        steps.append(Step(op=op, path=_join(base_dir, rest)))

    if not steps and not errors:
        errors.append("no steps found in model output")

    if errors:
        return ParseResult(None, errors, warnings)
    return ParseResult(Plan(goal=goal, steps=steps), [], warnings)


def _statements(text: str):
    """Yield logical statements, joining multi-line <<<...>>> blocks.

    The scanner tracks whether it is inside a content block so that content
    containing newlines, blank lines, or lines that look like other steps is
    carried through verbatim instead of being re-parsed as grammar.
    """
    buffer: list[str] = []
    inside_block = False

    for line in text.splitlines():
        if not inside_block:
            if "<<<" in line and ">>>" not in line.split("<<<", 1)[1]:
                buffer.append(line)
                inside_block = True
                continue
            yield line
            continue

        buffer.append(line)
        if ">>>" in line:
            yield "\n".join(buffer)
            buffer = []
            inside_block = False

    if buffer:
        # An unterminated block is surfaced rather than silently dropped.
        yield "\n".join(buffer)


def _join(base_dir: str, path: str) -> str:
    import os

    if not base_dir:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


@dataclass
class BridgeOutcome:
    model_text: str
    parse: ParseResult
    run: Optional[RunOutcome] = None

    @property
    def acted(self) -> bool:
        return bool(self.run and self.run.executed)

    def summary(self) -> str:
        lines = [f"model proposed {len(self.parse.plan.steps) if self.parse.plan else 0} steps"]
        if self.parse.errors:
            lines.append(f"parse errors: {self.parse.errors}")
            lines.append("-> nothing executed (unparseable output is inert)")
            return "\n".join(lines)
        lines.append(f"goal: {self.parse.plan.goal}")
        for w in self.parse.warnings:
            lines.append(f"warning: {w}")
        if self.run is None:
            lines.append("-> parsed but not executed")
            return "\n".join(lines)
        for o in self.run.executed:
            lines.append(f"  [{'ok' if o.verified else 'FAIL'}] "
                         f"{o.step.describe()} :: {o.detail}")
        for o in self.run.skipped:
            lines.append(f"  [refused] {o.step.describe()} :: {o.detail}")
        lines.append(f"success={self.run.success} verified="
                     f"{sum(1 for o in self.run.executed if o.verified)}")
        return "\n".join(lines)


def model_drives_machine(
    model_text: str,
    kernel: ControlKernel,
    base_dir: str = "",
    execute: bool = True,
    max_steps: int = 0,
) -> BridgeOutcome:
    """Parse model output and, if it parses, offer it to the kernel."""
    parsed = parse_plan(model_text, base_dir=base_dir, max_steps=max_steps)
    if not parsed.ok:
        kernel.audit.append("bridge", "parse_failed",
                            {"errors": parsed.errors})
        return BridgeOutcome(model_text, parsed, None)

    if not execute:
        return BridgeOutcome(model_text, parsed, None)

    run = kernel.execute(parsed.plan)
    kernel.audit.append("bridge", "model_plan_run", {
        "goal": parsed.plan.goal,
        "executed": len(run.executed),
        "verified": sum(1 for o in run.executed if o.verified),
        "success": run.success,
    })
    return BridgeOutcome(model_text, parsed, run)