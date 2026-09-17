"""End-to-end pipeline demo: model -> debate -> judge -> machine -> audit.

Run:  python3 -m forge.pipeline_demo --steps 200

This is the demo the framework was missing.  Before it, there were three
islands that never spoke: a model that trained, agents that debated, and a
kernel that could operate a filesystem.  Here they run as one chain, on a
real temporary machine, and every stage is visible.

The honest headline appears in section 4: a 2.6M-parameter byte-level model
usually does not emit parseable plans, so the pipeline falls back -- and says
so, in the audit chain, rather than pretending the model did the work.
"""
from __future__ import annotations

import argparse
import os
import tempfile

from forge.agents.act import ActionChannel, default_fallback_plan
from forge.agents.backends import available_providers
from forge.agents.local import resolve_backend
from forge.agents.orchestrator import Orchestrator
from forge.agents.roles import Judge, build_team
from forge.config import ForgeConfig
from forge.control.kernel import ControlKernel
from forge.control.scope import Op
from forge.control.trust import Level
from forge.model.transformer import ForgeLM
from forge.training.trainer import Trainer
from forge.viz import table
from forge.viz.report import build_report

TASK = (
    "Harden a Python service that parses untrusted input: write down the "
    "trust-boundary rules and record how the agent verified its own work."
)


def section(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def indent(text: str, pad: str = "  ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


def main() -> None:
    ap = argparse.ArgumentParser(description="Forge end-to-end pipeline demo")
    ap.add_argument("--steps", type=int, default=200, help="LM training steps")
    ap.add_argument("--rounds", type=int, default=3, help="debate rounds")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--use-model", action="store_true",
                    help="ask the trained model for plans (usually fails to parse)")
    args = ap.parse_args()

    machine = os.path.realpath(tempfile.mkdtemp(prefix="forge_pipeline_"))
    cfg = ForgeConfig()
    cfg.train.steps = args.steps
    cfg.train.out_dir = os.path.join(args.out, "pipeline")

    # ---------------------------------------------------------------- 1
    section("1. TRAIN THE MODEL")
    model = ForgeLM(cfg.model)
    trainer = Trainer(cfg, model)
    logs = trainer.train_lm(on_step=lambda l: None)
    ckpt = os.path.join(cfg.train.out_dir, "model.pt")
    trainer.save(ckpt)
    first = sum(l.loss for l in logs[:10]) / 10
    last = sum(l.loss for l in logs[-10:]) / 10
    print(f"  parameters : {model.num_params():,}")
    print(f"  loss       : {first:.4f} -> {last:.4f}")
    print(f"  checkpoint : {ckpt}")

    # ---------------------------------------------------------------- 2
    section("2. WIRE THE SUB-AGENTS")
    providers = available_providers()
    if providers:
        print(f"  live provider keys found: {', '.join(providers)}")
    else:
        print("  no provider keys -> deterministic rule backends")
    backend = resolve_backend("local" if args.use_model else "rules",
                              checkpoint=ckpt)
    team = build_team()
    print(f"  sub-agents : {', '.join(sorted(team))}")
    print(f"  backend    : {backend.name}")

    # ---------------------------------------------------------------- 3
    section("3. THE MACHINE, AND ITS GUARDS")
    kernel = ControlKernel([machine], approve=lambda d, s: True,
                           audit_path=os.path.join(machine, "audit.log"))
    # Seeded by a human: a class at dry_run can never earn promotion itself.
    for op in (Op.WRITE, Op.MKDIR, Op.READ, Op.LIST, Op.STAT):
        kernel.trust.seed(op, Level.AUTO)
    print(f"  machine root   : {machine}")
    print(f"  capabilities   : {len(kernel.caps.active())} granted")
    print(f"  trust (seeded) : {', '.join(sorted(op.value for op in [Op.WRITE, Op.MKDIR, Op.READ]))}")
    print(f"  audit head     : {kernel.audit.head()[:24]}...")

    channel = ActionChannel(kernel, machine, backend=backend,
                            fallback_plan=default_fallback_plan)

    # ---------------------------------------------------------------- 4
    section("4. DEBATE -> JUDGE -> ACTION (the whole chain)")
    events: list[tuple[str, dict]] = []
    orch = Orchestrator(
        team, Judge(), max_rounds=args.rounds,
        on_event=lambda n, p: events.append((n, p)),
        channel=channel,
    )
    result = orch.run(TASK)

    rows = []
    for r in result.rounds:
        rows.append([r.index, len(r.proposals), len(r.critiques),
                     len(r.blocked),
                     r.ruling.verdict.value if r.ruling else "-"])
    print(indent(table(["round", "proposals", "critiques", "blocked",
                        "verdict"], rows)))
    print(f"\n  accepted  : {result.accepted}")
    print(f"  rationale : {result.final.rationale if result.final else 'n/a'}")

    if result.channel is not None:
        ch = result.channel
        print(f"\n  action channel:")
        print(f"    model parsed own output : {ch.model_parsed}")
        print(f"    used fallback           : {ch.used_fallback}")
        print(f"    steps executed          : {ch.steps_executed}")
        print(f"    steps verified          : {ch.verified}")
        print(f"    rolled back             : {ch.rolled_back}")
        if ch.used_fallback:
            print("\n    NOTE: the model's output did not parse as a plan, so the")
            print("    deterministic fallback ran instead. This is recorded in the")
            print("    audit chain as 'fallback' -- visible, not silently hidden.")
        if ch.run is not None:
            for o in ch.run.executed:
                mark = "ok" if o.verified else "FAIL"
                print(f"    [{mark}] {o.step.describe():<28} {o.detail}")
            for o in ch.run.skipped:
                print(f"    [refused] {o.step.describe():<24} {o.detail}")
    else:
        print("\n  no action taken (no accepted ruling)")

    # ---------------------------------------------------------------- 5
    section("5. WHAT ACTUALLY HAPPENED ON DISK")
    for root, dirs, files in os.walk(machine):
        dirs[:] = [d for d in dirs if d != ".forge_quarantine"]
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, machine)
            print(f"  {rel}  ({os.path.getsize(p)}B)")
    doc = os.path.join(machine, "docs", "harden-a-python-service-that-parses.md")
    if os.path.exists(doc):
        print(f"\n  contents of {os.path.relpath(doc, machine)}:")
        print(indent(open(doc).read().strip()))

    # ---------------------------------------------------------------- 6
    section("6. VERIFICATION AND AUDIT")
    run = result.channel.run if result.channel else None
    if run is not None:
        print(f"  verified against real disk : {run.verification.ok}")
        print(f"  audit records              : {run.verification.n_records}")
        print(f"  success                    : {run.success}")
    log = kernel.audit
    print(f"\n  audit chain intact : {log.verify().ok}")
    print(f"  chain head         : {log.head()[:24]}...")
    print("\n  events recorded in the chain:")
    for action in ("perceive", "simulate", "action:mkdir", "action:write",
                   "fallback", "plan_refused", "run_complete"):
        n = len(log.find(action))
        if n:
            print(f"    {action:<18} x{n}")

    # ---------------------------------------------------------------- 7
    section("7. FAILURE PATH: VERIFICATION FAILS -> AUTO-ROLLBACK")
    kernel.trust.seed(Op.WRITE, Level.AUTO)
    before = set()
    for root, _d, files in os.walk(machine):
        for f in files:
            before.add(os.path.relpath(os.path.join(root, f), machine))

    from forge.control.kernel import Plan, Step
    bad_target = os.path.join(machine, "docs", "residue.md")
    bad = kernel.plan("claim something untrue", [
        Step(Op.WRITE, bad_target, content="real bytes written",
             expect_content_after="a claim that will not match"),
    ])
    outcome = kernel.execute(bad)
    print(f"  failed verifications : {len(outcome.failed_verifications)}")
    print(f"  auto-rolled back     : {outcome.rolled_back}")
    print(f"  reported success     : {outcome.success}")
    print(f"  residue on disk      : {os.path.exists(bad_target)}")
    print(f"  audit recorded       : {bool(kernel.audit.find('auto_rollback'))}")
    print("\n  the machine is exactly as it was before the failed plan:")
    after = set()
    for root, _d, files in os.walk(machine):
        d = [x for x in _d if x != ".forge_quarantine"]
        for f in files:
            after.add(os.path.relpath(os.path.join(root, f), machine))
    added = after - before
    print(f"    new files: {sorted(added) if added else 'none'}")

    # ---------------------------------------------------------------- 8
    section("8. REPORT")
    report = os.path.join(args.out, "pipeline_report.html")
    build_report(
        report,
        [
            ("Training loss",
             __import__("forge.viz.report", fromlist=["loss_section"])
             .loss_section([l.loss for l in logs])),
            ("Debate rounds",
             f"<pre>{table(['round', 'proposals', 'critiques', 'blocked', 'verdict'], rows)}</pre>"),
            ("Machine actions", f"<pre>{indent(channel_result_text(result))}</pre>"),
            ("Trust ladder",
             f"<pre>{table(['op', 'level', 'ok', 'fail'], [[r['op'], r['level'], r['successes'], r['failures']] for r in kernel.trust.table()])}</pre>"),
        ],
        meta=result.summary(),
    )
    print(f"  {report}")
    print(f"\n{'=' * 76}")
    print(f"  audit records: {log.n()}   chain intact: {log.verify().ok}")
    print(f"{'=' * 76}")


def channel_result_text(result) -> str:
    if result.channel is None:
        return "no action taken"
    ch = result.channel
    lines = [f"model_parsed={ch.model_parsed} fallback={ch.used_fallback} "
             f"executed={ch.steps_executed} verified={ch.verified} "
             f"rolled_back={ch.rolled_back}"]
    if ch.run is not None:
        for o in ch.run.executed:
            lines.append(f"  [{'ok' if o.verified else 'FAIL'}] {o.step.describe()}")
        for o in ch.run.skipped:
            lines.append(f"  [refused] {o.step.describe()}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()