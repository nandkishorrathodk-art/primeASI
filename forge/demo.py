"""Forge end-to-end demo.

Run:  python -m forge.demo --steps 150
Produces: a trained checkpoint, a text sample, an HTML report, and one
orchestrated multi-agent debate over a security/coding task.
"""
from __future__ import annotations

import argparse
import os
import time

from forge.agents.backends import available_providers
from forge.agents.orchestrator import Orchestrator
from forge.agents.roles import Judge, build_team
from forge.config import ForgeConfig
from forge.data import make_shapes
from forge.model.transformer import ForgeLM
from forge.training.trainer import Trainer
from forge.viz import render_attention, render_image, sparkline, table
from forge.viz.report import build_report, expert_section, loss_section

TASK = (
    "Design a defensive security review pipeline for a Python service that "
    "parses untrusted input, and specify how an agentic loop should verify the "
    "result before trusting it."
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forge end-to-end demo")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--vision-steps", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = ForgeConfig()
    cfg.train.steps = args.steps
    cfg.train.out_dir = os.path.join(args.out, "base")
    cfg.model.vision.image_size = 32

    section("1. MODEL")
    model = ForgeLM(cfg.model)
    print(f"parameters      : {model.num_params():,}")
    print(f"layers / heads  : {cfg.model.n_layers} / {cfg.model.n_heads}")
    print(f"experts (MoE)   : {cfg.model.moe.num_experts}, top_k={cfg.model.moe.top_k}")
    print(f"vision          : {'patch encoder @ ' + str(cfg.model.vision.image_size) + 'px' if cfg.model.vision.enabled else 'off'}")
    print(f"device          : {cfg.train.device}")

    section("2. LANGUAGE TRAINING")
    trainer = Trainer(cfg, model)
    t0 = time.time()
    logs = trainer.train_lm(on_step=lambda l: None)
    elapsed = time.time() - t0
    for l in logs[:: max(1, args.steps // 8)]:
        print(f"  step {l.step:4d}  loss {l.loss:.4f}  ce {l.ce:.4f}  moe {l.moe:.4f}")
    first = sum(l.loss for l in logs[:10]) / 10
    last = sum(l.loss for l in logs[-10:]) / 10
    print(f"\n  mean loss first 10 steps : {first:.4f}")
    print(f"  mean loss last  10 steps : {last:.4f}")
    print(f"  wall clock               : {elapsed:.1f}s  ({model.num_params() / elapsed:,.0f} params/s)")

    section("3. MoE ROUTING")
    usage = logs[-1].expert_usage
    if usage:
        print(render_attention(usage, [f"expert {i}" for i in range(len(usage))]))
        spread = max(usage) / (min(usage) + 1e-9)
        print(f"\n  max/min usage ratio: {spread:.2f}  (near 1.0 = balanced routing)")

    section("4. VISION TOWER")
    imgs, labels = make_shapes(2)
    print("sample input (class {}):".format(labels[0].item()))
    print(render_image(imgs[0], width=32))
    acc_log: list[float] = []
    trainer.train_vision_classifier(
        steps=args.vision_steps,
        on_step=lambda l, a: acc_log.append(a),
    )
    if acc_log:
        print(f"\n  accuracy curve : {sparkline(acc_log, 50)}")
        print(f"  first 10 steps : {sum(acc_log[:10]) / 10:.3f}")
        print(f"  last  10 steps : {sum(acc_log[-10:]) / 10:.3f}")

    section("5. TEXT GENERATION")
    from forge.tokenizer import TOKENIZER
    import torch

    prompts = ["def add(a, b):", '{"agent": "security", "action":']
    model.eval()
    for p in prompts:
        ids = torch.tensor([TOKENIZER.encode(p, add_bos=True)])
        out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=30)
        text = TOKENIZER.decode(out[0].tolist())
        print(f"\n  prompt : {p!r}")
        print(f"  output : {text[:220]!r}")

    ckpt = os.path.join(cfg.train.out_dir, "model.pt")
    trainer.save(ckpt)
    print(f"\n  checkpoint -> {ckpt}")

    section("6. MULTI-AGENT DEBATE")
    providers = available_providers()
    if providers:
        print(f"  live teacher/agent backends detected: {', '.join(providers)}")
    else:
        print("  no provider API keys found -> deterministic rule backends")
        print("  (set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI / XAI / DEEPSEEK /")
        print("   NVIDIA_API_KEY to route individual sub-agents to those models)")
    team = build_team()
    orch = Orchestrator(team, Judge(), max_rounds=args.rounds)
    result = orch.run(TASK)

    rows = []
    for r in result.rounds:
        rows.append([
            r.index,
            len(r.proposals),
            len(r.critiques),
            len(r.blocked),
            r.ruling.verdict.value if r.ruling else "-",
        ])
    print("\n" + table(["round", "proposals", "critiques", "blocked", "verdict"], rows))
    print(f"\n  accepted : {result.accepted}")
    print(f"  rationale: {result.final.rationale if result.final else 'n/a'}")
    if result.final and result.final.blocking_issues:
        print("  blocking issues:")
        for b in result.final.blocking_issues:
            print(f"    - {b}")

    prov = orch.board.provenance("task")
    print(f"\n  blackboard keys : {', '.join(orch.board.keys())}")
    print(f"  provenance(task): {prov[0]['author']} rev{prov[0]['revision']}")

    section("7. REPORT")
    report = os.path.join(args.out, "report.html")
    build_report(
        report,
        [
            ("Language model loss", loss_section([l.loss for l in logs])),
            ("MoE expert utilisation", expert_section(
                usage, [f"e{i}" for i in range(len(usage))])),
            ("Vision accuracy", loss_section(acc_log)),
            ("Debate outcome", f"<pre>{table(['round','proposals','critiques','blocked','verdict'], rows)}</pre>"),
        ],
        meta={"params": model.num_params(), "steps": args.steps,
              "loss_first": round(first, 4), "loss_last": round(last, 4),
              "accepted": result.accepted},
    )
    print(f"  {report}")


if __name__ == "__main__":
    main()