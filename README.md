# Forge

A small agentic LLM framework you can actually train on a CPU.

Everything in this repo runs end-to-end on 4 cores and 15 GB of RAM with no
GPU. That constraint is a feature: it forces every component to be real,
measurable, and inspectable rather than a thin wrapper around a datacentre.

## What is actually in here

| Component | Status | Notes |
|---|---|---|
| Byte tokenizer | works | dependency-free, VOCAB 260 |
| MoE transformer LLM | trains | shared + routed experts, load balancing |
| Vision tower | trains | patch encoder, reaches ~95% on synthetic task |
| Multi-agent debate loop | works | 5 sub-agents + judge |
| Security analyzer | works | static AST + regex, never executes code |
| Lab-scoped probe | works | private ranges + explicit allowlist only |
| Self-distillation (EMA) | works | no external dependency |
| API-teacher distillation | opt-in | only if you hold the keys |
| Reports | works | text + self-contained HTML |

## Quick start

```bash
python3 -m pip install numpy torch
python3 -m pytest tests/ -q          # 35 tests, ~4s
python3 -m forge.demo --steps 250    # trains, generates, debates, reports
```

The demo prints a trained-model sample, MoE routing usage, the vision accuracy
curve, a full agent debate transcript, and writes `runs/report.html`.

## What this is NOT

Read this section carefully, because most "build your own LLM" material
conveniently skips it.

**You cannot train a frontier model here.** GPT-4 class training costs hundreds
of millions of dollars and needs tens of thousands of GPUs. A 2M-parameter
model (what this repo trains) is roughly one-millionth of that. Anyone
promising otherwise is selling something.

**You cannot "train GPT, Claude, Gemini, Grok and DeepSeek into your model."**
Their weights are not public, they are not licensed for that, and even given
the weights, distilling them requires datacentre compute. What you *can* do:

- Distil *soft labels* from any model you have API access to. That is real
  training signal. It is why `forge/training/distill.py` exists.
- Learn from your own data with a model small enough to train locally.
- Route sub-agents to different frontier models you have keys for, so their
  *opinions* (not weights) inform one debate. See `forge/agents/backends.py`.

The framework detects which provider keys exist and uses exactly those. It
never fabricates a provider's opinion.

**Some acronyms are not what they sound like.** Honest map:

| Term | Reality |
|---|---|
| MoE | Implemented here (routing, shared experts, balancing loss) |
| SLM / MLM | Small LM / Masked LM — config variants, not a separate architecture |
| VLM | Here: an image encoder injecting summary tokens into the LM token stream |
| LCM / MAM | Not standard published architectures. If you invent one, define it; this repo will not pretend to implement a paper that does not exist |
| SAM | Segment Anything (Meta) — promptable *segmentation*, a vision task, not a language recipe |

## Architecture

```
text ──▶ byte tokenizer ──▶ ┌─────────────────────────┐
                            │  ForgeLM                │
images ──▶ patch encoder ──▶│  ├ attn (causal, KV$)   │──▶ logits
                            │  └ MoE (shared+routed)  │
                            └─────────────────────────┘
```

Each block: `x + attn(norm(x))`, then `x + moe(norm(x))`.

MoE layer:

```
                ┌── shared expert(s) ── every token ──────────┐
x ── router ──▶ │                                             ├── + ──▶ out
                └── top_k of N routed experts (token choice) ─┘
```

Key details, all of which have tests:

- **Shared experts** absorb generic patterns so routed experts can specialise.
- **Router jitter** during training prevents early expert collapse.
- **Load-balancing loss** uses dispatched *token counts*, not router
  probabilities — a confident router can still be perfectly balanced.
- **Expert-choice routing** is available and perfectly balanced but **non-causal**
  (it inspects the whole batch), so it is training-only. This is enforced by a
  test that guards the documented invariant.

## The agentic layer

Five sub-agents: `security`, `coding`, `hacking`, `vision`, `architecture`.

```
task ─▶ parallel proposals ─▶ cross-critique ─▶ HARD SECURITY GATE ─▶ judge
              ▲                                                        │
              └──────────────── revise ────────────────────────────────┘
```

Three things make this more than "a bunch of prompts":

1. **Forced disagreement.** Agents critique each other's proposals, not their
   own. Five agents that agree produce nothing.
2. **Evidence requirement.** The judge cannot accept a proposal with no
   evidence — that rule is enforced in code *after* the model answers, so a
   talkative judge cannot override it.
3. **A gate the model cannot talk its way past.** `check_policy()` is a
   deterministic filter outside the model. Where a neural judge can be
   persuaded, a regex gate cannot.

### Blackboard provenance

Every claim is attributed, versioned, and evidence-linked:

```python
board.write("design", "v2", author="architecture-agent",
            evidence=[Evidence("test", "pytest::test_moe_balance")])
board.provenance("design")
# [{'revision': 1, 'author': 'coding-agent', ...},
#  {'revision': 2, 'author': 'architecture-agent', ...}]
board.unsupported_claims()   # anything asserted without evidence
```

## Security posture

This framework has offensive-security roles, so the guardrails are explicit and
tested rather than aspirational.

**Refused outright** (`forge/agents/roles.py`, tested): malware construction,
targeted exploits against live hosts, attacks on third parties, credential
theft, scanning public infrastructure.

**Supported** (`forge/security/`): threat modelling, static analysis of code
you own, fuzzing, hardening, detection rules, and low-rate reachability probes
against private/lab ranges. `check_scope()` blocks public hosts unless the user
explicitly allowlists them, which turns authorised testing into a deliberate
act. Rate limited to 30 requests/minute.

**No code execution.** `forge/security/analyzer.py` reads and pattern-matches
via `ast`; it never runs submitted code. An agentic coding loop that executes
model-generated code is a remote-code-execution hole by design, so this repo
does not do it.

## Distillation, honestly

`EMATeacher` keeps a slow-moving copy of the student and distils it back. It
needs nothing external and measurably smooths the loss. That is real
self-distillation.

To use real teachers:

```bash
export OPENAI_API_KEY=...      # whichever you actually have
export ANTHROPIC_API_KEY=...
python3 -c "
from forge.training.distill import collect_teacher_texts
print(collect_teacher_texts(['def parse(x):']).keys())
"
```

Missing keys are skipped, never simulated.

## Layout

```
forge/
  config.py           all hyperparameters
  tokenizer.py        byte tokenizer
  data.py             synthetic corpus + shapes dataset
  model/
    moe.py            shared/routed experts, two routing modes, balance loss
    vision.py         patch encoder -> summary tokens
    transformer.py    causal attention with KV cache, generate()
  agents/
    protocol.py       typed messages, evidence
    blackboard.py     provenance-tracked shared state
    backends.py       rules / local / remote provider backends
    roles.py          the 5 sub-agents, policy gate, judge
    orchestrator.py   debate rounds, hard security gate
  security/
    analyzer.py       static analysis (no execution)
    sandbox.py        scope-enforced lab probing
  training/
    trainer.py        LM + vision loops, cosine LR, checkpoints
    distill.py        EMA self-distillation, optional API teachers
  viz/                text renderers + standalone HTML report
```

## Extending it

Add a sub-agent: append a spec to `build_team()` in `forge/agents/roles.py`.

Add a provider: add an entry to `PROVIDERS` in `forge/agents/backends.py`.

Swap the tokenizer: implement `encode`/`decode`/`vocab_size` and pass it in.

Add a routing policy: implement `_your_policy` in `MoELayer.forward` and
register the name in the assertion. If it is non-causal, say so in the
docstring and the tests — the distinction matters for generation.

## Numbers from a real run

300 steps, CPU only (4 cores). This is one measured run, not a cherry-pick;
rerun `python3 -m forge.demo` to reproduce.

```
parameters        : 2,591,232
wall clock        : 56.6s  (45,758 params/s)
loss first 10     : 5.3663
loss last  10     : 1.5087
expert usage      : 0.504 / 0.412 / 0.674 / 0.410
vision accuracy   : 0.281 -> 0.994
```

Routing balance varies run to run (1.2 to 1.9 max/min ratio). That variance is
the honest number: at 2.6M parameters and 300 steps the router has not fully
settled, and the balancing loss holds it in a workable band rather than
pinning it to uniformity evenly. A production MoE would train orders of
magnitude longer to get tight balance.

The loss genuinely drops and the vision tower genuinely learns; both are
asserted in the test suite, not merely printed.

## Test suite

35 tests, ~3.4s, no mocks - everything exercises real code paths:

- tokenizer round-trip including multibyte UTF-8
- MoE: shared experts actually contribute, expert-choice routing is balanced,
  token-choice load sums to `top_k`, balance loss grows under collapse,
  no expert starves during real training
- KV cache produces logits identical to a single full forward pass
- trainer: loss decreases; vision accuracy improves
- judge refuses to accept unevidenced work even when the model says "accept"
- policy gate blocks a malicious agent's output mid-debate
- scope check rejects public hosts and honours an explicit allowlist

```bash
python3 -m pytest tests/ -q
```