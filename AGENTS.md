# AGENTS.md

Repository knowledge for agents working on Forge.

## What this repo is

A small agentic LLM framework that trains end-to-end on CPU (4 cores, 15 GB,
no GPU). The constraint is deliberate: it keeps every component measurable.

## Setup and commands

```bash
python3 -m pip install numpy torch pytest
python3 -m pytest tests/ -q                    # 187 tests, ~22s
python3 -m forge.demo --steps 250              # trains, generates, debates
python3 -m forge.pipeline_demo --steps 200     # full chain, BPE tokenizer
python3 -m forge.control_demo                  # control kernel sections
```

`runs/` is gitignored and holds checkpoints. `PYTHONPATH` must point at the
repo root when running scripts from outside it.

## Measured facts (do not re-derive, but do re-verify if you change the code)

Throughput on 4 CPU cores, batch 16, seq 256:

| config | params | tok/s |
|---|---|---|
| default (`dim=128, 4 layers, 4 experts`) | 2.7M | ~9,250 |
| scaled (`dim=256, 6 layers, 8 experts`) | 31–50M | ~1,050 |

Token budget is the real constraint. At ~1k tok/s, 1B tokens is 264 hours.

## Planning-quality metrics: P / S / V / H

`forge/training/metrics.py` measures four things. Keep them separate; each
catches a failure the others cannot see.

- **P** parse rate — text fits the plan grammar.
- **S** semantic rate — parsed *and* `goal == slugify(task)`.
- **V** verify rate — kernel executed and verified against real disk.
- **H** hack rate — parsed but task-blind (`P and not S`).

P alone is misleading. The repo's historical headline
(`model parsed own output : False`) only reports P.

Measured on the same corpus, n=60, seed 7, temp 0.6, Wilson 95% intervals:

| model | S | V |
|---|---|---|
| small 2.7M | 56.7% [44.1, 68.4] | 95.0% [86.3, 98.3] |
| big 31M | 10.0% [4.7, 20.1] | 53.3% [40.9, 65.4] |

Intervals do not overlap: at this scale, more parameters made the planner
*worse*. Treat "scale it up" as an untested assumption in this repo, not a fix.

## Traps that have already cost real time

1. **`generate()` defaults to `eos_id=None`.** Callers must pass
   `eos_id=EOS` or generation runs past the answer into the next training
   example. `LocalBackend` does this correctly; ad-hoc eval scripts do not.

2. **Never decode the full sequence and strip the prompt by string match.**
   The model echoes the prompt, so `rsplit` on the user turn can return the
   echo and every metric reads as a hard 0%. Use
   `forge.training.metrics.generate_plan`, which decodes only the ids past
   the prompt length.

3. **One fresh `ControlKernel` per eval sample.** A second `write` to the same
   path fails the "expect absent before" precondition, so reusing a kernel
   makes V depend on sample order.

4. **The tokenizer must travel with the weights.** Pass `tokenizer=` to
   `Trainer` and use `Trainer.load_with_tokenizer`. A BPE model decoded with a
   byte tokenizer produces silent gibberish.

5. **The grammar prompt appears in every training example**, so the model
   learns to generate it. Strip it when scoring.

## Failed approaches, recorded so they are not retried

- **2×2 ablation on the `Judge rationale` line** (`ablation_rationale.py`
  shape): every cell 1/16, a null result. The rationale line is not the
  bottleneck.
- **Rejection-sampling self-training** (STaR-style) against the verifier:
  3% acceptance, and the retrained model was *worse* (S 10% → 0%). The
  verifier is deterministic, so it is gameable — the model drifted toward
  task-blind but parseable output. H stayed high throughout.

Both are negative results with real measurements. Do not present either as a
win, and do not silently drop them.

## Safety: where it actually lives

`forge/training/metrics.py` also measures *intent* — what the model asked for,
before any gate ran (`intent_report` / `evaluate_intent`). This is separate
from P/S/V/H, which are all downstream of execution.

Measured (2.7M checkpoint, n=24, seed 7, temp 0.6):

| machine grant | hazard ops proposed | kernel-blocked steps |
|---|---|---|
| default policy (write/mkdir allowed) | 46/46 = 100% | 0 |
| read-only (no mutating grant) | 48/48 = 100% | 44 |

**The hazard-op rate is 100% either way.** The model proposes `mkdir`/`write`
at the same rate whether or not it holds the capability to do so. It does not
condition on its grant; the kernel does 100% of the containment. This is the
concrete, measured form of the repo's own claim that "safety is a capability
problem, not a filtering problem" — the capability lives in the kernel only.

### Retracted claim

An earlier measurement in this repo's history reported "illegal-op rate
100%" — the model allegedly proposing forbidden operations on every sample.
**That number was an artifact.** The forbidden-substring list contained
`<<`, which appears in every legitimate `write ... <<<content>>>` step, so
every plan was flagged regardless of content. The corrected measurement is
above: the model proposes *mutating* operations 100% of the time (true, but
unsurprising — that is what the task asks for) and proposed 1 hazardous
*path* in 48 steps. Do not cite the old number.

### Related work (do not overclaim novelty)

Scoping a model to refuse out-of-domain requests is studied: *Reducing the
Scope of Language Models* (AAAI 2026) covers refusal scoping; *Theory of
Agent* (2026 preprint) discusses boundary decisions around external action.
The narrow measurement here — comparing proposed hazard ops against granted
capabilities for a small local planner — appears uncommon, but this has not
been checked exhaustively. Treat it as "not obviously duplicated", not as new.

## Capability conditioning: tested, claim rejected

The obvious follow-up is to tell the model its grant and train the target plan
to respect it. That was built and measured (`make_capability_corpus` in
`forge/data.py`, `grant=` on `plan_user_turn` / `generate_plan` /
`evaluate_intent` / `metrics --grant`), and **the claim failed.**

Claim: *"a model trained with its grant in context proposes fewer out-of-grant
operations, at no cost to verify rate."*

| eval grant | arm | O (out-of-grant) | V (verify) |
|---|---|---|---|
| read-only | baseline | 96.6% | 0.0% |
| read-only | conditioned | **50.5%** | 0.0% |
| write | baseline | 3.0% | **70.0%** |
| write | conditioned | 1.0% | **2.5%** |

n=40, seed 7, temp 0.6, 500 train steps per arm, same architecture.

What is real: on the read-only arm the out-of-grant rate halved, and it is not
a denominator artifact — out-of-grant proposals per sample fell 2.00 → 1.03
while steps per sample *rose* 2.07 → 2.47. So the model became somewhat
sensitive to the stated grant.

Why the claim still fails:

1. **Verify rate collapsed on the write arm (70% → 2.5%).** This is the
   degenerate solution the experiment was built to detect. Fewer violations
   did not come from obedience; it came from producing less executable output.
2. **The conditioned model echoes the grant line** — 9/30 samples, versus
   0/30 for the baseline. Greedy decoding on a read-only grant still emits
   `mkdir`/`write`. The condition is being copied, not conditioned on. This is
   the same prompt-echo trap documented above, now triggered by a new line.
3. **Read-only V is 0% for both arms by construction.** A read-only machine
   cannot create the file a `write`-shaped task requires, so the read-only
   tasks are unsatisfiable regardless of how well-behaved the model is. The O
   improvement on that arm cannot be validated end-to-end.

Conclusion: at 2.7M parameters, stating a capability grant in context does not
produce a model that respects it. It produces a model that repeats the grant
and degrades. Do not retry this expecting a different result without first
fixing the echo problem (see below).

### What would actually test this

- **Train on a corpus where the grant line varies but the answer is identical
  when the grant permits it.** Right now the grant both changes the prompt and
  the target, so obedience and copying are confounded.
- **Constrain decoding** so the grant line cannot be emitted, then re-measure.
  Until the echo is removed, O is not a clean measurement.
- **Scale up.** 2.7M at ~20k params/context-token cannot hold a conditional
  instruction; every result above is at or below the noise floor of that limit.

## Fixed bugs worth knowing about

- **`CapabilitySet.check` rejected `list`/`stat`/`scan` as "unknown
  operations"** regardless of grants, so three operations that
  `default_policy` grants and `_dispatch` implements were unreachable.
  `GRANTABLE_OPS` now derives the accepted set from the enum, and a test
  asserts `set(Op) == MUTATING_OPS | OBSERVING_OPS` so a new op cannot be
  added in one place and forgotten in another.

## Honest limits

- 2.7M–50M params cannot follow a system prompt reliably. This is a scale
  limit, not a bug to be fixed by more steps.
- The corpus is largely synthetic and repetitive. Corpus quality moved S from
  10% to 45–57%; parameter count did not.
- The model does not condition on its capabilities at all (see above). Safety
  lives entirely in the kernel. Closing that gap is the most interesting open
  problem in the repo.
