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

## Honest limits

- 2.7M–50M params cannot follow a system prompt reliably. This is a scale
  limit, not a bug to be fixed by more steps.
- The corpus is largely synthetic and repetitive. Corpus quality moved S from
  10% to 45–57%; parameter count did not.
- `I` (illegal-op proposal rate) measured 100% on the default model: it
  proposed forbidden operations every time and was blocked every time. Safety
  currently lives entirely in the kernel, not in the model. This is the most
  interesting open problem in the repo.
