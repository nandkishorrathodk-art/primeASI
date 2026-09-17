# Forge

A small agentic LLM framework you can actually train on a CPU.

Everything in this repo runs end-to-end on 4 cores and 15 GB of RAM with no
GPU. That constraint is a feature: it forces every component to be real,
measurable, and inspectable rather than a thin wrapper around a datacentre.

## What is actually in here

| Component | Status | Notes |
|---|---|---|
| Byte tokenizer | works | dependency-free baseline, VOCAB 260 |
| BPE tokenizer | works | trained on-corpus, VOCAB 1024, 1 token for `mkdir` |
| MoE transformer LLM | trains | shared + routed experts, load balancing |
| Vision tower | trains | patch encoder, reaches ~95% on synthetic task |
| Multi-agent debate loop | works | 5 sub-agents + judge |
| Security analyzer | works | static AST + regex, never executes code |
| Lab-scoped probe | works | private ranges + explicit allowlist only |
| Self-distillation (EMA) | works | no external dependency |
| API-teacher distillation | opt-in | only if you hold the keys |
| Reports | works | text + self-contained HTML |
| **Control kernel** | works | capability scoping, reversible actions, earned autonomy |
| **Model-to-machine bridge** | works | model text -> typed plan, never executed as code |
| **Action channel** | works | debate -> judge -> one verified machine action |
| **End-to-end pipeline** | works | `python3 -m forge.pipeline_demo` |

## Quick start

```bash
python3 -m pip install numpy torch
python3 -m pytest tests/ -q           # 136 tests, ~5.9s
python3 -m forge.demo --steps 250     # trains, generates, debates, reports
python3 -m forge.control_demo         # 11 control sections on a real filesystem
python3 -m forge.pipeline_demo --steps 150   # the whole chain, end to end
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

## Controlling a machine (the "bounded agency" layer)

`forge/control/` answers a different question from the rest of the repo: not
"can it think", but "how does a model operate a computer without becoming a
liability". Full write-up in [docs/BOUNDED_AGENCY.md](docs/BOUNDED_AGENCY.md).

```bash
python3 -m forge.control_demo
```

Four enforced principles:

1. **Capability, not filtering** - no delete capability means no delete is
   expressible, whatever the model says.
2. **Reversibility by construction** - every mutation records its own inverse;
   even `delete` only quarantines. Nothing here ever unlinks a file.
3. **Earned autonomy** - `dry_run -> approval -> auto`, per operation class,
   promoted on verified successes and dropped to `dry_run` on one failure.
4. **Verify against the machine** - post-conditions are checked on the real
   filesystem, never taken from the model's claim.

Model output is parsed into a typed plan and is **never executed as code**.
That yields two independent layers, which the demo shows biting:

```
attack             | model acted | stopped by
-------------------+-------------+------------------
injection via code | no          | grammar (layer 1)
absolute escape    | no          | kernel  (layer 2)
shell metachar     | no          | grammar (layer 1)
secrets file       | no          | kernel  (layer 2)
```

Honest limits, stated in the same document: the audit chain is tamper
*evident*, not tamper *proof*; and autonomy has to be **seeded** by a human,
because a class at `dry_run` can never earn promotion by itself.

## The whole chain, connected

For most of this project's life there were three islands that never spoke: a
model that trained, agents that debated, and a kernel that could operate a
filesystem. `forge/agents/act.py` is the wire between them, and
`forge.pipeline_demo` runs the result.

```
ForgeLM trains
     |
     v
5 sub-agents debate  ->  policy gate  ->  judge verdict
                                               |
                          (only ACCEPT has authority over the machine)
                                               v
                          plan request in a strict grammar
                                               |
                     model output parses? ----+---- no ---> deterministic
                              |  yes                            fallback,
                              v                                recorded in
                     simulate: scope, trust,                    the audit
                     preconditions, risk surface                chain
                              |
                              v
                     execute -> verify against real disk
                              |
                    verification failed? -> auto-rollback
                              |
                              v
                     hash-chained audit, tamper-evident
```

Run it:

```bash
python3 -m forge.pipeline_demo --steps 150
```

### The honest headline

**A 2.6M-parameter byte-level model does not emit parseable plans.** The demo
prints `model parsed own output : False` and `used fallback : True`, rather
than pretending otherwise. This is expected at this scale and it is the single
most useful signal the pipeline produces: it tells you exactly what to work on
next.

The pipeline handles it the way it should. An unparseable proposal still
produces a *deterministic, auditable* action instead of nothing, and the
substitution is recorded in the audit chain as `fallback` -- visible, never
silent. A pipeline that quietly swaps in rules and reports success is lying
about what ran.

### Why the fallback is not a model call

Calling a backend and then ignoring its reply is theatre. The fallback is a
pure function from task to plan text (`default_fallback_plan`), so it is
deterministic and reproducible. There is no backend involved.

### Nothing acts before the debate concludes

Only an `ACCEPT` ruling has authority over the machine. A `REVISE` or
`REJECT` cannot reach the kernel at all -- there is a test asserting that a
non-accepted run leaves the ledger empty. The judge approves the *work*; it
does not approve the *target*, which is why a well-formed debate naming a
secrets file still gets refused by the kernel on its own authority.

## BPE tokenizer, larger context, and what the model still cannot do

Three measured improvements and one honest failure. All numbers reproduced on
4 CPU cores.

### Measured, not assumed

```
                                     before      after
vocabulary                           260         1024 (trained BPE)
tokens for 'mkdir'                   5           1
grammar prompt size                  418 tokens  83 tokens
model context                        128         256
throughput @256 context              -           9,009 tok/s
```

The 128-token context was the worst of these: the grammar prompt did not fit,
so the model was being asked to follow instructions it physically could not
see. The byte tokenizer cost 1.00 tokens per character, so the instruction
block alone exceeded the window.

### The bug that mattered most

`corpus_to_tensor` defaulted to the module-level byte tokenizer, and the
trainer never passed its own. A model built with a 1024-token BPE vocabulary
was therefore trained on **byte ids**. Training ran, the loss fell, nothing
raised an error -- but the embedding table was half unused and the ids were
meaningless. It only became visible when generation was actually checked
rather than when training was watched.

Fix: the tokenizer is passed explicitly, and it is now **stored in the
checkpoint** so a model can never be decoded with the wrong vocabulary.
Decoding a BPE model with a byte tokenizer produces gibberish silently, which
is the worst possible failure mode.

### The honest failure

With BPE, 256-token context, and 40% of the corpus teaching the plan grammar:
**the model still produces 0/3 parseable plans.** Measured diagnostics:

```
loss on prompt region : 10.79      the model cannot even predict the prompt
loss on plan region   : 4.43       and does not know the grammar
corpus duplicate ratio: 91.2%      271 unique lines out of 3,090
grammar prompt in training data: 0 occurrences
```

Three separate problems, in order of severity:

1. **The inference prompt format appears zero times in training.** The model
   is given `Output only plan lines:` and has never seen that string. It
   cannot condition on instructions it has never seen, so it falls back to
   whatever dominates the corpus -- which is agent-JSON:
   `{"agent": "architecture", "action": "critique", ...}` repeated until the
   token budget runs out. That is exactly the observed output.

2. **The corpus is 91% duplicate.** 271 unique lines repeated to 3,090. A
   language model trained on near-duplicate text learns to regurgitate, not
   to generalise, and any held-out compression number is meaningless.

3. **The model is far too small to follow a system prompt.** 2.7M parameters
   at 20k parameters per context token cannot hold instruction-following
   behaviour. This is a scale limit, not a bug, and it is the honest ceiling
   on what this repository can demonstrate.

The pipeline handles all three correctly: they are limitations of the model,
not of the control layer. The action channel still produces a deterministic,
audited action, and the substitution is recorded. But the headline stays
`model parsed own output : False`.

### What would actually fix it

Not more parameters. In measured order of impact:

1. **Train on the exact inference format.** Every training example should be
   `[system prompt][user turn][plan]`, the same shape the channel sends.
2. **A corpus that is not 91% duplicates**, at least tens of MB.
3. **More compute than a 4-core CPU**, which is the honest end of what this
   environment can offer.

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
  control/
    scope.py          capability tokens, PathGuard, deny by default
    actions.py        reversible actions, transactions, quarantine
    audit.py          hash-chained tamper-evident log
    trust.py          earned autonomy: dry_run -> approval -> auto
    world.py          budgeted, self-truncation-reporting perception
    kernel.py         perceive-plan-simulate-approve-act-verify
    bridge.py         model text -> typed plan (never executed as code)
  agents/
    act.py            ActionChannel: debate -> judge -> verified action
    local.py          wire a trained checkpoint in as a backend
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

136 tests, ~5.9s, no mocks - everything exercises real code paths:

- tokenizer round-trip including multibyte UTF-8
- MoE: shared experts actually contribute, expert-choice routing is balanced,
  token-choice load sums to `top_k`, balance loss grows under collapse,
  no expert starves during real training
- KV cache produces logits identical to a single full forward pass
- trainer: loss decreases; vision accuracy improves
- judge refuses to accept unevidenced work even when the model says "accept"
- policy gate blocks a malicious agent's output mid-debate
- scope check rejects public hosts and honours an explicit allowlist

Plus the control layer: symlink escape blocked, an empty capability set
denies everything, expired grants denied, audit tampering detected with the
exact break index, transactions roll back fully, quarantine restores bytes
intact, trust cannot bootstrap itself, low confidence escalates even at auto,
an adversarial model's `exec(...)` output cannot parse, only an ACCEPT ruling
reaches the machine, auto-rollback unwinds just the failed run and not earlier
work, and a debate that approves work naming a secrets file is still refused.

```bash
python3 -m pytest tests/ -q
```