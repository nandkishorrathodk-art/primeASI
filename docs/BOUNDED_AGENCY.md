# Bounded Agency: my answer to "make it control the whole computer"

You asked me for my own idea. Here it is, with the honest framing first.

## What I am NOT claiming

**Nobody knows how to build ASI.** There is no recipe, no architecture, no
scaling law that gets you there. If I told you "here is how you turn this into
ASI", I would be lying to you, and you would waste months finding out.

**"Puppet mode" is not a missing feature, it is a category error.** A system
that can do *anything* cannot be supervised by construction. Full control and
safety are not a tradeoff you tune; they are the same dial.

So the real question is not *"how do I make it all-powerful"*. It is:

> **How do I give a system as much real control as possible while keeping
> every action recoverable and attributable?**

That question has an answer. That answer is engineering, not fiction, and it
is implemented in this repository.

## The idea: bounded agency

Four principles. Each one is enforced in code, not in a prompt.

### 1. Capability, not filtering

A model told "do not delete my files" can be talked out of it. A model that
holds **no delete capability** cannot issue one - there is nothing to persuade.
Safety is a capability-availability problem, not a content-filtering problem.

`forge/control/scope.py` - typed action space, deny by default, capability
grants with TTL and call budgets, and a `PathGuard` that resolves real paths
so `..` traversal and symlink escapes fail at the root.

Denials carry a **constructive alternative**. An agent that oversteps is
redirected, not just stopped. A wall teaches nothing; a wall with a door in it
teaches the boundary.

### 2. Reversibility by construction

The reason autonomous systems are scary is not that they make mistakes. It is
that mistakes are **permanent**. Remove permanence and the entire risk profile
changes.

`forge/control/actions.py` - every mutating action returns its own inverse:

- `write` snapshots prior content before replacing
- `append` records the exact byte offset to truncate back to
- `delete` **moves to quarantine**; nothing in this codebase ever unlinks
- `mkdir` records removal

An action that cannot construct its inverse is refused rather than performed.
`Transaction` groups steps so a multi-step plan either fully lands or fully
unwinds. `undo_all()` returns the machine to its prior state.

### 3. Autonomy as a measurement, not a setting

Fixed autonomy levels throw away calibration information the system generates
every time it acts. So make autonomy a *measurement*:

```
dry_run  ->  approval  ->  auto
```

`forge/control/trust.py` - each operation class climbs one rung after
`promote_after` consecutive **verified** successes, and drops back to
`dry_run` on a single failure. Trust is slow to earn and instant to lose.

Two details that matter more than the ladder itself:

- **Per-class, not global.** Writing a new file is not the same risk as
  deleting one. Each earns its own level.
- **Uncertainty beats history.** A confidence below the floor forces an action
  down to `approval` no matter how good the record is. A confident system that
  is sometimes wrong is more dangerous than a hesitant one.

**The bootstrap problem, stated honestly:** a class at `dry_run` never
executes, so it can never earn promotion. Autonomy must therefore be *seeded*
by a human (`trust.seed(...)`). That is the correct place for the decision to
live - an agent must not be able to grant itself permission by repeated
attempts. There is a test asserting exactly this.

### 4. Verify against the machine, not the model's claim

This is the failure mode that makes agentic systems untrustworthy: the model
says it did the thing, and everything downstream believes it.

`forge/control/kernel.py` re-reads the real filesystem after acting. A step
counts as successful only if the machine agrees the expected state is now
true. "The model said it worked" is never evidence.

## The loop

```
perceive -> plan -> simulate -> approve -> act -> verify -> journal
            ^                                            |
            +------------------ adapt -------------------+
```

**Simulate before acting.** `simulate()` checks scope, trust, preconditions,
and risk surface against the world digest - without touching anything. A plan
that would escape the sandbox, exceed trust, or touch a
name-suggests-secrets path is refused before the first write. Most damage is
prevented here, by noticing earlier rather than filtering harder.

## The bridge: how a model actually drives the machine

`forge/control/bridge.py` converts model text into a typed `Plan`. The rule:

> **The model's output is data, never instructions.**
> Nothing the model writes is executed, evaluated, or interpreted as code.

The grammar has no production for code. `exec(...)`, `eval(...)`,
`__import__(...)`, `$(...)`, `;`, `|` - none of them can be expressed, so none
of them can be executed. Anything that does not fit the grammar is a parse
failure, and a parse failure means nothing happens.

That gives **two independent defense layers**, and the demo proves it:

```
attack             | model acted | stopped by
-------------------+-------------+------------------
injection via code | no          | grammar (layer 1)
absolute escape    | no          | kernel  (layer 2)
shell metachar     | no          | grammar (layer 1)
secrets file       | no          | kernel  (layer 2)
```

## And honest perception

`forge/control/world.py` - "observe the whole computer" is a **compression**
problem, since a filesystem cannot fit in a context window. The digest is
hierarchical, budgeted, ranked by relevance (changed files before unchanged,
source before cache), skips `.git`/`__pycache__`/`node_modules`, reads only
metadata never contents, and **reports its own truncation**. A controller that
does not know what it could not see is worse than one that sees less.

## Tamper-evident, not tamper-proof

`forge/control/audit.py` hash-chains every action record. Modifying any
historical entry breaks the chain, and `verify()` reports the exact index
where the break starts.

**The honest limit:** this *detects* tampering, it does not *prevent* it.
Someone who can rewrite the whole file can rebuild a consistent chain. That is
why the chain head is printed on every run and can be anchored externally.
Detection plus external anchoring is the achievable guarantee; anyone claiming
more is overselling.

## Run it

```bash
python3 -m forge.control_demo      # 11 sections, all on a real filesystem
python3 -m pytest tests/ -q        # 95 tests, ~3.4s, no mocks
```

The demo shows, live: bounded perception, a verified plan, a plan refused
before touching anything, symlink escape blocked, a secrets file refused,
delete-and-restore from quarantine, full rollback, audit tamper detection
with the exact break index, autonomy earned over successive runs, an
adversarial model stopped by two independent layers, and the big red button.

## What this is, in one line

Not ASI. A **control architecture** that is honest about its own limits, where
capability is granted rather than assumed, every action is reversible,
autonomy is earned from measurement, and nothing the model produces is ever
executed as code.

That is the part of "controlling the whole computer" that is real, and it is
the foundation anything larger would have to be built on.

## Where a next step could go

Directions that are genuinely open, and not fantasy:

1. **A learned world model for simulation.** Right now `simulate()` uses
   rules. A model that predicts the *consequences* of a plan would let the
   kernel refuse bad plans it has never explicitly seen blocked.
2. **Process-level control with the same discipline.** The same capability +
   rollback + audit structure applied to services and containers.
3. **Formal invariants.** Express the safety properties as checkable
   assertions over the audit chain, so a run can be proven to have respected
   its bounds.
4. **Learned trust calibration.** Replace the fixed streak rule with a
   calibration model over (task type, uncertainty, historical accuracy).