"""Synthetic corpora.

We generate data programmatically so the whole pipeline is reproducible and
ships with zero downloads: a structured language corpus (so the LM can learn
real syntax, not noise) and a synthetic-shapes vision corpus (so the vision
tower has a learnable signal).
"""
from __future__ import annotations

import os
import random
from typing import Iterator, Optional

import torch

from forge.control.scope import Op
from forge.tokenizer import TOKENIZER

SECURITY_SNIPPETS = [
    "def check_permissions(user, resource):\n    return user.role in resource.allowed_roles\n",
    "def sanitize(sql):\n    return sql.replace(';', '').replace('--', '')\n",
    "policy: never log credentials, tokens, or session identifiers",
    "threat model: an attacker with network access can replay an old token",
    "scan the dependency tree for known CVEs before every release",
    "a password must be hashed with a salted slow KDF, never stored raw",
    "input validation happens at the trust boundary, not in the UI",
    "detection rule: alert when a single IP triggers 50 failed logins in 60s",
]

CODING_SNIPPETS = [
    "def add(a, b):\n    return a + b\n",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "for i in range(10):\n    print(i * i)\n",
    "class Parser:\n    def parse(self, text):\n        return text.split()\n",
    "def mean(xs):\n    return sum(xs) / len(xs)\n",
    "try:\n    value = int(raw)\nexcept ValueError:\n    value = 0\n",
]

AGENT_TEMPLATES = [
    '{{"agent": "{a}", "action": "propose", "artefact": "{art}"}}',
    '{{"agent": "{a}", "action": "critique", "target": "{a2}", "severity": "{sev}"}}',
    '{{"agent": "{a}", "action": "verify", "test": "{art}", "passed": true}}',
]

DOMAINS = ["security", "coding", "hacking", "vision", "architecture"]
ARTEFACTS = ["parser", "patch", "test-suite", "router", "encoder", "policy"]
SEVERITIES = ["low", "medium", "high"]


def _random_agent_line(rng: random.Random) -> str:
    return rng.choice(AGENT_TEMPLATES).format(
        a=rng.choice(DOMAINS),
        a2=rng.choice(DOMAINS),
        art=rng.choice(ARTEFACTS),
        sev=rng.choice(SEVERITIES),
    )


# ---------------------------------------------------------- plan grammar data

PLAN_TASKS = [
    "harden the input parser", "document the trust boundary",
    "record the threat model", "write the hardening notes",
    "capture the audit findings", "note the routing tradeoffs",
    "summarise the vision encoder risks", "record the review outcome",
    "draft the security policy", "log the validation rules",
    "document the failure modes", "note the detection rules",
    "review the token expiry logic", "record the capacity limits",
    "note the retry semantics", "document the rollback procedure",
    "capture the evidence chain", "note the permission model",
    "record the quarantine policy", "document the audit trail",
    "review the input validation", "note the rate limits",
    "document the escalation path", "record the trust ladder",
    "note the expert routing balance", "document the vision pipeline",
    "capture the failure injection plan", "note the static analysis rules",
    "record the scope boundaries", "document the verification steps",
]

# Body bullets are chosen by keyword so the content is *derivable* from the
# task.  Picking them at random would make the target unlearnable: a model
# cannot predict an arbitrary answer, and no amount of training fixes that.
BODY_BY_TOPIC = {
    "parser": ["- reject input you cannot parse", "- fail closed, not open"],
    "trust": ["- validate at the trust boundary, not in the UI",
              "- never log credentials"],
    "threat": ["- record the threat model before the patch",
               "- one change per review"],
    "hardening": ["- prefer constant-time comparison for tokens",
                  "- alert on repeated failures"],
    "audit": ["- hash-chain every action record",
              "- report the exact broken index"],
    "routing": ["- balance on dispatched token counts",
                "- penalise collapsed experts"],
    "vision": ["- pool patches into summary tokens",
               "- keep the encoder small enough to train"],
    "review": ["- demand evidence for every claim",
               "- block on high-severity findings"],
    "security": ["- treat every external field as hostile",
                 "- log the decision, not the secret"],
    "validation": ["- validate before the side effect",
                   "- one rule per input class"],
    "failure": ["- a partial unverified change is worse than none",
                "- unwind on verification failure"],
    "detection": ["- alert on a sustained pattern, not a single event",
                  "- tune for a low false-positive rate"],
    "token": ["- rotate on privilege change", "- expire short and reissue"],
    "capacity": ["- measure the knee before scaling",
                 "- two times context can cost three times the step"],
    "retry": ["- exponential backoff with jitter",
              "- make retried operations idempotent"],
    "rollback": ["- snapshot before mutating",
                 "- restore the prior bytes, never just delete"],
    "evidence": ["- attribute every claim to an agent",
                 "- cite the test or trace that proves it"],
    "permission": ["- deny by default", "- grant with a TTL and a budget"],
    "quarantine": ["- move rather than unlink",
                   "- keep the original path for restore"],
    "trail": ["- every entry covers the previous hash",
              "- anchor the head externally"],
    "limits": ["- rate limit at thirty requests per minute",
               "- allow only private ranges unless allowlisted"],
    "escalation": ["- low confidence forces approval",
                   "- trust is slow to earn and fast to lose"],
    "ladder": ["- promote on verified successes only",
               "- demote to dry run on one failure"],
    "balance": ["- shared experts carry the common patterns",
                "- use a scale-free balancing loss"],
    "pipeline": ["- keep the action channel separate from the debate",
                 "- record every fallback in the audit chain"],
    "injection": ["- inject a failure and check the rollback",
                  "- verify against the machine, not the claim"],
    "analysis": ["- parse, never execute",
                 "- flag hardcoded credentials"],
    "scope": ["- resolve real paths to defeat symlinks",
              "- reject traversal at the root"],
    "verification": ["- re-read the filesystem after acting",
                     "- never trust the model's claim of success"],
}

DEFERRED_TOPIC = ["- review the task", "- record the outcome"]

PLAN_SLUGS = [
    "harden-the-input-parser", "document-the-trust-boundary",
    "record-the-threat-model", "write-the-hardening-notes",
    "capture-the-audit-findings", "note-the-routing-tradeoffs",
    "summarise-the-encoder-risks", "record-the-review-outcome",
]


def slugify(text: str) -> str:
    """The exact transform the model must learn: lowercase, spaces to hyphens.

    Making this the ground truth is what turns plan generation from an
    arbitrary recall task into a learnable transformation.
    """
    out = []
    for ch in text.lower().strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_/":
            if out and out[-1] != "-":
                out.append("-")
    return "".join(out).strip("-")


def body_for(task: str) -> list[str]:
    """Pick body bullets by keyword, so the content follows from the task."""
    low = task.lower()
    for topic, bullets in BODY_BY_TOPIC.items():
        if topic in low:
            return list(bullets)
    return list(DEFERRED_TOPIC)


def grammar_prompt() -> str:
    """The one grammar prompt, imported lazily to avoid an import cycle."""
    from forge.agents.act import GRAMMAR_PROMPT

    return GRAMMAR_PROMPT


def make_instruct_example(task: str, max_steps: int = 3,
                          rng: Optional[random.Random] = None,
                          eos: str = "<eos>", grant=None) -> str:
    """One training example shaped exactly like the real inference call.

    This is the fix for the measured failure where the inference prompt
    format appeared **zero** times in training.  The channel calls the model
    as ``f"{system}\\n\\n{user}\\n\\n"``; if the training data is not in that
    exact shape, the model is asked to follow instructions it has never seen
    and falls back to whatever dominates the corpus.

    Two measured additions:

    * ``max_steps`` mirrors the channel's parameter, because the user turn
      states the step budget and the model should honour it.
    * Each example ends with an explicit ``<eos>`` special token.  Without a
      stop signal the model does not know the answer is finished and continues
      straight into the next example in the corpus.  A measured run produced a
      *correct* plan followed immediately by the system prompt again, which is
      the classic missing-terminator failure.
    """
    slug = slugify(task)
    steps = [f"goal: {slug}"]
    if grant is None or Op.MKDIR in grant:
        steps.append("mkdir docs")
        steps.append(f"write docs/{slug}.md <<<")
        steps.extend(body_for(task))
        steps.append(">>>")
    elif Op.WRITE in grant:
        steps.append(f"write {slug}.md <<<")
        steps.extend(body_for(task))
        steps.append(">>>")
    else:
        # Read-only machine: the only faithful plan is to read.
        steps.append(f"read docs/{slug}.md")
        steps.append("list docs")
    if rng is not None and rng.random() < 0.25 and (not grant or Op.READ in grant):
        steps.append(f"read docs/{slug}.md")
    plan = "\n".join(steps)

    # Build the user turn with the same function the channel uses, so the two
    # cannot drift.  Half the examples carry a judge rationale because the
    # channel sends one, and the model must be robust to both shapes.
    from forge.agents.act import plan_user_turn

    rationale = None
    if rng is not None and rng.random() < 0.5:
        rationale = "no high-severity finding outstanding"
    user = plan_user_turn(task, max_steps, rationale=rationale, grant=grant)
    return f"{grammar_prompt()}\n\n{user}\n\n{plan}{eos}"


def make_instruct_corpus(n: int = 400, seed: int = 0) -> str:
    """Many `[system][user][plan]` triples in the exact inference shape."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        task = rng.choice(PLAN_TASKS)
        examples.append(make_instruct_example(task, max_steps=3, rng=rng))
    return "\n\n".join(examples)


# The grants a capability-conditioned corpus varies over.  Deliberately
# includes both a mutating set and a read-only one, because a corpus that only
# ever shows "everything allowed" cannot teach a model to notice a restriction.
GRANT_SETS = [
    frozenset({Op.READ, Op.LIST, Op.STAT, Op.MKDIR, Op.WRITE}),
    frozenset({Op.READ, Op.LIST, Op.STAT, Op.WRITE}),
    frozenset({Op.READ, Op.LIST, Op.STAT}),
    frozenset({Op.READ, Op.LIST}),
]


def make_capability_corpus(n: int = 800, seed: int = 0) -> str:
    """Instruction examples where the user turn states the grant and the
    target plan respects it.

    This is the training half of the capability-conditioning experiment: the
    measured baseline is that the model proposes mutating operations at a 100%
    rate regardless of what it is allowed to do, so the corpus must contain
    examples where the grant genuinely constrains the answer.  Without the
    read-only examples the model never sees a restricted target and has
    nothing to learn from.
    """
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        task = rng.choice(PLAN_TASKS)
        grant = rng.choice(GRANT_SETS)
        examples.append(
            make_instruct_example(task, max_steps=3, rng=rng, grant=grant)
        )
    return "\n\n".join(examples)


def iter_plan_examples(n: int = 50, seed: int = 0):
    """Yield plan blocks alone, which a parser test can isolate."""
    rng = random.Random(seed)
    for _ in range(n):
        task = rng.choice(PLAN_TASKS)
        slug = slugify(task)
        body = "\n".join(body_for(task))
        yield (f"goal: {slug}\nmkdir docs\n"
               f"write docs/{slug}.md <<<\n{body}\n>>>")


def make_grammar_corpus(n: int = 400, seed: int = 0,
                        include_prompts: bool = True) -> str:
    """Deprecated shape kept for compatibility; prefer make_instruct_corpus.

    Kept because older tests and the tokenizer suite reference it, but the
    instruction-shaped corpus is what the pipeline actually needs.
    """
    if include_prompts:
        return make_instruct_corpus(n, seed)
    return "\n".join(iter_plan_examples(n, seed))


def make_corpus(n_snippets: int = 400, seed: int = 0,
                grammar_ratio: float = 0.6) -> str:
    """Instruction-shaped plan data plus code, prose and agent JSON.

    ``grammar_ratio`` defaults to 0.6 rather than 0.4 because a measured
    experiment showed the plan grammar is the thing that fails: at 40% with
    the wrong prompt format the model produced 0/3 parseable plans.  The
    instruction examples are duplicated on purpose, since repeating a
    transformation is how a small model learns it.
    """
    rng = random.Random(seed)
    parts: list[str] = []
    n_grammar = max(1, int(n_snippets * grammar_ratio))
    for i in range(n_snippets - n_grammar):
        kind = i % 3
        if kind == 0:
            parts.append(rng.choice(CODING_SNIPPETS))
        elif kind == 1:
            parts.append(rng.choice(SECURITY_SNIPPETS))
        else:
            parts.append("\n".join(_random_agent_line(rng) for _ in range(4)))
    # Weighted heavily: the instruction shape is what the pipeline needs.
    parts.extend([make_instruct_corpus(n_grammar, seed=seed + 1)] * 3)
    rng.shuffle(parts)
    return "\n\n".join(parts)


def corpus_to_tensor(text: str, tokenizer=None) -> torch.Tensor:
    """Encode a corpus with the given tokenizer.

    The tokenizer must be passed explicitly.  It previously defaulted to the
    module-level byte tokenizer, so a model built with a BPE vocabulary was
    trained on byte ids: half its embedding rows went untouched and the ids it
    did see were meaningless. Training still ran and the loss still fell,
    which is exactly why this went unnoticed until generation was checked.
    """
    tok = tokenizer or TOKENIZER
    ids = tok.encode(text, add_bos=False, add_eos=False)
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random contiguous windows; inputs and shifted targets."""
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError("corpus too short for the requested seq_len")
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


def iter_batches(
    data: torch.Tensor, batch_size: int, seq_len: int,
    steps: int, device: str = "cpu",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(steps):
        yield sample_batch(data, batch_size, seq_len, device)


# ---------------------------------------------------------------- vision data

def make_shapes(
    n: int,
    size: int = 32,
    channels: int = 3,
    n_classes: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Images of a centred square whose *brightness* encodes the class.

    Brightness is used because it is trivially learnable (the model should
    reach near-perfect accuracy on it), which makes the vision tower's health
    immediately visible.  Shape variety is added so the encoder cannot just
    read the mean pixel value -- it still has to pool over patches.
    """
    g = torch.Generator().manual_seed(seed)
    imgs = torch.zeros(n, channels, size, size)
    labels = torch.randint(0, n_classes, (n,), generator=g)

    for i in range(n):
        level = (labels[i].item() + 1) / n_classes
        side = int(size * (0.3 + 0.3 * torch.rand(1, generator=g).item()))
        top = torch.randint(0, size - side, (1,), generator=g).item()
        left = torch.randint(0, size - side, (1,), generator=g).item()
        shade = level * torch.ones(side, side)
        imgs[i, :, top:top + side, left:left + side] = shade
    return imgs, labels


class ShapeDataset:
    """On-the-fly dataset so memory stays flat regardless of n."""

    def __init__(self, n: int = 1000, size: int = 32, seed: int = 0) -> None:
        self.n = n
        self.size = size
        self.seed = seed
        self._imgs, self._labels = make_shapes(n, size=size, seed=seed)

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.n, (batch_size,))
        return self._imgs[idx], self._labels[idx]

    def __len__(self) -> int:
        return self.n