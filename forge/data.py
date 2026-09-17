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
]

PLAN_SLUGS = [
    "harden-the-input-parser", "document-the-trust-boundary",
    "record-the-threat-model", "write-the-hardening-notes",
    "capture-the-audit-findings", "note-the-routing-tradeoffs",
    "summarise-the-encoder-risks", "record-the-review-outcome",
]

PLAN_BODIES = [
    "- validate at the trust boundary, not in the UI\n- never log credentials",
    "- reject input you cannot parse\n- fail closed, not open",
    "- hash passwords with a salted slow KDF\n- rotate tokens on privilege change",
    "- treat every external field as hostile\n- log the decision, not the secret",
    "- prefer constant-time comparison for tokens\n- alert on repeated failures",
    "- record the threat model before the patch\n- one change per review",
]


def make_grammar_corpus(n: int = 400, seed: int = 0,
                        include_prompts: bool = True) -> str:
    """Examples in exactly the grammar the action channel asks the model for.

    This exists because of a measured gap: the original corpus contained
    **zero** occurrences of ``mkdir``, ``write``, ``goal:``, ``<<<`` or
    ``>>>``.  The pipeline was asking the model to emit a language it had
    never seen a single example of.  No amount of parameters or tokenizer
    work fixes that; the examples have to be present.

    The prompt comes *before* the plan, matching how the channel actually
    calls the model.  Reversing that order would train the model to emit a
    question after its answer.

    ``include_prompts=False`` yields plan blocks alone, which is what the
    parser tests need: a prompt line is not a plan step and must not be
    expected to parse as one.
    """
    rng = random.Random(seed)
    blocks: list[str] = []
    for _ in range(n):
        task = rng.choice(PLAN_TASKS)
        slug = rng.choice(PLAN_SLUGS)
        body = rng.choice(PLAN_BODIES)
        pad = " " * rng.choice([0, 0, 2, 4])
        steps = [
            "goal: " + slug,
            "mkdir docs",
            f"write docs/{slug}.md <<<",
            body,
            ">>>",
        ]
        if rng.random() < 0.3:
            steps.append(f"read docs/{slug}.md")
        if rng.random() < 0.2:
            steps.append("run_tests run_tests")
        plan = "\n".join(pad + s for s in steps)
        if include_prompts:
            blocks.append(f"Task: {task}\nProduce at most {len(steps)} steps.")
        blocks.append(plan)
    return "\n".join(blocks)


def iter_plan_examples(n: int = 50, seed: int = 0):
    """Yield individual plan blocks that should each parse on their own."""
    slug = "x"
    rng = random.Random(seed)
    for _ in range(n):
        s = rng.choice(PLAN_SLUGS)
        body = rng.choice(PLAN_BODIES)
        yield (
            "goal: " + s + "\n"
            "mkdir docs\n"
            f"write docs/{s}.md <<<\n{body}\n>>>"
        )


def make_corpus(n_snippets: int = 400, seed: int = 0,
                grammar_ratio: float = 0.4) -> str:
    """Structured text: code, prose, agent JSON, and plan-grammar examples.

    ``grammar_ratio`` controls how much of the corpus teaches the plan
    grammar the action channel depends on.  It defaults high because that
    grammar is what the end-to-end pipeline needs the model to emit.
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
    parts.append(make_grammar_corpus(n_grammar, seed=seed + 1))
    rng.shuffle(parts)
    return "\n".join(parts)


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