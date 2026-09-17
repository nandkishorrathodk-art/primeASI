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


def make_corpus(n_snippets: int = 400, seed: int = 0) -> str:
    """Structured text: code + prose + agent-JSON lines."""
    rng = random.Random(seed)
    parts: list[str] = []
    for i in range(n_snippets):
        kind = i % 3
        if kind == 0:
            parts.append(rng.choice(CODING_SNIPPETS))
        elif kind == 1:
            parts.append(rng.choice(SECURITY_SNIPPETS))
        else:
            parts.append("\n".join(_random_agent_line(rng) for _ in range(4)))
    return "\n".join(parts)


def corpus_to_tensor(text: str) -> torch.Tensor:
    ids = TOKENIZER.encode(text, add_bos=False, add_eos=False)
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